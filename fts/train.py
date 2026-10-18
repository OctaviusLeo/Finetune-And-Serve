"""LoRA fine-tuning Runs (Req 4.1, 4.2, 4.7).

A Run fine-tunes the bf16 Base_Model with a LoRA Adapter (base weights frozen)
using the fixed settings in config/train.yaml and the determinism settings in
fts.lora. Each training example is the compact prompt plus the gold JSON as the
assistant turn; the prompt and the assistant part are tokenized separately so
the loss covers exactly the assistant tokens and the prompt boundary matches
inference. Data order is a seeded shuffle per epoch. The loss is logged every
``logging_steps`` optimizer steps, only the final Adapter is saved (to
``~/fts-artifacts/adapters/<run_id>/``), the Validation F1 is computed with
fts.infer, and results/runs/<run_id>.json records the loss curve, the full
config, the seed and the 40-character commit.

The trainer reads only ``splits.train`` and ``splits.validation``; it never
touches the Test_Split.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from fts import infer, prompts, repro, settings
from fts.data import Example, Splits
from fts.eval_core import bootstrap_ci
from fts.lora import LoraSpec, apply_lora, count_parameters, limit_cuda_memory, set_determinism

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get("FTS_ARTIFACTS", Path.home() / "fts-artifacts"))
IGNORE = -100


@dataclass(frozen=True)
class TrainConfig:
    name: str  # "A" | "B" | "C"
    lora_r: int
    lora_alpha: int
    learning_rate: float
    epochs: float  # 2
    max_seq_len: int  # 256
    per_device_batch_size: int  # 8
    grad_accum: int  # 2
    seed: int


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    adapter_path: str  # relative to ~/fts-artifacts
    loss_curve: tuple[tuple[int, float], ...]  # (optimizer step, training loss)
    config: TrainConfig
    val_metric: float  # Validation field-level F1
    commit: str  # full 40-char hash

    def to_dict(self) -> dict:
        d = asdict(self)
        d["loss_curve"] = [list(p) for p in self.loss_curve]
        return d


def run_id(config: TrainConfig) -> str:
    return f"{config.name}-seed{config.seed}"


def config_from(base: settings.TrainSettings, name: str, lora_r: int, lora_alpha: int, learning_rate: float, seed: int) -> TrainConfig:
    return TrainConfig(name=name, lora_r=lora_r, lora_alpha=lora_alpha, learning_rate=learning_rate, epochs=base.epochs,
                       max_seq_len=base.max_seq_len, per_device_batch_size=base.per_device_batch_size, grad_accum=base.grad_accum, seed=seed)


def encode_example(tokenizer: Any, templates: prompts.Templates, example: Example, max_seq_len: int) -> tuple[list[int], list[int]]:
    """(input_ids, labels) with labels -100 on the prompt, truncated to ``max_seq_len``."""
    messages = prompts.training_messages(templates, example)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    if not full.startswith(prompt):
        raise ValueError(f"chat template does not extend the prompt for example {example.example_id}")
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"])
    ids = (prompt_ids + target_ids)[:max_seq_len]
    labels = ([IGNORE] * len(prompt_ids) + target_ids)[:max_seq_len]
    return ids, labels


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    import torch

    gen = torch.Generator().manual_seed(seed * 1000 + epoch)
    return torch.randperm(n, generator=gen).tolist()


def collate(batch: Sequence[tuple[list[int], list[int]]], pad_id: int) -> dict:
    """Right-pad a micro-batch to its longest example."""
    import torch

    width = max(len(ids) for ids, _ in batch)
    input_ids = [ids + [pad_id] * (width - len(ids)) for ids, _ in batch]
    labels = [lab + [IGNORE] * (width - len(lab)) for _, lab in batch]
    mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch]
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def micro_count(n: int, batch_size: int, epochs: float) -> int:
    """Micro-batches in a Run: whole epochs of ceil(n / batch) each, a fractional last epoch cut short."""
    whole = int(epochs)
    per_epoch = math.ceil(n / batch_size)
    return whole * per_epoch + math.ceil((epochs - whole) * n / batch_size)


def micro_batches(encoded: Sequence[tuple[list[int], list[int]]], batch_size: int, seed: int, epochs: float) -> Iterator[tuple[int, list]]:
    """Yield (epoch, micro-batch) in seeded order, ``micro_count`` of them."""
    n = len(encoded)
    total = micro_count(n, batch_size, epochs)
    produced = 0
    for epoch in range(math.ceil(epochs)):
        order = epoch_order(n, seed, epoch)
        for start in range(0, n, batch_size):
            if produced >= total:
                return
            yield epoch, [encoded[i] for i in order[start : start + batch_size]]
            produced += 1


def optimizer_steps(n_examples: int, config: TrainConfig) -> int:
    return math.ceil(micro_count(n_examples, config.per_device_batch_size, config.epochs) / config.grad_accum)


def train_loop(model: Any, encoded: Sequence[tuple[list[int], list[int]]], config: TrainConfig, ts: settings.TrainSettings, device: str, pad_id: int,
               log: Callable[[str], None] = lambda m: print(m, flush=True), max_steps: int | None = None) -> list[tuple[int, float]]:
    """Run the optimizer loop; returns the loss curve logged every ``logging_steps`` steps.

    ``max_steps`` stops early (the determinism check runs 30 steps of the real schedule).
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    total_steps = optimizer_steps(len(encoded), config)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=ts.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(ts.warmup_ratio * total_steps), total_steps)
    model.train()
    curve: list[tuple[int, float]] = []
    window: list[float] = []
    step, accum, accum_loss = 0, 0, 0.0
    micro_total = micro_count(len(encoded), config.per_device_batch_size, config.epochs)
    for i, (_, batch) in enumerate(micro_batches(encoded, config.per_device_batch_size, config.seed, config.epochs)):
        tensors = {k: v.to(device) for k, v in collate(batch, pad_id).items()}
        # The final optimizer step may have fewer micro-batches; scale by how many it actually has.
        remaining = micro_total - (i - accum)
        group = min(config.grad_accum, remaining)
        out = model(**tensors)
        (out.loss / group).backward()
        accum_loss += float(out.loss.detach()) / group
        accum += 1
        if accum == group:
            torch.nn.utils.clip_grad_norm_(params, ts.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window.append(accum_loss)
            accum, accum_loss = 0, 0.0
            if step % ts.logging_steps == 0 or step == total_steps:
                curve.append((step, sum(window) / len(window)))
                window = []
                log(f"step {step}/{total_steps} loss {curve[-1][1]:.4f}")
            if max_steps is not None and step >= max_steps:
                break
    return curve


def _default_model_factory(model_settings: settings.ModelSettings, device: str) -> Any:
    from fts.lora import load_base_model

    return load_base_model(model_settings.name, model_settings.revision, model_settings.dtype, device)


class PEFTTrainer:
    def __init__(self, model_settings: settings.ModelSettings, train_settings: settings.TrainSettings, templates: prompts.Templates, *,
                 device: str = "cuda", artifacts: Path = ARTIFACTS, results_dir: Path = REPO_ROOT / "results" / "runs",
                 predictions_dir: Path = infer.PREDICTIONS_DIR, model_factory: Callable[[settings.ModelSettings, str], Any] = _default_model_factory,
                 tokenizer: Any = None, val_batch: int = infer.COMPACT_BATCH, log: Callable[[str], None] = lambda m: print(m, flush=True)) -> None:
        self.model_settings = model_settings
        self.ts = train_settings
        self.templates = templates
        self.device = device
        self.artifacts = artifacts
        self.results_dir = results_dir
        self.predictions_dir = predictions_dir
        self.model_factory = model_factory
        self.tokenizer = tokenizer
        self.val_batch = val_batch
        self.log = log

    @staticmethod
    def training_splits(splits: Splits) -> tuple[Sequence[Example], Sequence[Example]]:
        """The only access to the data: Train and Validation (never the Test_Split)."""
        return splits.train, splits.validation

    def _tokenizer(self) -> Any:
        if self.tokenizer is None:
            from fts.lora import load_tokenizer

            self.tokenizer = load_tokenizer(self.model_settings.name, self.model_settings.revision)
        return self.tokenizer

    def build_model(self, config: TrainConfig) -> Any:
        set_determinism(config.seed)
        if self.device.startswith("cuda"):
            limit_cuda_memory(self.ts.cuda_memory_fraction)
        spec = LoraSpec(r=config.lora_r, alpha=config.lora_alpha, dropout=self.ts.lora_dropout, target_modules=self.ts.target_modules)
        return apply_lora(self.model_factory(self.model_settings, self.device), spec, self.ts.gradient_checkpointing)

    def encode(self, examples: Sequence[Example], max_seq_len: int) -> list[tuple[list[int], list[int]]]:
        tok = self._tokenizer()
        return [encode_example(tok, self.templates, ex, max_seq_len) for ex in examples]

    def train(self, splits: Splits, config: TrainConfig, stamp: repro.Stamp) -> RunRecord:
        import torch

        train, validation = self.training_splits(splits)
        rid = run_id(config)
        start = time.monotonic()
        model = self.build_model(config)
        trainable, total = count_parameters(model)
        encoded = self.encode(train, config.max_seq_len)
        tok = self._tokenizer()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        self.log(f"{rid}: {len(encoded)} training examples, {optimizer_steps(len(encoded), config)} optimizer steps")
        curve = train_loop(model, encoded, config, self.ts, self.device, tok.pad_token_id, self.log)
        train_seconds = time.monotonic() - start

        adapter_rel = Path("adapters") / rid
        model.save_pretrained(str(self.artifacts / adapter_rel))

        model.eval()
        if self.ts.gradient_checkpointing:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        preds = infer.predict(model, tok, validation, lambda e: prompts.finetune_messages(self.templates, e.inputs), self.val_batch)
        infer.save_predictions(self.predictions_dir / f"val_{rid}.jsonl", validation, preds)
        report = infer.score(validation, preds)
        val_ci = bootstrap_ci(report.scores, seed=config.seed)
        peak = torch.cuda.max_memory_reserved() / 2**20 if self.device.startswith("cuda") else 0.0

        record = RunRecord(run_id=rid, adapter_path=adapter_rel.as_posix(), loss_curve=tuple(curve), config=config, val_metric=val_ci.point, commit=stamp.commit)
        payload = record.to_dict()
        payload.update({
            "val_ci": val_ci.as_dict(),
            "val_not_scored": len(report.not_scored),
            "splits_used": ["train", "validation"],
            "train_examples": len(encoded),
            "validation_examples": len(validation),
            "optimizer_steps": curve[-1][0] if curve else 0,
            "trainable_params": trainable,
            "total_params": total,
            "fixed_settings": {k: v for k, v in asdict(self.ts).items() if k != "dry_run"},
            "base_model": {"name": self.model_settings.name, "revision": self.model_settings.revision},
            "train_seconds": round(train_seconds, 1),
            "total_seconds": round(time.monotonic() - start, 1),
            "peak_reserved_mib": round(peak),
        })
        repro.write_json(self.results_dir / f"{rid}.json", payload, stamp)
        self.log(f"{rid}: Validation F1 {val_ci.point:.4f} [{val_ci.lower:.4f}, {val_ci.upper:.4f}]")
        return record


def load_run(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record_from_dict(d: dict) -> RunRecord:
    return RunRecord(run_id=d["run_id"], adapter_path=d["adapter_path"], loss_curve=tuple((int(s), float(v)) for s, v in d["loss_curve"]),
                     config=TrainConfig(**d["config"]), val_metric=float(d["val_metric"]), commit=d["commit"])


def with_seed(config: TrainConfig, seed: int) -> TrainConfig:
    return replace(config, seed=seed)
