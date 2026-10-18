"""Local inference for the baselines and evaluation (design.md "Local inference").

One engine produces every Base_Model and Finetuned_Model number: Hugging Face
``transformers`` generation in bf16, greedy (``do_sample=False``),
``max_new_tokens`` 128, left padding, examples in dataset order, batch 32 for
the compact prompt and 8 for the long baseline prompts. An Adapter is applied
with ``peft`` without merging. Predictions are saved per example as JSONL under
runs/predictions/. A gold target that the metric cannot parse is reported as
"not scored" with its cause and the remaining examples are still scored.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fts.data import Example
from fts.metric import parse_prediction, score_example

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = REPO_ROOT / "runs" / "predictions"
MAX_NEW_TOKENS = 128
COMPACT_BATCH = 32
LONG_BATCH = 8


@dataclass(frozen=True)
class Prediction:
    example_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ScoreReport:
    scores: list[float] = field(default_factory=list)
    scored_ids: list[str] = field(default_factory=list)
    not_scored: list[dict] = field(default_factory=list)  # {"id", "cause"}

    @property
    def per_example(self) -> dict[str, float]:
        return dict(zip(self.scored_ids, self.scores))


def load_for_inference(name: str, revision: str | None, adapter_path: str | Path | None = None, dtype: str = "bfloat16", device: str = "cuda") -> tuple[Any, Any]:
    """The Base_Model (optionally with an unmerged LoRA Adapter) and its tokenizer, ready to generate."""
    from fts.lora import load_base_model, load_tokenizer

    model = load_base_model(name, revision, dtype, device)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    model.config.use_cache = True
    return model, load_tokenizer(name, revision)


def render_prompts(tokenizer: Any, message_lists: Sequence[list[dict]]) -> list[str]:
    return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_lists]


def stop_token_ids(*values: int | list[int] | None) -> set[int]:
    """Token ids that end a completion (eos ids from the tokenizer and generation config, and the pad id)."""
    ids: set[int] = set()
    for v in values:
        if v is not None:
            ids.update(v if isinstance(v, list) else [v])
    return ids


def generate_batch(model: Any, tokenizer: Any, prompts: Sequence[str], max_new_tokens: int = MAX_NEW_TOKENS) -> list[tuple[str, int, int]]:
    """Greedy-decode one left-padded batch; returns (text, prompt tokens, completion tokens) per prompt."""
    import torch

    tokenizer.padding_side = "left"
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    eos = tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[:, input_ids.shape[1] :].tolist()
    results = []
    stop_ids = stop_token_ids(eos, getattr(model.generation_config, "eos_token_id", None), tokenizer.pad_token_id)
    for row, mask in zip(new_tokens, attention_mask.tolist()):
        kept = []
        for tok_id in row:
            if tok_id in stop_ids:
                kept.append(tok_id)
                break
            kept.append(tok_id)
        text = tokenizer.decode([t for t in kept if t not in stop_ids], skip_special_tokens=True)
        results.append((text, int(sum(mask)), len(kept)))
    return results


def predict(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    build_messages: Callable[[Example], list[dict]],
    batch_size: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    progress: Callable[[int, int], None] | None = None,
) -> list[Prediction]:
    """Predict every example in dataset order, ``batch_size`` at a time."""
    preds: list[Prediction] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = render_prompts(tokenizer, [build_messages(ex) for ex in batch])
        for ex, (text, p_tok, c_tok) in zip(batch, generate_batch(model, tokenizer, prompts, max_new_tokens)):
            preds.append(Prediction(ex.example_id, text, p_tok, c_tok))
        if progress is not None:
            progress(len(preds), len(examples))
    return preds


def score(examples: Sequence[Example], predictions: Sequence[Prediction]) -> ScoreReport:
    """Score predictions against gold; malformed or missing gold is reported as not scored."""
    by_id = {p.example_id: p for p in predictions}
    report = ScoreReport()
    for ex in examples:
        if not isinstance(ex.target, str) or parse_prediction(ex.target) is None:
            report.not_scored.append({"id": ex.example_id, "cause": "gold target is missing or not valid task JSON"})
            continue
        pred = by_id.get(ex.example_id)
        if pred is None:
            report.not_scored.append({"id": ex.example_id, "cause": "no prediction"})
            continue
        report.scores.append(score_example(pred.text, ex.target))
        report.scored_ids.append(ex.example_id)
    return report


def save_predictions(path: Path | str, examples: Sequence[Example], predictions: Sequence[Prediction]) -> Path:
    """One JSON line per example: id, gold, prediction, token counts, and its F1 when scorable."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gold = {ex.example_id: ex.target for ex in examples}
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for p in predictions:
            row = asdict(p)
            row["gold"] = gold.get(p.example_id)
            try:
                row["f1"] = score_example(p.text, row["gold"]) if row["gold"] else None
            except ValueError:
                row["f1"] = None
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def load_predictions(path: Path | str) -> list[Prediction]:
    preds = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                preds.append(Prediction(row["example_id"], row["text"], row["prompt_tokens"], row["completion_tokens"]))
    return preds
