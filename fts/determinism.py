"""Training determinism check (Req 4.6, 0.3, 0.4).

Train the Final_Configuration twice from scratch for 30 optimizer steps with the
same seed, logging the loss at every step, and compare the two curves within
1e-6. PyTorch's determinism mode is warn-only, so kernels without a
deterministic implementation announce themselves with a UserWarning; those
warnings are captured. Outcomes:

* identical - every loss matches within 1e-6 (exit 0);
* documented - a mismatch, and PyTorch named at least one nondeterministic
  kernel; the values, the largest difference and the kernels are recorded in
  docs/ENVIRONMENT.md (exit 0, Req 0.3's documented-hardware exclusion);
* unexplained - a mismatch with no named kernel (exit 1).
"""

from __future__ import annotations

import gc
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from fts import data, envdoc, prompts, repro, settings
from fts.train import PEFTTrainer, TrainConfig, train_loop

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = 30
TOLERANCE = 1e-6
SECTION = "Determinism notes: training"


def nondeterminism_messages(records: Sequence[warnings.WarningMessage]) -> list[str]:
    """First line of each distinct warning that names nondeterministic behaviour."""
    found = []
    for w in records:
        text = str(w.message).strip().splitlines()[0] if str(w.message).strip() else ""
        if "deterministic" in text.lower() and text not in found:
            found.append(text)
    return found


def compare(a: Sequence[float], b: Sequence[float], tol: float = TOLERANCE) -> tuple[bool, float]:
    """(all equal within tol and same length, largest absolute difference)."""
    if len(a) != len(b):
        return False, float("inf")
    diff = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    return diff <= tol, diff


def classify(equal: bool, kernels: Sequence[str]) -> str:
    if equal:
        return "identical"
    return "documented" if kernels else "unexplained"


def run_once(trainer: PEFTTrainer, train_examples: Sequence[data.Example], config: TrainConfig, steps: int) -> tuple[list[float], list[str]]:
    """One fresh model trained for ``steps`` optimizer steps; (loss per step, nondeterminism warnings)."""
    model = trainer.build_model(config)
    encoded = trainer.encode(train_examples, config.max_seq_len)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        curve = train_loop(model, encoded, config, replace(trainer.ts, logging_steps=1), trainer.device, trainer._tokenizer().pad_token_id,
                           log=lambda m: None, max_steps=steps)
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is a dependency
        pass
    return [v for _, v in curve], nondeterminism_messages(caught)


def check(trainer: PEFTTrainer, splits: data.Splits, config: TrainConfig, steps: int = STEPS,
          log: Callable[[str], None] = lambda m: print(m, flush=True)) -> dict:
    train_examples, _ = trainer.training_splits(splits)
    first, k1 = run_once(trainer, train_examples, config, steps)
    log(f"run 1: {len(first)} losses")
    second, k2 = run_once(trainer, train_examples, config, steps)
    log(f"run 2: {len(second)} losses")
    equal, diff = compare(first, second)
    kernels = list(dict.fromkeys(k1 + k2))
    return {
        "config": asdict(config),
        "steps": steps,
        "tolerance": TOLERANCE,
        "run1_losses": first,
        "run2_losses": second,
        "max_abs_diff": diff,
        "equal": equal,
        "nondeterministic_kernels": kernels,
        "status": classify(equal, kernels),
    }


def render_section(result: dict, stamp: repro.Stamp) -> str:
    cfg = result["config"]
    lines = [
        f"- Training determinism check ({result['steps']} optimizer steps of config {cfg['name']}, seed {cfg['seed']}, run twice): "
        f"**{result['status']}**; largest loss difference {result['max_abs_diff']:.3g} (tolerance {result['tolerance']:g})",
    ]
    if result["nondeterministic_kernels"]:
        lines.append("- Nondeterministic kernels named by PyTorch's warn-only determinism mode:")
        lines += [f"  - `{k}`" for k in result["nondeterministic_kernels"]]
    if not result["equal"]:
        pairs = ", ".join(f"{a:.6f}/{b:.6f}" for a, b in zip(result["run1_losses"], result["run2_losses"]))
        lines.append(f"- Loss per step, run 1 / run 2: {pairs}")
    lines += envdoc.stamp_lines(stamp)
    return "\n".join(lines)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, trainer: PEFTTrainer | None = None, env_md: Path = envdoc.ENVIRONMENT_MD) -> int:
    import json

    sweep_path = root / "results" / "sweep.json"
    if not sweep_path.exists():
        print("FAIL: results/sweep.json is missing; the Final_Configuration comes from the sweep", file=sys.stderr)
        return 1
    config = TrainConfig(**json.loads(sweep_path.read_text(encoding="utf-8"))["selected"])
    stamp = repro.start_run(seed=config.seed)
    splits = data.load_splits(root / "data" / "massive_en")
    trainer = trainer or PEFTTrainer(settings.load_model_settings(root / "config" / "model.yaml"), settings.load_train_settings(root / "config" / "train.yaml"),
                                     prompts.load_templates(root / "config" / "prompts"))
    result = check(trainer, splits, config)
    repro.write_json(root / "results" / "train_determinism.json", result, stamp)
    envdoc.replace_section(SECTION, render_section(result, stamp), env_md)
    print(f"training determinism: {result['status']}, largest difference {result['max_abs_diff']:.3g}, kernels {result['nondeterministic_kernels']}")
    if result["status"] == "unexplained":
        print("FAIL: training losses differ and PyTorch named no nondeterministic kernel", file=sys.stderr)
        return 1
    return 0
