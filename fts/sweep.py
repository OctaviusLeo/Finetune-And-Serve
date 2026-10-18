"""Sweep and final multi-seed Runs (Req 4.3, 4.4, 4.5, 4.7).

``run_sweep`` trains configs A, B and C (config/sweep.yaml) with seed 1 and
selects the Final_Configuration by Validation F1, ties going to the config
listed first. ``run_final_seeds`` trains the selected config with seeds 1, 2
and 3, reusing the sweep's seed-1 Run, and names the serving seed: the highest
Validation F1, ties going to the lowest seed. A Run whose record and Adapter
already exist is reused rather than retrained, so an interrupted job can resume.

    python -m fts.sweep sweep   -> results/sweep.json
    python -m fts.sweep final   -> results/final_runs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from fts import data, prompts, repro, settings
from fts.train import ARTIFACTS, PEFTTrainer, RunRecord, TrainConfig, config_from, record_from_dict, run_id

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_YAML = REPO_ROOT / "config" / "sweep.yaml"
RATIONALE = "docs/SWEEP_RATIONALE.md"
MIN_CONFIGS = 3
MIN_SEEDS = 3

TrainFn = Callable[[TrainConfig], RunRecord]


@dataclass(frozen=True)
class SweepResult:
    configs: tuple[TrainConfig, ...]
    val_metrics: dict[str, float]  # config name -> Validation F1
    selected: TrainConfig  # Final_Configuration
    rationale_path: str


def load_sweep_configs(base: settings.TrainSettings, path: Path | str = SWEEP_YAML) -> tuple[list[TrainConfig], list[int]]:
    d = settings.load_yaml(path)
    configs = [config_from(base, c["name"], int(c["lora_r"]), int(c["lora_alpha"]), float(c["learning_rate"]), int(d["seed"])) for c in d["configs"]]
    return configs, [int(s) for s in d["final_seeds"]]


def select_final(configs: Sequence[TrainConfig], val_metrics: Mapping[str, float]) -> tuple[TrainConfig, float]:
    """Highest Validation F1; a tie goes to the config listed first."""
    if len(configs) < MIN_CONFIGS:
        raise ValueError(f"the sweep needs at least {MIN_CONFIGS} configurations, got {len(configs)}")
    best = configs[0]
    for c in configs[1:]:
        if val_metrics[c.name] > val_metrics[best.name]:
            best = c
    return best, val_metrics[best.name]


def serving_seed(records: Sequence[RunRecord]) -> int:
    """Seed with the highest Validation F1; a tie goes to the lowest seed."""
    return min(records, key=lambda r: (-r.val_metric, r.config.seed)).config.seed


def check_seeds(seeds: Sequence[int]) -> None:
    if len(set(seeds)) < MIN_SEEDS or len(set(seeds)) != len(seeds):
        raise ValueError(f"final training needs at least {MIN_SEEDS} distinct seeds, got {list(seeds)}")


def run_sweep(splits: data.Splits, configs: Sequence[TrainConfig], train_fn: TrainFn, rationale_path: str = RATIONALE) -> tuple[SweepResult, list[RunRecord]]:
    """Train each config and select by Validation F1 (Train/Validation only)."""
    if len(configs) < MIN_CONFIGS:
        raise ValueError(f"the sweep needs at least {MIN_CONFIGS} configurations, got {len(configs)}")
    if len({c.seed for c in configs}) != 1:
        raise ValueError("all sweep configs must share one seed")
    records = [train_fn(c) for c in configs]
    metrics = {r.config.name: r.val_metric for r in records}
    selected, _ = select_final(configs, metrics)
    return SweepResult(tuple(configs), metrics, selected, rationale_path), records


def run_final_seeds(splits: data.Splits, final: TrainConfig, seeds: Sequence[int], train_fn: TrainFn,
                    existing: Mapping[int, RunRecord] | None = None) -> list[RunRecord]:
    """One Run per seed of the Final_Configuration; Runs in ``existing`` (by seed) are reused."""
    check_seeds(seeds)
    existing = existing or {}
    return [existing[s] if s in existing else train_fn(replace(final, seed=s)) for s in seeds]


def reusable_record(config: TrainConfig, results_dir: Path, artifacts: Path) -> RunRecord | None:
    """A previously finished Run of exactly this config whose Adapter still exists."""
    p = results_dir / f"{run_id(config)}.json"
    if not p.exists():
        return None
    record = record_from_dict(json.loads(p.read_text(encoding="utf-8")))
    if record.config != config or not (artifacts / record.adapter_path / "adapter_config.json").exists():
        return None
    return record


def make_train_fn(trainer: PEFTTrainer, splits: data.Splits, stamp: repro.Stamp, results_dir: Path, artifacts: Path,
                  log: Callable[[str], None] = lambda m: print(m, flush=True)) -> TrainFn:
    def train_fn(config: TrainConfig) -> RunRecord:
        found = reusable_record(config, results_dir, artifacts)
        if found is not None:
            log(f"{run_id(config)}: reusing the finished Run from commit {found.commit}")
            return found
        return trainer.train(splits, config, stamp)

    return train_fn


def sweep_payload(result: SweepResult, records: Sequence[RunRecord]) -> dict:
    return {
        "configs": [asdict(c) for c in result.configs],
        "val_metrics": result.val_metrics,
        "runs": {r.config.name: {"run_id": r.run_id, "val_metric": r.val_metric, "commit": r.commit, "record": f"results/runs/{r.run_id}.json"} for r in records},
        "selected": asdict(result.selected),
        "selected_val_metric": result.val_metrics[result.selected.name],
        "selection_rule": "highest Validation F1; ties go to the config listed first",
        "rationale_path": result.rationale_path,
    }


def final_payload(final: TrainConfig, records: Sequence[RunRecord]) -> dict:
    return {
        "final_config": asdict(final),
        "runs": [{"run_id": r.run_id, "seed": r.config.seed, "val_metric": r.val_metric, "adapter_path": r.adapter_path, "commit": r.commit,
                  "loss_curve": [list(p) for p in r.loss_curve], "config": asdict(r.config), "record": f"results/runs/{r.run_id}.json"} for r in records],
        "seeds": [r.config.seed for r in records],
        "serving_seed": serving_seed(records),
        "serving_rule": "highest Validation F1; ties go to the lowest seed",
    }


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, trainer: PEFTTrainer | None = None, artifacts: Path = ARTIFACTS) -> int:
    parser = argparse.ArgumentParser(prog="python -m fts.sweep", description="LoRA sweep and final seeds.")
    parser.add_argument("stage", choices=["sweep", "final"])
    args = parser.parse_args(argv)
    base = settings.load_train_settings(root / "config" / "train.yaml")
    configs, seeds = load_sweep_configs(base, root / "config" / "sweep.yaml")
    stamp = repro.start_run(seed=configs[0].seed)
    if not (root / RATIONALE).exists():
        print(f"FAIL: {RATIONALE} must be written before any sweep Run", file=sys.stderr)
        return 1
    splits = data.load_splits(root / "data" / "massive_en")
    results_dir = root / "results" / "runs"
    trainer = trainer or PEFTTrainer(settings.load_model_settings(root / "config" / "model.yaml"), base, prompts.load_templates(root / "config" / "prompts"),
                                     artifacts=artifacts, results_dir=results_dir)
    train_fn = make_train_fn(trainer, splits, stamp, results_dir, artifacts)

    if args.stage == "sweep":
        result, records = run_sweep(splits, configs, train_fn)
        repro.write_json(root / "results" / "sweep.json", sweep_payload(result, records), stamp)
        print(f"selected {result.selected.name} with Validation F1 {result.val_metrics[result.selected.name]:.4f}; {result.val_metrics}")
        return 0

    sweep_path = root / "results" / "sweep.json"
    if not sweep_path.exists():
        print("FAIL: results/sweep.json is missing; run the sweep first", file=sys.stderr)
        return 1
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    final = TrainConfig(**sweep["selected"])
    seed1 = reusable_record(final, results_dir, artifacts)
    records = run_final_seeds(splits, final, seeds, train_fn, existing={final.seed: seed1} if seed1 else None)
    repro.write_json(root / "results" / "final_runs.json", final_payload(final, records), stamp)
    print(f"final seeds {[r.config.seed for r in records]}: Validation F1 {[round(r.val_metric, 4) for r in records]}; serving seed {serving_seed(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
