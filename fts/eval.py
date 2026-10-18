"""Single-command evaluation of the Finetuned_Model (Req 5.1-5.8, 0.1, 0.3).

    python -m fts.eval [--config results/final_runs.json]
        Evaluates every final seed's Adapter on the Test_Split (compact prompt,
        fts.infer, batch 32) and writes, stamped with command, seed and commit:
        results/finetuned.json      - each seed's Test F1 with a 95% CI, the mean and
                                      the sample standard deviation across seeds;
        results/error_analysis.json - every failure of the serving seed, grouped.

    python -m fts.eval --rerun-check --seed N
        Re-runs seed N and compares with its saved predictions: predictions must be
        identical and the metric equal within 1e-6. Differing predictions are recorded
        in docs/ENVIRONMENT.md as GPU-kernel nondeterminism (Req 0.3's exclusion), and
        then only the scoring of the identical predictions must match.

The bootstrap seed is ``BOOTSTRAP_SEED`` (0), recorded as the stamp's seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from fts import data, envdoc, error_analysis, infer, prompts, repro, settings
from fts.eval_core import EvalResult, evaluate_predictions, seed_summary, split_digest
from fts.infer import Prediction
from fts.train import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_SEED = 0
TOLERANCE = 1e-6
SECTION = "Determinism notes: evaluation"

Predictor = Callable[[dict, Sequence[data.Example]], list[Prediction]]


def default_predictor(root: Path = REPO_ROOT, artifacts: Path = ARTIFACTS, device: str = "cuda") -> Predictor:
    """Loads the Base_Model with one Run's Adapter (unmerged) and predicts with the compact prompt."""
    model_settings = settings.load_model_settings(root / "config" / "model.yaml")
    templates = prompts.load_templates(root / "config" / "prompts")

    def predict(run: dict, test: Sequence[data.Example]) -> list[Prediction]:
        import gc

        import torch

        model, tok = infer.load_for_inference(model_settings.name, model_settings.revision, artifacts / run["adapter_path"], device=device)
        preds = infer.predict(model, tok, test, lambda e: prompts.finetune_messages(templates, e.inputs), infer.COMPACT_BATCH,
                              progress=lambda d, n: print(f"  {run['run_id']}: {d}/{n}", flush=True) if d % 640 < infer.COMPACT_BATCH or d == n else None)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return preds

    return predict


def evaluate_runs(runs: Sequence[dict], test: Sequence[data.Example], predict: Predictor, predictions_dir: Path) -> tuple[list[EvalResult], dict[int, list[Prediction]]]:
    results, preds_by_seed = [], {}
    for run in runs:
        preds = predict(run, test)
        infer.save_predictions(predictions_dir / f"test_{run['run_id']}.jsonl", test, preds)
        results.append(evaluate_predictions(run["run_id"], run["seed"], test, preds, BOOTSTRAP_SEED))
        preds_by_seed[run["seed"]] = preds
        print(f"{run['run_id']}: Test F1 {results[-1].metric.point:.4f} [{results[-1].metric.lower:.4f}, {results[-1].metric.upper:.4f}]", flush=True)
    return results, preds_by_seed


def finetuned_payload(results: Sequence[EvalResult], serving_seed: int) -> dict:
    summary = seed_summary({r.seed: r.metric.point for r in results})
    return {
        "seeds": [r.to_dict() for r in results],
        "mean": summary.mean,
        "spread": summary.spread,
        "spread_measure": "sample standard deviation across seeds",
        "serving_seed": serving_seed,
        "test_ids_sha256": results[0].test_ids_sha256,
        "n_test": results[0].n_test,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "prompt_template": f"{prompts.template_path('system')}, {prompts.template_path('finetune')}",
    }


def compare_predictions(first: Sequence[Prediction], second: Sequence[Prediction]) -> list[str]:
    """Ids whose prediction text differs between two runs."""
    a = {p.example_id: p.text for p in first}
    return [p.example_id for p in second if a.get(p.example_id) != p.text]


def rerun_payload(run: dict, test: Sequence[data.Example], first: Sequence[Prediction], second: Sequence[Prediction]) -> dict:
    differing = compare_predictions(first, second)
    r1 = evaluate_predictions(run["run_id"], run["seed"], test, first, BOOTSTRAP_SEED)
    r2 = evaluate_predictions(run["run_id"], run["seed"], test, second, BOOTSTRAP_SEED)
    same = set(differing)
    identical_ids = [e for e in test if e.example_id not in same]
    s1 = evaluate_predictions(run["run_id"], run["seed"], identical_ids, [p for p in first if p.example_id not in same], BOOTSTRAP_SEED) if identical_ids else None
    s2 = evaluate_predictions(run["run_id"], run["seed"], identical_ids, [p for p in second if p.example_id not in same], BOOTSTRAP_SEED) if identical_ids else None
    full_equal = all(abs(x - y) <= TOLERANCE for x, y in zip(r1.metric.as_dict().values(), r2.metric.as_dict().values()))
    subset_equal = s1 is not None and all(abs(x - y) <= TOLERANCE for x, y in zip(s1.metric.as_dict().values(), s2.metric.as_dict().values()))
    if not differing:
        status = "identical" if full_equal else "metric_mismatch"
    else:
        status = "documented_gpu_nondeterminism" if subset_equal else "metric_mismatch"
    return {
        "run_id": run["run_id"],
        "seed": run["seed"],
        "status": status,
        "differing_ids": differing,
        "first": r1.metric.as_dict(),
        "second": r2.metric.as_dict(),
        "identical_subset": {"n": len(identical_ids), "first": s1.metric.as_dict() if s1 else None, "second": s2.metric.as_dict() if s2 else None},
        "tolerance": TOLERANCE,
    }


def render_rerun_section(result: dict, stamp: repro.Stamp) -> str:
    lines = [f"- Evaluation re-run of {result['run_id']} on the Test_Split: **{result['status']}**; "
             f"{len(result['differing_ids'])} prediction(s) differed; Test F1 {result['first']['point']:.6f} then {result['second']['point']:.6f}"]
    if result["differing_ids"]:
        lines.append(f"- Differing ids (GPU-kernel nondeterminism, Req 0.3 exclusion): {', '.join(result['differing_ids'])}")
        sub = result["identical_subset"]
        lines.append(f"- Scoring of the {sub['n']} identical predictions: {sub['first']} then {sub['second']}")
    return "\n".join(lines + envdoc.stamp_lines(stamp))


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, predict: Predictor | None = None, env_md: Path = envdoc.ENVIRONMENT_MD) -> int:
    parser = argparse.ArgumentParser(prog="python -m fts.eval", description="Evaluate the Finetuned_Model on the Test_Split.")
    parser.add_argument("--config", default="results/final_runs.json", help="final Runs to evaluate (written by python -m fts.sweep final)")
    parser.add_argument("--seed", type=int, default=None, help="with --rerun-check: the seed to re-run")
    parser.add_argument("--rerun-check", action="store_true", help="re-run one seed and compare with its saved predictions")
    args = parser.parse_args(argv)
    stamp = repro.start_run(seed=BOOTSTRAP_SEED)
    config_path = root / args.config
    if not config_path.exists():
        print(f"FAIL: {args.config} is missing; run python -m fts.sweep final first", file=sys.stderr)
        return 1
    final = json.loads(config_path.read_text(encoding="utf-8"))
    test = data.load_splits(root / "data" / "massive_en").test
    predict = predict or default_predictor(root)
    predictions_dir = root / "runs" / "predictions"

    if args.rerun_check:
        seed = final["serving_seed"] if args.seed is None else args.seed
        run = next(r for r in final["runs"] if r["seed"] == seed)
        saved = predictions_dir / f"test_{run['run_id']}.jsonl"
        if not saved.exists():
            print(f"FAIL: {saved} is missing; run python -m fts.eval first", file=sys.stderr)
            return 1
        second = predict(run, test)
        infer.save_predictions(predictions_dir / f"test_{run['run_id']}_rerun.jsonl", test, second)
        result = rerun_payload(run, test, infer.load_predictions(saved), second)
        repro.write_json(root / "results" / "eval_determinism.json", result, stamp)
        envdoc.replace_section(SECTION, render_rerun_section(result, stamp), env_md)
        print(f"re-run {run['run_id']}: {result['status']}, {len(result['differing_ids'])} differing predictions")
        return 0 if result["status"] != "metric_mismatch" else 1

    results, preds_by_seed = evaluate_runs(final["runs"], test, predict, predictions_dir)
    serving = final["serving_seed"]
    repro.write_json(root / "results" / "finetuned.json", finetuned_payload(results, serving), stamp)
    analysis = error_analysis.error_analysis(error_analysis.pairs_from_predictions(test, preds_by_seed[serving]))
    payload = analysis.to_dict() | {"seed": serving, "run_id": next(r["run_id"] for r in final["runs"] if r["seed"] == serving),
                                    "test_ids_sha256": split_digest(test)}
    repro.write_json(root / "results" / "error_analysis.json", payload, stamp)
    print(f"mean Test F1 {finetuned_payload(results, serving)['mean']:.4f}; error analysis of {analysis.analyzed} failures: "
          f"{ {g.group: g.count for g in analysis.groups} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
