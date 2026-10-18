"""Unit tests for the `python -m fts.eval` entry point (task 4.6).

Exercises the pure, GPU-free paths: the stamped report assembled by
``run_eval``, the default failure taxonomy, and the non-interactive ``main``.
No heavy ML import is triggered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fts.eval import _default_classify, build_parser, main, run_eval


@dataclass(frozen=True)
class _Ex:
    example_id: str
    inputs: str
    target: object


@dataclass(frozen=True)
class _Splits:
    test: tuple


def _echo_predict(example):
    return example.target


def test_run_eval_assembles_stamped_report():
    """run_eval returns the Test metric + CI, seed summary, and grouped failures, each stamped."""
    test = (
        _Ex("ok-1", "a", '{"n": 1}'),
        _Ex("bad-1", "b", "not json"),  # scores 0 -> a failure, groups 'malformed-gold'
        _Ex("ok-2", "c", '{"n": 2}'),
    )
    report = run_eval(
        _Splits(test=test),
        seed=5,
        predict=_echo_predict,
        per_seed_metric={1: 0.8, 2: 0.6, 3: 0.7},
    )

    # Stamp round-trips and carries a 40-char commit.
    from fts import repro

    stamp = repro.deserialize(report["stamp"])
    assert stamp.seed == 5
    assert len(stamp.commit) == 40

    # Test metric: the two JSON examples score 1.0, the non-JSON echoes to 1.0 too
    # (prediction == gold after strip), so all three are scored.
    assert report["test_metric"]["scored"] == 3
    assert report["test_metric"]["lower"] <= report["test_metric"]["point"]
    assert report["test_metric"]["point"] <= report["test_metric"]["upper"]

    # Seed summary mean/spread over the fixed per-seed metric.
    assert abs(report["seed_summary"]["mean"] - 0.7) <= 1e-9
    assert abs(report["seed_summary"]["spread"] - 0.2) <= 1e-9

    # Error analysis group counts sum to the analyzed total.
    analysis = report["error_analysis"]
    assert sum(g["count"] for g in analysis["groups"]) == analysis["analyzed"]


def test_run_eval_groups_failures_by_default_taxonomy():
    """A wrong prediction is counted and grouped by the documented default taxonomy."""
    test = (_Ex("f1", "a", '{"n": 1}'), _Ex("f2", "b", "plain target"))

    def wrong_predict(example):
        return '{"other": 0}'  # never matches -> every example is a failure

    report = run_eval(
        _Splits(test=test),
        seed=0,
        predict=wrong_predict,
        per_seed_metric={10: 0.5, 20: 0.5, 30: 0.5},
    )
    analysis = report["error_analysis"]
    assert analysis["analyzed"] == 2
    groups = {g["group"]: g["count"] for g in analysis["groups"]}
    assert groups == {"wrong-fields": 1, "malformed-gold": 1}


def test_default_classify_taxonomy():
    assert _default_classify(_Ex("a", "x", '{"k": 1}')) == "wrong-fields"
    assert _default_classify(_Ex("b", "x", "not json")) == "malformed-gold"
    assert _default_classify(_Ex("c", "x", 123)) == "missing-gold"


def test_main_runs_non_interactively_and_exits_zero(capsys):
    """`python -m fts.eval` runs non-interactively, prints a stamped JSON report, exits 0."""
    rc = main(["--config", "cfg.json", "--seed", "11", "--adapter-path", "adapters/x"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    from fts import repro

    stamp = repro.deserialize(out["stamp"])  # seed lives inside the stamp
    assert stamp.seed == 11
    assert out["config"] == "cfg.json"
    assert out["adapter_path"] == "adapters/x"


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.seed == 0
    assert args.config is None
    assert args.base_model == "base-model-tbd"
