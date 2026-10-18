"""Unit tests for fts.eval_core.bootstrap_ci (task 2.7)."""

from __future__ import annotations

import numpy as np
import pytest

from fts.eval_core import MIN_RESAMPLES, BootstrapCI, bootstrap_ci


def test_interval_matches_numpy_percentiles():
    samples = [0.0, 0.5, 1.0, 1.0, 0.8, 0.2]
    ci = bootstrap_ci(samples, seed=7)
    rng = np.random.default_rng(7)
    data = np.asarray(samples)
    means = data[rng.integers(0, 6, size=(MIN_RESAMPLES, 6))].mean(axis=1)
    assert ci.point == pytest.approx(np.mean(samples))
    assert ci.lower == pytest.approx(min(np.quantile(means, 0.025), ci.point))
    assert ci.upper == pytest.approx(max(np.quantile(means, 0.975), ci.point))


def test_constant_samples_give_a_point_interval():
    assert bootstrap_ci([1.0] * 10, seed=0) == BootstrapCI(1.0, 1.0, 1.0)
    assert bootstrap_ci([0.3], seed=0) == BootstrapCI(0.3, 0.3, 0.3)


def test_larger_sample_gives_narrower_interval():
    rng = np.random.default_rng(0)
    small = rng.uniform(size=50).tolist()
    large = rng.uniform(size=2000).tolist()
    w_small = (lambda c: c.upper - c.lower)(bootstrap_ci(small, seed=1))
    w_large = (lambda c: c.upper - c.lower)(bootstrap_ci(large, seed=1))
    assert w_large < w_small


def test_more_resamples_allowed_and_confidence_changes_width():
    s = np.random.default_rng(3).uniform(size=200).tolist()
    ci95 = bootstrap_ci(s, seed=2, resamples=2000)
    ci80 = bootstrap_ci(s, seed=2, resamples=2000, confidence=0.80)
    assert (ci80.upper - ci80.lower) < (ci95.upper - ci95.lower)


@pytest.mark.parametrize("kwargs", [{"samples": []}, {"resamples": 999}, {"confidence": 1.0}, {"confidence": 0.0}])
def test_invalid_arguments(kwargs):
    args = {"samples": [0.5, 0.6], "seed": 0, **kwargs}
    with pytest.raises(ValueError):
        bootstrap_ci(**args)


def test_as_dict():
    assert BootstrapCI(0.5, 0.4, 0.6).as_dict() == {"point": 0.5, "lower": 0.4, "upper": 0.6}


# --- evaluator and seed summary (task 4.2; Req 5.3, 5.7, 5.8) -------------------------------

from fts import data as _data  # noqa: E402
from fts import eval_core  # noqa: E402
from fts.infer import Prediction  # noqa: E402

GOLD = '{"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]}'
TEST = (
    _data.Example("1", "test", "a", GOLD),
    _data.Example("2", "test", "b", "this gold is malformed"),
    _data.Example("3", "test", "c", GOLD),
)
PREDS = [Prediction("1", GOLD, 10, 5), Prediction("2", GOLD, 10, 5), Prediction("3", '{"intent": "alarm_set", "slots": []}', 10, 5)]


def test_malformed_test_example_is_not_scored_and_the_rest_continue():
    r = eval_core.evaluate_predictions("B-seed1", 1, TEST, PREDS, bootstrap_seed=0)
    assert r.scored == 2 and r.n_test == 3
    assert r.not_scored == [{"id": "2", "cause": "gold target is missing or not valid task JSON"}]
    assert r.per_example == {"1": 1.0, "3": pytest.approx(2 / 3)}
    assert r.metric.point == pytest.approx((1.0 + 2 / 3) / 2)
    d = r.to_dict()
    assert d["not_scored"] == 1 and d["metric"]["point"] == r.metric.point and d["test_ids_sha256"] == eval_core.split_digest(TEST)


def test_same_seed_and_predictions_give_equal_values():
    a = eval_core.evaluate_predictions("B-seed1", 1, TEST, PREDS, bootstrap_seed=5)
    b = eval_core.evaluate_predictions("B-seed1", 1, TEST, PREDS, bootstrap_seed=5)
    for x, y in ((a.metric.point, b.metric.point), (a.metric.lower, b.metric.lower), (a.metric.upper, b.metric.upper)):
        assert abs(x - y) <= 1e-6


def test_evaluate_uses_the_predictor_on_the_test_split_only():
    class Run:
        run_id, config = "B-seed2", type("C", (), {"seed": 2})()

    splits = _data.Splits((), (), TEST)
    seen = []
    r = eval_core.evaluate(Run(), splits, seed=0, predict=lambda run, test: seen.append((run.run_id, len(test))) or PREDS)
    assert seen == [("B-seed2", 3)] and r.seed == 2 and r.run_id == "B-seed2"


def test_no_scorable_examples_raises():
    with pytest.raises(ValueError):
        eval_core.evaluate_predictions("x", 1, TEST[1:2], PREDS[1:2], bootstrap_seed=0)


def test_seed_summary_mean_and_sample_std_on_fixture():
    s = eval_core.seed_summary({3: 0.80, 1: 0.90, 2: 0.85})
    assert list(s.per_seed) == [1, 2, 3]
    assert s.mean == pytest.approx(0.85) and s.spread == pytest.approx(0.05)  # sample std of (0.90, 0.85, 0.80)
    with pytest.raises(ValueError):
        eval_core.seed_summary({1: 0.9})
