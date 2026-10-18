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
