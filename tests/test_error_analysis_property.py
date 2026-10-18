"""Property-based tests for grouped error analysis + seed summary (task 4.5).

Feature: finetune-and-serve, Property 6: Error-analysis group counts sum to the analyzed total (>=min(50, available failures))
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from fts.error_analysis import MIN_FAILURES, error_analysis, seed_summary


@dataclass(frozen=True)
class _Ex:
    example_id: str
    inputs: str
    target: str


@dataclass(frozen=True)
class _Splits:
    test: tuple


# A pool of distinct failure-group labels the classifier may assign.
_GROUPS = ("missing-field", "extra-field", "wrong-value", "format-error", "other")


@given(
    n_failures=st.integers(min_value=0, max_value=120),
    n_passes=st.integers(min_value=0, max_value=40),
    group_ids=st.lists(st.integers(min_value=0, max_value=4), min_size=120, max_size=120),
)
def test_group_counts_sum_to_analyzed(n_failures, n_passes, group_ids):
    """Feature: finetune-and-serve, Property 6: Error-analysis group counts sum to the analyzed total (>=min(50, available failures))."""
    # Failures: predictor returns a wrong JSON object -> score 0 (< threshold).
    failures = [
        _Ex(f"fail-{i}", f"in-{i}", f'{{"n": {i}}}') for i in range(n_failures)
    ]
    # Passes: predictor echoes the gold -> score 1.0 (not a failure).
    passes = [
        _Ex(f"pass-{i}", f"in-{i}", f'{{"m": {i}}}') for i in range(n_passes)
    ]
    test = tuple(failures + passes)
    splits = _Splits(test=test)

    def predict(example):
        if example.example_id.startswith("pass-"):
            return example.target  # perfect -> not a failure
        return '{"wrong": true}'  # always a failure

    def classify(example):
        idx = int(example.example_id.split("-")[1])
        return _GROUPS[group_ids[idx % len(group_ids)]]

    result = error_analysis(predict, splits, classify)

    # Group counts always sum to the analyzed total.
    assert sum(g.count for g in result.groups) == result.analyzed

    # analyzed is min(50, available failures); never more than available.
    expected_analyzed = min(MIN_FAILURES, n_failures)
    assert result.analyzed == expected_analyzed
    assert result.analyzed >= min(MIN_FAILURES, n_failures)
    assert result.analyzed <= n_failures


# Fixed fixture asserting seed-summary mean/spread exactly.
_FIXED_PER_SEED = {1: 0.80, 2: 0.60, 3: 0.70}


def test_seed_summary_mean_and_spread_on_fixed_fixture():
    """seed_summary reports the exact mean and max-minus-min spread (Req 6.4)."""
    summary = seed_summary(_FIXED_PER_SEED)
    assert abs(summary.mean - 0.70) <= 1e-9
    assert abs(summary.spread - 0.20) <= 1e-9
    assert dict(summary.per_seed) == _FIXED_PER_SEED


@given(
    per_seed=st.dictionaries(
        keys=st.integers(min_value=0, max_value=1000),
        values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=10,
    )
)
def test_seed_summary_mean_within_values_and_nonneg_spread(per_seed):
    """seed_summary mean lies within the per-seed values and spread is a non-negative range (Req 6.4)."""
    summary = seed_summary(per_seed)
    values = list(per_seed.values())
    assert min(values) - 1e-9 <= summary.mean <= max(values) + 1e-9
    assert summary.spread >= 0.0
    assert abs(summary.spread - (max(values) - min(values))) <= 1e-9
