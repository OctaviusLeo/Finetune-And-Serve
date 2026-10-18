"""Feature: finetune-and-serve, Property 5: Bootstrap CI brackets the point estimate.

For any non-empty sample array of values in [0, 1] and any seed, the 95%
bootstrap CI computed from at least 1,000 resamples satisfies
lower <= point <= upper, and recomputing with the same seed yields an identical
interval within 1e-6.

Validates: Requirements 3.2, 5.1, 5.7
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from fts.eval_core import MIN_RESAMPLES, bootstrap_ci

samples_strategy = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=200)


@settings(max_examples=100, derandomize=True, deadline=None)
@given(samples=samples_strategy, seed=st.integers(min_value=0, max_value=2**31 - 1), resamples=st.sampled_from([MIN_RESAMPLES, 1500]))
def test_bootstrap_ci_brackets_point_and_is_deterministic(samples, seed, resamples):
    """Feature: finetune-and-serve, Property 5: Bootstrap CI brackets the point estimate."""
    ci = bootstrap_ci(samples, seed=seed, resamples=resamples)
    assert 0.0 <= ci.lower <= ci.point <= ci.upper <= 1.0
    again = bootstrap_ci(samples, seed=seed, resamples=resamples)
    assert abs(ci.point - again.point) <= 1e-6
    assert abs(ci.lower - again.lower) <= 1e-6
    assert abs(ci.upper - again.upper) <= 1e-6
