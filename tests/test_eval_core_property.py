"""Property-based test for the bootstrap CI (task 4.2).

Feature: finetune-and-serve, Property 5: Bootstrap CI brackets the point and is seed-deterministic
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from fts.eval_core import bootstrap_ci

_samples = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=40,
)


@given(
    samples=_samples,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_ci_brackets_point_and_is_seed_deterministic(samples, seed):
    """Feature: finetune-and-serve, Property 5: Bootstrap CI brackets the point and is seed-deterministic."""
    ci = bootstrap_ci(samples, seed=seed)

    # The interval always brackets the point estimate.
    assert ci.lower <= ci.point <= ci.upper

    # Same seed yields an identical interval within 1e-6.
    again = bootstrap_ci(samples, seed=seed)
    assert abs(ci.point - again.point) <= 1e-6
    assert abs(ci.lower - again.lower) <= 1e-6
    assert abs(ci.upper - again.upper) <= 1e-6
