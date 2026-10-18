"""Property-based test for load-test summarization (task 5.5).

Feature: finetune-and-serve, Property 9: P50<=P95, percentiles computed over successes only, failed count equals failed outcomes
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from fts.loadtest import RequestOutcome, summarize_outcomes


@st.composite
def _outcomes(draw):
    """A list of request outcomes mixing successes and failures."""
    size = draw(st.integers(min_value=0, max_value=60))
    outcomes = []
    for _ in range(size):
        ok = draw(st.booleans())
        latency = draw(
            st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
        )
        tokens = draw(st.integers(min_value=0, max_value=4096))
        outcomes.append(
            RequestOutcome(ok=ok, latency_ms=latency, completion_tokens=tokens)
        )
    return outcomes


@given(outcomes=_outcomes())
def test_summarize_outcomes_properties(outcomes):
    """Feature: finetune-and-serve, Property 9: P50<=P95, percentiles computed over successes only, failed count equals failed outcomes."""
    stats = summarize_outcomes(outcomes)

    successes = [o for o in outcomes if o.ok]
    expected_failed = sum(1 for o in outcomes if not o.ok)

    # Failed count equals the number of failed outcomes.
    assert stats.failed_requests == expected_failed

    # P50 never exceeds P95.
    assert stats.p50_latency_ms <= stats.p95_latency_ms

    if successes:
        # Percentiles are computed over SUCCESSES only: they fall within the
        # range of successful latencies (never influenced by failed outcomes,
        # which carry latency 0.0 but are excluded).
        success_latencies = [o.latency_ms for o in successes]
        lo, hi = min(success_latencies), max(success_latencies)
        assert lo - 1e-9 <= stats.p50_latency_ms <= hi + 1e-9
        assert lo - 1e-9 <= stats.p95_latency_ms <= hi + 1e-9
        # Throughput is non-negative.
        assert stats.tokens_per_second >= 0.0
    else:
        # No successes -> degenerate zero stats (failed-only or empty input).
        assert stats.p50_latency_ms == 0.0
        assert stats.p95_latency_ms == 0.0
        assert stats.tokens_per_second == 0.0
