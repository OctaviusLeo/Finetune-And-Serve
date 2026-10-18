"""Feature: finetune-and-serve, Property 9: Load-test latency percentile ordering and failed-request exclusion.

For any set of per-request outcomes (latency samples plus failed flags), the
reported P50 is at most the reported P95, both are computed only over
successful requests, and the reported failed-request count equals the number
of failed outcomes.

Validates: Requirements 7.1, 7.5
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from fts.loadtest import RequestOutcome, summarize

outcome = st.builds(RequestOutcome, ok=st.booleans(), latency_ms=st.floats(min_value=0.0, max_value=60_000.0, allow_nan=False),
                    completion_tokens=st.integers(min_value=0, max_value=128))


@settings(max_examples=100, derandomize=True, deadline=None)
@given(outcomes=st.lists(outcome, max_size=200), wall=st.floats(min_value=0.001, max_value=600.0, allow_nan=False))
def test_percentile_ordering_and_failed_exclusion(outcomes, wall):
    """Feature: finetune-and-serve, Property 9: Load-test latency percentile ordering and failed-request exclusion."""
    stats = summarize(outcomes, wall)
    ok = [o for o in outcomes if o.ok]
    assert stats.failed_requests == len(outcomes) - len(ok) and stats.successful_requests == len(ok)
    assert stats.p50_latency_ms <= stats.p95_latency_ms
    if ok:
        lat = [o.latency_ms for o in ok]
        assert abs(stats.p95_latency_ms - float(np.percentile(lat, 95))) <= 1e-9
        assert min(lat) <= stats.p50_latency_ms <= max(lat)
        assert abs(stats.tokens_per_second - sum(o.completion_tokens for o in ok) / wall) <= 1e-6
    else:
        assert stats.p50_latency_ms == stats.p95_latency_ms == stats.tokens_per_second == 0.0
