"""Feature: finetune-and-serve, Property 7: Hosted-baseline spend-cap enforcement.

For any sequence of per-request token counts, prices, and a Spend_Cap, the ledger
reports cumulative spend equal to the defining formula over completed requests
and never completes a request whose worst-case cost would push recorded spend
above the Spend_Cap; the request that would cross it is not sent.

Validates: Requirements 3.3, 3.4
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from fts.spend import Prices, SpendCapExceeded, SpendLedger

request = st.tuples(
    st.integers(min_value=0, max_value=50_000),  # prompt tokens (the worst-case estimate is at least this)
    st.integers(min_value=1, max_value=512),  # max completion tokens
).flatmap(lambda pm: st.tuples(st.just(pm[0]), st.just(pm[1]), st.integers(min_value=0, max_value=pm[1])))


@settings(max_examples=100, derandomize=True, deadline=None)
@given(
    requests=st.lists(request, min_size=0, max_size=40),
    prompt_price=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    completion_price=st.floats(min_value=0.0, max_value=80.0, allow_nan=False),
    cap=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
)
def test_spend_cap_enforcement(requests, prompt_price, completion_price, cap):
    prices = Prices(prompt_per_mtok=prompt_price, completion_per_mtok=completion_price)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = SpendLedger(Path(tmp) / "ledger.jsonl", cap_usd=cap, prices=prices)
        completed = []
        stopped_at = None
        for i, (prompt, max_out, actual_out) in enumerate(requests):
            try:
                ledger.check_tokens(prompt, max_out)
            except SpendCapExceeded as exc:
                stopped_at = i
                assert exc.spent_usd + exc.worst_case_usd > cap
                break
            ledger.record(prompt, actual_out, model="m")
            completed.append((prompt, actual_out))

        expected = sum((p * prompt_price + c * completion_price) / 1_000_000 for p, c in completed)
        assert math.isclose(ledger.total_usd, expected, rel_tol=1e-9, abs_tol=1e-12)
        assert ledger.total_usd <= cap + 1e-12
        assert ledger.requests == len(completed)
        if stopped_at is not None:
            prompt, max_out, _ = requests[stopped_at]
            assert ledger.total_usd + prices.cost(prompt, max_out) > cap
        reloaded = SpendLedger(Path(tmp) / "ledger.jsonl", cap_usd=cap, prices=prices)
        assert math.isclose(reloaded.total_usd, ledger.total_usd, rel_tol=1e-12, abs_tol=1e-15)
