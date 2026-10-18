"""Unit tests for fts.spend (ledger, prices, worst-case pre-check)."""

from __future__ import annotations

import json

import pytest

from fts import settings, spend

PRICES = spend.Prices(prompt_per_mtok=0.125, completion_per_mtok=0.50)


def test_cost_formula():
    assert PRICES.cost(1_000_000, 0) == pytest.approx(0.125)
    assert PRICES.cost(1000, 100) == pytest.approx((1000 * 0.125 + 100 * 0.5) / 1e6)


def test_prices_from_hosted_config():
    cfg = settings.load_yaml(settings.CONFIG_DIR / "hosted.yaml")
    prices = spend.prices_from_config(cfg)
    assert prices == spend.Prices(prompt_per_mtok=0.042, completion_per_mtok=0.0)
    assert cfg["spend_cap_usd"] == 2.00 and cfg["model"] == "jev-1.13.0"
    with_cache_write = {"prices_usd_per_1m_tokens": {"input": 0.10, "cache_write": 0.125, "output": 0.5}}
    assert spend.prices_from_config(with_cache_write) == spend.Prices(prompt_per_mtok=0.125, completion_per_mtok=0.5)
    assert spend.estimate_text_tokens("x" * 10) == 4


def test_estimate_prompt_tokens():
    msgs = [{"role": "system", "content": "abc"}, {"role": "user", "content": "abcd"}]
    assert spend.estimate_prompt_tokens(msgs) == 1 + 2 + 2 * spend.TOKENS_PER_MESSAGE


def test_record_appends_and_reloads(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = spend.SpendLedger(path, cap_usd=1.0, prices=PRICES, clock=lambda: 123.0)
    e1 = ledger.record(1000, 10, "m-1", tag="check")
    ledger.record(2000, 20, "m-1")
    assert e1["cost_usd"] == pytest.approx(PRICES.cost(1000, 10)) and e1["tag"] == "check"
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[1]["total_usd"] == pytest.approx(PRICES.cost(3000, 30))
    reloaded = spend.SpendLedger(path, cap_usd=1.0, prices=PRICES)
    assert reloaded.requests == 2 and reloaded.total_usd == pytest.approx(ledger.total_usd)
    s = reloaded.summary()
    assert s["prompt_tokens"] == 3000 and s["completion_tokens"] == 30 and s["spend_cap_usd"] == 1.0


def test_check_refuses_request_that_could_cross_the_cap(tmp_path):
    ledger = spend.SpendLedger(tmp_path / "l.jsonl", cap_usd=PRICES.cost(1000, 128), prices=PRICES)
    ledger.check_tokens(1000, 128)  # exactly at the cap is allowed
    with pytest.raises(spend.SpendCapExceeded) as excinfo:
        ledger.check_tokens(1001, 128)
    err = excinfo.value
    assert err.spent_usd == 0 and err.cap_usd == ledger.cap_usd and "not sent" in str(err)


def test_check_with_messages_uses_estimate(tmp_path):
    ledger = spend.SpendLedger(tmp_path / "l.jsonl", cap_usd=0.0, prices=PRICES)
    with pytest.raises(spend.SpendCapExceeded):
        ledger.check([{"role": "user", "content": "hi"}], 1)
