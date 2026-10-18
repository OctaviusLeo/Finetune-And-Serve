"""Unit tests for fts.jev_client with the HTTP layer faked."""

from __future__ import annotations

import json

import httpx
import pytest

from fts import jev_client, spend
from fts.model_client import MissingSecretError, ModelConnectionError, ModelHTTPError, ModelTimeoutError

STATE = {"request": "wake me up at five am"}
QUESTIONS = {"q": {"type": "noul", "instructions": "Does `request` ask for an alarm?"}}


def ok(model="jev-1.13.0", input_tokens=50, output_tokens=7):
    return httpx.Response(200, json={"model": model, "answers": {"q": {"type": "noul", "noul": 0.9}},
                                     "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})


def make(handler, monkeypatch, ledger=None, sleeps=None, **kw):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    return jev_client.JevClient("jev-1.13.0", transport=httpx.MockTransport(handler), ledger=ledger,
                                sleep=(sleeps.append if sleeps is not None else lambda s: None), **kw)


def test_missing_or_blank_key_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(MissingSecretError) as excinfo:
        jev_client.JevClient("jev-1.13.0", transport=httpx.MockTransport(lambda r: pytest.fail("request sent")))
    assert excinfo.value.env_var == "TYPESAFE_API_KEY"
    monkeypatch.setenv("TYPESAFE_API_KEY", "  ")
    with pytest.raises(MissingSecretError):
        jev_client.JevClient("jev-1.13.0")


def test_request_shape_and_response(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return ok()

    with make(handler, monkeypatch) as client:
        r = client.ask(STATE, QUESTIONS)
        assert client.url == jev_client.DEFAULT_URL and "ts-secret" not in client.body(STATE, QUESTIONS)
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone" and seen["auth"] == "Bearer ts-secret"
    assert seen["body"] == {"model": "jev-1.13.0", "state": STATE, "questions": QUESTIONS}
    assert r == jev_client.JevResponse(answers={"q": {"type": "noul", "noul": 0.9}}, input_tokens=50, output_tokens=7, model="jev-1.13.0")


def test_retries_429_and_529_with_backoff(monkeypatch):
    responses = [httpx.Response(429), httpx.Response(529, headers={"retry-after": "2"}), ok()]
    sleeps = []
    assert make(lambda r: responses.pop(0), monkeypatch, sleeps=sleeps, backoff_s=1.0).ask(STATE, QUESTIONS).input_tokens == 50
    assert sleeps == [1.0, 2.0]


def test_retries_stop_after_max_attempts_and_bad_retry_after(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, headers={"retry-after": "soon"}, text="busy")

    sleeps = []
    with pytest.raises(ModelHTTPError) as excinfo:
        make(handler, monkeypatch, sleeps=sleeps, backoff_s=0.5, max_attempts=3).ask(STATE, QUESTIONS)
    assert excinfo.value.status == 503 and len(calls) == 3 and sleeps == [0.5, 1.0]


def test_non_retryable_status_raises(monkeypatch):
    with pytest.raises(ModelHTTPError) as excinfo:
        make(lambda r: httpx.Response(422, json={"detail": "criteria missing"}), monkeypatch).ask(STATE, QUESTIONS)
    assert excinfo.value.status == 422 and "criteria" in excinfo.value.body


def test_timeout_and_connection_errors_are_typed(monkeypatch):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ModelTimeoutError):
        make(slow, monkeypatch).ask(STATE, QUESTIONS)
    with pytest.raises(ModelConnectionError):
        make(down, monkeypatch).ask(STATE, QUESTIONS)


def test_ledger_checks_then_records(monkeypatch, tmp_path):
    ledger = spend.SpendLedger(tmp_path / "l.jsonl", cap_usd=1.0, prices=spend.Prices(0.042, 0.0))
    client = make(lambda r: ok(input_tokens=1000, output_tokens=99), monkeypatch, ledger=ledger, ledger_tag="hosted_few_shot")
    client.ask(STATE, QUESTIONS)
    assert ledger.requests == 1 and ledger.entries[0]["tag"] == "hosted_few_shot" and ledger.entries[0]["model"] == "jev-1.13.0"
    assert ledger.total_usd == pytest.approx(1000 * 0.042 / 1e6)


def test_ledger_refusal_sends_nothing(monkeypatch, tmp_path):
    ledger = spend.SpendLedger(tmp_path / "l.jsonl", cap_usd=0.0, prices=spend.Prices(0.042, 0.0))
    calls = []
    with pytest.raises(spend.SpendCapExceeded):
        make(lambda r: calls.append(r) or ok(), monkeypatch, ledger=ledger).ask(STATE, QUESTIONS)
    assert calls == []


def test_client_from_config(monkeypatch):
    from fts import hosted

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    c = jev_client.client_from_config(hosted.load_config(), ledger_tag="x")
    assert c.model == "jev-1.13.0" and c.url == "https://api.typesafe.ai/v1/systemone" and c.timeout_s == 120.0 and c.ledger_tag == "x"
