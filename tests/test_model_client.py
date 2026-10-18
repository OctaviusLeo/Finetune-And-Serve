"""Unit tests for fts.model_client with the HTTP layer faked (task 1.4; Req 0.13)."""

from __future__ import annotations

import json

import httpx
import pytest

from fts import model_client as mc
from fts import spend

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "set an alarm"}]


def ok_body(text="{}", model="gpt-6-luna-2026-09-01", usage=True):
    body = {"model": model, "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
    if usage:
        body["usage"] = {"prompt_tokens": 50, "completion_tokens": 7}
    return body


def make_client(handler, *, api_key_env=None, ledger=None, sleeps=None, **cfg):
    config = mc.ClientConfig(base_url="https://api.example.invalid/v1", model="gpt-6-luna", api_key_env=api_key_env, **cfg)
    return mc.ModelClient(
        config, ledger=ledger, transport=httpx.MockTransport(handler), sleep=(sleeps.append if sleeps is not None else lambda s: None)
    )


# --- missing secret -------------------------------------------------------------------------

def test_missing_secret_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("FTS_TEST_KEY", raising=False)
    calls = []
    with pytest.raises(mc.MissingSecretError) as excinfo:
        make_client(lambda r: calls.append(r), api_key_env="FTS_TEST_KEY")
    assert calls == []
    assert excinfo.value.env_var == "FTS_TEST_KEY" and "FTS_TEST_KEY" in str(excinfo.value)


def test_empty_secret_is_missing(monkeypatch):
    monkeypatch.setenv("FTS_TEST_KEY", "   ")
    with pytest.raises(mc.MissingSecretError):
        mc.read_secret("FTS_TEST_KEY")


def test_key_is_sent_as_bearer_and_never_in_repr(monkeypatch):
    monkeypatch.setenv("FTS_TEST_KEY", "sk-test-secret")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=ok_body())

    client = make_client(handler, api_key_env="FTS_TEST_KEY")
    client.complete(MESSAGES, 16)
    assert seen["auth"] == "Bearer sk-test-secret"
    assert "sk-test-secret" not in repr(client.config)


def test_no_key_needed_for_local_server():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=ok_body())

    make_client(handler).complete(MESSAGES, 8)
    assert seen["auth"] is None


# --- request body and response parsing ------------------------------------------------------

def test_payload_and_completion_fields():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body('{"intent": "alarm_set", "slots": []}'))

    client = make_client(handler, extra={"reasoning_effort": "none"}, seed=3)
    out = client.complete(MESSAGES, 128)
    assert seen["url"] == "https://api.example.invalid/v1/chat/completions"
    body = seen["body"]
    assert body["model"] == "gpt-6-luna" and body["messages"] == MESSAGES
    assert body["max_completion_tokens"] == 128 and body["temperature"] == 0.0 and body["seed"] == 3
    assert body["reasoning_effort"] == "none"
    assert out == mc.Completion(
        text='{"intent": "alarm_set", "slots": []}', prompt_tokens=50, completion_tokens=7,
        model="gpt-6-luna-2026-09-01", finish_reason="stop", usage_reported=True,
    )
    assert client.sent_params() == ["max_completion_tokens", "reasoning_effort", "seed", "temperature"]


def test_max_tokens_field_for_llama_server():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    make_client(handler, max_tokens_field="max_tokens", temperature=None, seed=None).complete(MESSAGES, 64)
    assert seen["body"]["max_tokens"] == 64 and "temperature" not in seen["body"] and "seed" not in seen["body"]


def test_missing_usage_charges_the_estimate():
    out = make_client(lambda r: httpx.Response(200, json=ok_body(usage=False))).complete(MESSAGES, 32)
    assert not out.usage_reported and out.completion_tokens == 32
    assert out.prompt_tokens == spend.estimate_prompt_tokens(MESSAGES)


# --- timeout, connection errors, retries ---------------------------------------------------

def test_timeout_surfaces_as_typed_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(mc.ModelTimeoutError, match="timed out"):
        make_client(handler, timeout_s=5).complete(MESSAGES, 8)


def test_connection_error_is_typed():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(mc.ModelConnectionError):
        make_client(handler).complete(MESSAGES, 8)


def test_retry_on_429_then_success_with_backoff():
    responses = [httpx.Response(429), httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(200, json=ok_body())]
    sleeps = []
    client = make_client(lambda r: responses.pop(0), sleeps=sleeps, backoff_s=1.0)
    assert client.complete(MESSAGES, 8).completion_tokens == 7
    assert sleeps == [1.0, 3.0]


def test_retries_stop_after_max_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="overloaded")

    sleeps = []
    with pytest.raises(mc.ModelHTTPError) as excinfo:
        make_client(handler, sleeps=sleeps, backoff_s=0.5).complete(MESSAGES, 8)
    assert excinfo.value.status == 503 and len(calls) == 5 and sleeps == [0.5, 1.0, 2.0, 4.0]


def test_non_retryable_error_raises_immediately():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(mc.ModelHTTPError) as excinfo:
        make_client(handler).complete(MESSAGES, 8)
    assert excinfo.value.status == 401 and len(calls) == 1


def test_bad_retry_after_falls_back_to_backoff():
    responses = [httpx.Response(500, headers={"retry-after": "soon"}), httpx.Response(200, json=ok_body())]
    sleeps = []
    make_client(lambda r: responses.pop(0), sleeps=sleeps, backoff_s=2.0).complete(MESSAGES, 8)
    assert sleeps == [2.0]


# --- rejected parameters -------------------------------------------------------------------

def test_rejected_parameter_is_dropped_once_and_recorded():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature'", "param": "temperature"}})
        return httpx.Response(200, json=ok_body())

    client = make_client(handler)
    client.complete(MESSAGES, 8)
    client.complete(MESSAGES, 8)
    assert client.dropped_params == ["temperature"]
    assert len(bodies) == 3 and "temperature" not in bodies[2] and "seed" in bodies[2]


def test_400_for_other_parameter_is_an_error():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad", "param": "messages"}})

    with pytest.raises(mc.ModelHTTPError):
        make_client(handler).complete(MESSAGES, 8)


def test_400_with_non_json_body_is_an_error():
    with pytest.raises(mc.ModelHTTPError):
        make_client(lambda r: httpx.Response(400, text="<html>")).complete(MESSAGES, 8)


# --- ledger integration --------------------------------------------------------------------

def test_ledger_records_completed_requests(tmp_path):
    ledger = spend.SpendLedger(tmp_path / "l.jsonl", cap_usd=1.0, prices=spend.Prices(0.125, 0.5))
    client = make_client(lambda r: httpx.Response(200, json=ok_body()), ledger=ledger)
    client.ledger_tag = "hosted_check"
    client.complete(MESSAGES, 8)
    assert ledger.requests == 1 and ledger.entries[0]["model"] == "gpt-6-luna-2026-09-01"
    assert ledger.entries[0]["tag"] == "hosted_check"


def test_ledger_refusal_sends_nothing(tmp_path):
    ledger = spend.SpendLedger(tmp_path / "l.jsonl", cap_usd=0.0, prices=spend.Prices(0.125, 0.5))
    calls = []
    client = make_client(lambda r: calls.append(r) or httpx.Response(200, json=ok_body()), ledger=ledger)
    with pytest.raises(spend.SpendCapExceeded):
        client.complete(MESSAGES, 8)
    assert calls == [] and ledger.requests == 0


def test_client_from_config_and_context_manager(monkeypatch):
    monkeypatch.setenv("FTS_TEST_KEY", "k")
    cfg = {"base_url": "http://127.0.0.1:8080/v1", "model": "ft", "api_key_env": "FTS_TEST_KEY", "timeout_s": 7,
           "max_tokens_field": "max_tokens", "extra": {"a": 1}}
    with mc.client_from_config(cfg, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body()))) as client:
        assert client.config.timeout_s == 7.0 and client.config.extra == {"a": 1}
        assert client.url == "http://127.0.0.1:8080/v1/chat/completions"
        assert client.complete(MESSAGES, 4).text == "{}"
