"""Unit tests for fts.hosted (the M0 hosted API proof) with HTTP faked."""

from __future__ import annotations

import json
import shutil

import httpx
import pytest

from fts import hosted, repro

STAMP = repro.Stamp(command="python scripts/hosted_check.py", seed=0, commit="f" * 40)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    shutil.copy(hosted.HOSTED_YAML, tmp_path / "config" / "hosted.yaml")
    monkeypatch.setattr(hosted.repro, "start_run", lambda seed: STAMP)
    return tmp_path


def _reply(request):
    body = json.loads(request.content)
    if "temperature" in body:
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature'", "param": "temperature"}})
    return httpx.Response(200, json={
        "model": "gpt-6-luna-2026-09-30",
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 2},
    })


def test_missing_key_exits_2_without_request(root, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []
    rc = hosted.main(root=root, env_md=root / "ENV.md", transport=httpx.MockTransport(lambda r: calls.append(r)))
    assert rc == 2 and calls == []
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (root / "results" / "hosted_ledger.jsonl").exists()


def test_check_records_model_params_tokens_and_cost(root, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    env_md = root / "ENV.md"
    assert hosted.main(root=root, env_md=env_md, transport=httpx.MockTransport(_reply)) == 0
    data = json.loads((root / "results" / "hosted_check.json").read_text(encoding="utf-8"))
    check = data["check"]
    assert check["response_model"] == "gpt-6-luna-2026-09-30" and check["text"] == "OK"
    assert check["dropped_params"] == ["temperature"]
    assert "temperature" not in check["accepted_params"] and "reasoning_effort" in check["accepted_params"]
    assert check["cost_usd"] == pytest.approx((30 * 0.125 + 2 * 0.5) / 1e6)
    assert data["ledger"]["requests"] == 1 and data["stamp"]["commit"] == STAMP.commit
    text = env_md.read_text(encoding="utf-8")
    assert "**Spend_Cap: $2.00**" in text and "gpt-6-luna-2026-09-30" in text and STAMP.commit in text
    ledger_lines = (root / "results" / "hosted_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 1 and json.loads(ledger_lines[0])["tag"] == "hosted_check"


def test_http_failure_exits_1(root, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    rc = hosted.main(root=root, env_md=root / "ENV.md", transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no")))
    assert rc == 1 and "HTTP 401" in capsys.readouterr().err


def test_spend_cap_refusal_exits_1(root, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg_path = root / "config" / "hosted.yaml"
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8").replace("spend_cap_usd: 2.00", "spend_cap_usd: 0.0"), encoding="utf-8")
    calls = []
    rc = hosted.main(root=root, env_md=root / "ENV.md", transport=httpx.MockTransport(lambda r: calls.append(r)))
    assert rc == 1 and calls == [] and "Spend_Cap reached" in capsys.readouterr().err
