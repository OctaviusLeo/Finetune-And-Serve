"""Unit tests for fts.hosted (the M0 hosted API proof against Jev) with HTTP faked."""

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
    assert body["model"] == "jev-1.13.0" and body["state"] == hosted.CHECK_STATE and set(body["questions"]) == {"asks_alarm"}
    assert request.headers["authorization"] == "Bearer ts-test"
    return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"asks_alarm": {"type": "noul", "noul": 0.98}},
                                     "usage": {"input_tokens": 60, "output_tokens": 12}})


def test_missing_key_exits_2_without_request(root, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = []
    rc = hosted.main(root=root, env_md=root / "ENV.md", transport=httpx.MockTransport(lambda r: calls.append(r)))
    assert rc == 2 and calls == []
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
    assert not (root / "results" / "hosted_ledger.jsonl").exists()


def test_check_records_model_answer_tokens_and_cost(root, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    env_md = root / "ENV.md"
    assert hosted.main(root=root, env_md=env_md, transport=httpx.MockTransport(_reply)) == 0
    data = json.loads((root / "results" / "hosted_check.json").read_text(encoding="utf-8"))
    check = data["check"]
    assert check["response_model"] == "jev-1.13.0" and check["answer"] == 0.98
    assert check["input_tokens"] == 60 and check["cost_usd"] == pytest.approx(60 * 0.042 / 1e6)
    assert data["ledger"]["requests"] == 1 and data["stamp"]["commit"] == STAMP.commit
    text = env_md.read_text(encoding="utf-8")
    assert "**Spend_Cap: $2.00**" in text and "`model` field was `jev-1.13.0`" in text and STAMP.commit in text and "ts-test" not in text
    ledger_lines = (root / "results" / "hosted_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 1 and json.loads(ledger_lines[0])["tag"] == "hosted_check"


def test_http_failure_exits_1(root, monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    rc = hosted.main(root=root, env_md=root / "ENV.md", transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no")))
    assert rc == 1 and "HTTP 401" in capsys.readouterr().err


def test_spend_cap_refusal_exits_1(root, monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    cfg_path = root / "config" / "hosted.yaml"
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8").replace("spend_cap_usd: 2.00", "spend_cap_usd: 0.0"), encoding="utf-8")
    calls = []
    rc = hosted.main(root=root, env_md=root / "ENV.md", transport=httpx.MockTransport(lambda r: calls.append(r)))
    assert rc == 1 and calls == [] and "Spend_Cap reached" in capsys.readouterr().err
