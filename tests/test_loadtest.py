"""Unit tests for the load test (task 5.7; Req 7.3, 7.4, 7.6)."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import httpx
import pytest

from fts import data, gpu, loadtest, repro
from fts.loadtest import LevelReport, RequestOutcome
from fts.model_client import ClientConfig, ModelClient

STAMP = repro.Stamp(command="python scripts/loadtest.py", seed=0, commit="c" * 40)
REPO = Path(__file__).resolve().parent.parent


def level(c, sustained):
    return LevelReport(c, 100.0, 200.0, 50.0, 0 if sustained else 1, sustained)


def test_sustaining_criterion_on_fixtures():
    fast = loadtest.summarize([RequestOutcome(True, 900.0, 30)] * 10, wall_seconds=2.0)
    slow = loadtest.summarize([RequestOutcome(True, 1500.0, 30)] * 18 + [RequestOutcome(True, 2600.0, 30)] * 2, wall_seconds=4.0)
    failing = loadtest.summarize([RequestOutcome(True, 100.0, 30)] * 9 + [RequestOutcome(False, 60000.0, 0)], wall_seconds=1.0)
    assert loadtest.is_sustained(fast, 2000) and fast.tokens_per_second == 150.0
    assert slow.p95_latency_ms > 2000 and not loadtest.is_sustained(slow, 2000)
    assert failing.failed_requests == 1 and not loadtest.is_sustained(failing, 2000)
    assert not loadtest.is_sustained(loadtest.summarize([], 1.0), 2000)


def test_maximum_concurrency_is_the_highest_sustained_level():
    assert loadtest.max_sustained([level(1, True), level(2, True), level(4, False), level(8, True), level(16, False)]) == 8
    assert loadtest.max_sustained([level(1, False)]) == 0


def test_run_load_test_is_non_interactive_and_stamps_every_level():
    sent = []

    def send(utterance):
        sent.append(utterance)
        return RequestOutcome(True, 50.0 if len(utterance) < 3 else 3000.0, 10)

    report = loadtest.run_load_test(send, ["a", "bb", "ccc"], [1, 2], seed=0, warmup=2, timed=6, p95_limit_ms=2000,
                                    command=STAMP.command, commit=STAMP.commit, log=lambda m: None)
    assert len(sent) == 2 * (2 + 6)
    assert [lv.concurrency_level for lv in report.levels] == [1, 2]
    for lv in report.levels:
        assert lv.stamp == {"command": "python scripts/loadtest.py", "seed": 0, "concurrency": lv.concurrency_level, "commit": STAMP.commit}
        assert lv.successful_requests == 6 and lv.p50_latency_ms <= lv.p95_latency_ms
    d = report.to_dict()
    assert d["commit"] == STAMP.commit and d["max_concurrent_requests"] == report.max_concurrent_requests and len(d["levels"]) == 2


def test_prompts_are_drawn_with_the_seed():
    a = loadtest.draw_prompts(["x", "y", "z"], 10, random.Random(0))
    assert a == loadtest.draw_prompts(["x", "y", "z"], 10, random.Random(0)) and len(a) == 10


def test_run_level_counts_and_times():
    outcomes, wall = loadtest.run_level(lambda u: RequestOutcome(u != "bad", 10.0, 5), ["w", "good", "bad", "good"], concurrency=2, warmup=1)
    assert [o.ok for o in outcomes] == [True, False, True] and wall >= 0


def test_make_sender_times_and_marks_failures():
    def handler(request):
        content = json.loads(request.content)["messages"][-1]["content"]
        if "fail" in content:
            return httpx.Response(500, text="err")
        return httpx.Response(200, json={"model": "fts-ft", "choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}})

    client = ModelClient(ClientConfig(base_url="http://x/v1", model="fts-ft", max_attempts=1), transport=httpx.MockTransport(handler))
    send = loadtest.make_sender(client, lambda u: [{"role": "user", "content": u}], 128)
    good, bad = send("hello"), send("fail")
    assert good.ok and good.completion_tokens == 7 and good.latency_ms >= 0
    assert not bad.ok and bad.completion_tokens == 0


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    for f in ("serve.yaml", "loadtest.yaml"):
        shutil.copy(REPO / "config" / f, r / "config" / f)
    shutil.copytree(REPO / "config" / "prompts", r / "config" / "prompts")
    data.write_splits(data.Splits((), (), (data.Example("1", "test", "wake me", "{}"),)), r / "data" / "massive_en")
    monkeypatch.setattr(repro, "start_run", lambda seed: STAMP)
    return r


def test_main_writes_results_and_stops_the_server(root):
    events = []
    client = ModelClient(ClientConfig(base_url="http://x/v1", model="fts-ft", max_attempts=1), transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"model": "fts-ft", "choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}})))
    rc = loadtest.main([], root=root, serve_fn=lambda v: events.append(("serve", v)), stop_fn=lambda: events.append(("stop",)), client=client,
                       vram_query=lambda: gpu.GpuMemory(5000, 10240))
    assert rc == 0 and events == [("serve", "q4"), ("stop",)]
    out = json.loads((root / "results" / "loadtest.json").read_text(encoding="utf-8"))
    assert [lv["concurrency_level"] for lv in out["levels"]] == [1, 2, 4, 8, 16, 32]
    assert out["max_concurrent_requests"] == 32 and out["vram"]["within_budget"] and out["stamp"]["commit"] == STAMP.commit
    assert all(lv["successful_requests"] == 128 for lv in out["levels"])


def test_main_fails_when_server_does_not_start(root, capsys):
    def fail(v):
        raise RuntimeError("scripts/serve.sh q4 failed (1): missing model")

    assert loadtest.main([], root=root, serve_fn=fail, stop_fn=lambda: None, client=object()) == 1
    assert "missing model" in capsys.readouterr().err
