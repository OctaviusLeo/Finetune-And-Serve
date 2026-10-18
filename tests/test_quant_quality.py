"""Unit tests for the pre/post-quantization quality runner with the server and nvidia-smi faked."""

from __future__ import annotations

import json
import shutil

import httpx
import pytest

from fts import data, gpu, quant_quality, repro
from fts.model_client import ClientConfig, ModelClient

STAMP = repro.Stamp(command="python -m fts.quant_quality", seed=0, commit="b" * 40)
GOLD = ['{"intent": "alarm_set", "slots": []}', '{"intent": "play_music", "slots": []}']
TEST = tuple(data.Example(str(i), "test", f"u{i}", g) for i, g in enumerate(GOLD))


def client_factory(answers):
    """answers: variant currently served -> function(utterance) -> response."""
    state = {"variant": None}

    def handler(request):
        utt = json.loads(request.content)["messages"][-1]["content"].split("Request: ")[-1]
        return answers[state["variant"]](utt)

    def factory(cfg):
        return ModelClient(ClientConfig(base_url=cfg["base_url"], model=cfg["model"], max_tokens_field="max_tokens"), transport=httpx.MockTransport(handler))

    return state, factory


def ok(text):
    return httpx.Response(200, json={"model": "fts-ft", "choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 20, "completion_tokens": 8}})


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    shutil.copy(quant_quality.REPO_ROOT / "config" / "serve.yaml", r / "config" / "serve.yaml")
    shutil.copytree(quant_quality.REPO_ROOT / "config" / "prompts", r / "config" / "prompts")
    data.write_splits(data.Splits((), (), TEST), r / "data" / "massive_en")
    monkeypatch.setattr(quant_quality.repro, "start_run", lambda seed: STAMP)
    return r


def write_finetuned(root):
    repro.write_json(root / "results" / "finetuned.json", {"serving_seed": 2, "seeds": [
        {"seed": 1, "run_id": "B-seed1", "metric": {"point": 0.9, "lower": 0.89, "upper": 0.91}},
        {"seed": 2, "run_id": "B-seed2", "metric": {"point": 0.92, "lower": 0.91, "upper": 0.93}}]}, STAMP)


def test_quality_for_both_variants_with_reference_and_vram(root):
    write_finetuned(root)
    gold = {e.inputs: e.target for e in TEST}
    state, factory = client_factory({"f16": lambda u: ok(gold[u]), "q4": lambda u: ok(gold[u] if u == "u0" else "nope")})
    events = []
    rc = quant_quality.main(root=root, serve_fn=lambda v: events.append(("serve", v)) or state.update(variant=v),
                            stop_fn=lambda: events.append(("stop",)), client_factory=factory, vram_query=lambda: gpu.GpuMemory(4000, 10240))
    assert rc == 0 and events == [("serve", "f16"), ("stop",), ("serve", "q4"), ("stop",)]
    q = json.loads((root / "results" / "quant_quality.json").read_text(encoding="utf-8"))
    assert [v["method"] for v in q["variants"]] == ["gguf-f16", "gguf-q4_k_m"]
    assert q["variants"][0]["metric"]["point"] == 1.0 and q["variants"][1]["metric"]["point"] == 0.5
    assert q["reference_hf"]["seed"] == 2 and q["reference_hf"]["metric"]["point"] == 0.92
    assert q["variants"][1]["vram"]["peak_mib"] == 4000 and q["peak_vram_within_total"] is True and q["stamp"]["commit"] == STAMP.commit


def test_failed_requests_score_zero_and_are_counted(root):
    write_finetuned(root)
    gold = {e.inputs: e.target for e in TEST}
    state, factory = client_factory({"f16": lambda u: ok(gold[u]), "q4": lambda u: httpx.Response(400, json={"error": {"message": "bad"}})})
    rc = quant_quality.main(root=root, serve_fn=lambda v: state.update(variant=v), stop_fn=lambda: None, client_factory=factory,
                            vram_query=lambda: gpu.GpuMemory(4000, 10240))
    q = json.loads((root / "results" / "quant_quality.json").read_text(encoding="utf-8"))
    assert rc == 0 and q["variants"][1]["failed_requests"] == 2 and q["variants"][1]["metric"]["point"] == 0.0


def test_vram_over_total_fails(root):
    write_finetuned(root)
    gold = {e.inputs: e.target for e in TEST}
    state, factory = client_factory({"f16": lambda u: ok(gold[u]), "q4": lambda u: ok(gold[u])})
    rc = quant_quality.main(root=root, serve_fn=lambda v: state.update(variant=v), stop_fn=lambda: None, client_factory=factory,
                            vram_query=lambda: gpu.GpuMemory(10300, 10240))
    assert rc == 1


def test_server_start_failure_and_missing_reference(root, capsys):
    assert quant_quality.main(root=root, serve_fn=lambda v: None, stop_fn=lambda: None) == 1
    assert "finetuned.json is missing" in capsys.readouterr().err
    write_finetuned(root)

    def fail(v):
        raise RuntimeError("scripts/serve.sh f16 failed (1): no model")

    stops = []
    assert quant_quality.main(root=root, serve_fn=fail, stop_fn=lambda: stops.append(1)) == 1
    assert "no model" in capsys.readouterr().err


def test_serve_and_stop_call_the_scripts(monkeypatch, tmp_path):
    calls = []

    class Done:
        def __init__(self, rc):
            self.returncode, self.stderr = rc, "err"

    monkeypatch.setattr(quant_quality.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or Done(0))
    quant_quality.serve("q4", tmp_path)
    quant_quality.stop(tmp_path)
    assert calls == [["bash", "scripts/serve.sh", "q4"], ["docker", "compose", "-f", "docker-compose.yml", "stop", "serve"]]
    monkeypatch.setattr(quant_quality.subprocess, "run", lambda cmd, **kw: Done(1))
    with pytest.raises(RuntimeError, match="failed"):
        quant_quality.serve("f16", tmp_path)


def test_default_client_uses_key_only_when_required():
    cfg = quant_quality.settings.load_yaml(quant_quality.REPO_ROOT / "config" / "serve.yaml")
    client = quant_quality.default_client(cfg)
    assert client.config.api_key_env is None and client.config.max_tokens_field == "max_tokens" and client.config.model == "fts-ft"
