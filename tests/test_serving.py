"""Tests for serving (task 5.4; Req 6.4, 6.5, 6.6).

Unit tests run scripts/serve.sh with fake ``docker`` and ``curl`` executables on
PATH, and the VRAM monitor with nvidia-smi faked. The ``docker``-marked test
starts the real Serving_Endpoint and needs the model files from task 5.5.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from fts import gpu, settings, vram_monitor
from fts.model_client import client_from_config

REPO = Path(__file__).resolve().parent.parent
SERVE_SH = REPO / "scripts" / "serve.sh"


def fake_bin(tmp_path: Path, curl_ok: bool = True) -> Path:
    b = tmp_path / "bin"
    b.mkdir()
    log = tmp_path / "calls.log"
    for name, body in (("docker", f'echo "docker $*" >> {log}\nexit 0\n'), ("curl", f'echo "curl $*" >> {log}\nexit {0 if curl_ok else 7}\n')):
        p = b / name
        p.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return b


def run_serve(tmp_path, args, *, require_key, key=None, model_present=True, curl_ok=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "serve.yaml"
    text = (REPO / "config" / "serve.yaml").read_text(encoding="utf-8").replace("require_api_key: false", f"require_api_key: {'true' if require_key else 'false'}")
    cfg.write_text(text, encoding="utf-8")
    artifacts = tmp_path / "art"
    (artifacts / "models").mkdir(parents=True)
    if model_present:
        (artifacts / "models" / "ft-q4_k_m.gguf").write_bytes(b"gguf")
        (artifacts / "models" / "ft-f16.gguf").write_bytes(b"gguf")
    env = {k: v for k, v in os.environ.items() if k != "FTS_SERVE_API_KEY"}
    env.update({"PATH": f"{fake_bin(tmp_path, curl_ok)}:{env['PATH']}", "FTS_SERVE_CONFIG": str(cfg), "FTS_ARTIFACTS": str(artifacts), "FTS_SERVE_WAIT_S": "2"})
    if key is not None:
        env["FTS_SERVE_API_KEY"] = key
    out = subprocess.run(["bash", str(SERVE_SH), *args], capture_output=True, text=True, env=env)
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8") if (tmp_path / "calls.log").exists() else ""
    return out, calls


def test_missing_secret_fails_before_starting_and_names_the_variable(tmp_path):
    out, calls = run_serve(tmp_path, ["q4"], require_key=True)
    assert out.returncode == 1 and "FTS_SERVE_API_KEY" in out.stderr and "not started" in out.stderr
    assert calls == ""  # docker was never invoked


def test_empty_secret_is_missing(tmp_path):
    out, calls = run_serve(tmp_path, ["q4"], require_key=True, key="")
    assert out.returncode == 1 and calls == ""


def test_required_key_is_passed_by_name_only(tmp_path):
    out, calls = run_serve(tmp_path, ["q4"], require_key=True, key="secret-value")
    assert out.returncode == 0, out.stderr
    override = (REPO / "runs" / "serve.api-key.yml").read_text(encoding="utf-8")
    assert "${FTS_SERVE_API_KEY}" in override and "secret-value" not in override and "secret-value" not in calls
    assert "-f runs/serve.api-key.yml" in calls


def test_start_waits_for_health(tmp_path):
    out, calls = run_serve(tmp_path, ["f16"], require_key=False)
    assert out.returncode == 0 and "Serving_Endpoint ready: ft-f16.gguf" in out.stdout
    assert "docker compose -f docker-compose.yml up -d --force-recreate serve" in calls and "curl -sf http://127.0.0.1:8080/health" in calls


def test_default_variant_is_q4_and_bad_variant_is_rejected(tmp_path):
    out, _ = run_serve(tmp_path / "a", [], require_key=False)
    assert "ft-q4_k_m.gguf" in out.stdout
    out, calls = run_serve(tmp_path / "b", ["int8"], require_key=False)
    assert out.returncode == 2 and "usage" in out.stderr and calls == ""


def test_missing_model_file_fails(tmp_path):
    out, calls = run_serve(tmp_path, ["q4"], require_key=False, model_present=False)
    assert out.returncode == 1 and "fts.quantize" in out.stderr and calls == ""


def test_health_timeout_fails_with_logs(tmp_path):
    out, calls = run_serve(tmp_path, ["q4"], require_key=False, curl_ok=False)
    assert out.returncode == 1 and "did not return 200" in out.stderr and "logs --tail 50 serve" in calls


def test_vram_monitor_records_peak_and_budget(tmp_path):
    readings = iter([gpu.GpuMemory(3000, 10240), gpu.GpuMemory(7000, 10240)])
    with vram_monitor.VramMonitor(tmp_path / "v.csv", interval_s=0.01, query=lambda: next(readings, gpu.GpuMemory(5000, 10240))) as mon:
        pass
    s = mon.summary()
    assert s["peak_mib"] == 7000 and s["total_mib"] == 10240 and s["within_budget"] and s["samples"] >= 2
    lines = (tmp_path / "v.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "unix_time,used_mib,total_mib" and len(lines) == s["samples"] + 1
    mon.check()


def test_vram_budget_decision():
    vram_monitor.assert_within_budget(10240, 10240)
    with pytest.raises(vram_monitor.VramBudgetExceeded, match="10300 MiB exceeds"):
        vram_monitor.assert_within_budget(10300, 10240)


def test_vram_monitor_cli(tmp_path, capsys):
    assert vram_monitor.main(["--out", str(tmp_path / "m.csv"), "--seconds", "0"], query=lambda: gpu.GpuMemory(100, 200), sleep=lambda s: None) == 0
    assert vram_monitor.main(["--out", str(tmp_path / "m.csv"), "--seconds", "0"], query=lambda: gpu.GpuMemory(300, 200), sleep=lambda s: None) == 1
    assert "OVER budget" in capsys.readouterr().out


@pytest.mark.docker
def test_serving_endpoint_answers_a_chat_completion():
    """Starts the real Serving_Endpoint (Q4_K_M) and gets a completion through ModelClient."""
    if shutil.which("docker") is None:
        pytest.fail("docker is required for this test")
    try:
        out = subprocess.run(["bash", str(SERVE_SH), "q4"], capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, out.stderr
        cfg = settings.load_yaml(REPO / "config" / "serve.yaml")
        with client_from_config({**cfg, "api_key_env": None}) as client:
            completion = client.complete([{"role": "system", "content": "Answer with JSON."}, {"role": "user", "content": "Request: wake me up at seven am"}], 64)
        assert completion.text and completion.completion_tokens > 0 and completion.model
    finally:
        subprocess.run(["docker", "compose", "-f", "docker-compose.yml", "stop", "serve"], cwd=REPO, capture_output=True)
