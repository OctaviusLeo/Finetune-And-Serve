"""Unit tests for the VRAM-budget decision and prerequisite checks (task 1.2; Req 1.2, 1.4, 1.5).

nvidia-smi is faked through the sampler's query function, and the model is a
tiny random Qwen2 on the CPU, so these run on a CPU-only CI runner.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from fts import dry_run, gpu, repro, settings
from fts.lora import tiny_qwen2

STAMP = repro.Stamp(command="python scripts/dry_run.py", seed=0, commit="d" * 40)


def _cfg(**overrides):
    cfg = dry_run.config_from_settings(settings.load_model_settings(), settings.load_train_settings())
    small = replace(cfg, steps=2, per_device_batch_size=2, max_seq_len=8, grad_accum=1)
    return replace(small, **overrides)


def _fake_query(used=2000, total=10240):
    return lambda: gpu.GpuMemory(used, total)


# --- Python prerequisite -------------------------------------------------------------------

def test_python_guard_accepts_311_and_newer():
    assert dry_run.check_python((3, 11, 0)) is None
    assert dry_run.check_python((3, 12, 3)) is None


def test_python_guard_reports_old_version():
    msg = dry_run.check_python((3, 10, 9))
    assert msg is not None and "3.10.9" in msg and "3.11" in msg


def test_main_fails_and_records_old_python(tmp_path, monkeypatch):
    env_md = tmp_path / "ENV.md"
    monkeypatch.setattr(dry_run.repro, "start_run", lambda seed: STAMP)
    monkeypatch.setattr(dry_run, "check_python", lambda: "Python 3.10.1 is older than the required 3.11")
    assert dry_run.main([], env_md=env_md, result_json=tmp_path / "r.json") == 1
    text = env_md.read_text(encoding="utf-8")
    assert "FAILED: Python 3.10.1 is older than the required 3.11" in text and STAMP.commit in text


# --- VRAM budget decision ------------------------------------------------------------------

def test_budget_within_and_at_limit():
    assert dry_run.budget_failure(5000, 9000, 10240) is None
    assert dry_run.budget_failure(10240, 10240, 10240) is None


def test_budget_flags_each_peak_over_total():
    msg = dry_run.budget_failure(10300, 10500, 10240)
    assert "torch peak reserved 10300 MiB" in msg and "nvidia-smi peak 10500 MiB" in msg and "10240" in msg
    only_smi = dry_run.budget_failure(5000, 10241, 10240)
    assert "nvidia-smi peak 10241" in only_smi and "torch" not in only_smi


def test_budget_without_total_fails():
    assert "no VRAM total" in dry_run.budget_failure(1, 1, 0)


def test_run_on_cpu_passes_within_budget():
    result = dry_run.run(_cfg(), device="cpu", model_factory=lambda c, d: tiny_qwen2(), query=_fake_query(), sampler_interval_s=0.01)
    assert result.ok and result.failure is None
    assert result.peak_smi_mib == 2000 and result.total_mib == 10240
    assert len(result.losses) == 2 and 0 < result.trainable_params < result.total_params


def test_run_reports_over_budget_peak():
    result = dry_run.run(_cfg(), device="cpu", model_factory=lambda c, d: tiny_qwen2(), query=_fake_query(used=11000), sampler_interval_s=0.01)
    assert not result.ok and "nvidia-smi peak 11000 MiB > total 10240 MiB" in result.failure


# --- out of memory -------------------------------------------------------------------------

class OutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError (matched by class name)."""


def test_oom_is_reported_with_the_limit():
    def factory(cfg, device):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    result = dry_run.run(_cfg(), device="cpu", model_factory=factory, query=_fake_query(), sampler_interval_s=0.01)
    assert not result.ok
    assert "CUDA out of memory" in result.failure and "limit 10240 MiB" in result.failure


def test_oom_detection_and_message_edge_cases():
    assert dry_run.is_oom(RuntimeError("CUDA error: out of memory"))
    assert not dry_run.is_oom(ValueError("shape mismatch"))
    assert "OutOfMemoryError" in dry_run.oom_failure(OutOfMemoryError(""), 0, 10240)


def test_other_errors_propagate():
    def factory(cfg, device):
        raise ValueError("bad config")

    with pytest.raises(ValueError):
        dry_run.run(_cfg(), device="cpu", model_factory=factory, query=_fake_query(), sampler_interval_s=0.01)


# --- end to end main() with fakes ----------------------------------------------------------

TABLE = "| 0 NVIDIA GeForce RTX 3080 | 2767MiB / 10240MiB |\n"


def _patch_main(monkeypatch, result):
    monkeypatch.setattr(dry_run.repro, "start_run", lambda seed: STAMP)
    monkeypatch.setattr(dry_run.settings, "pin_revision", lambda: replace(settings.load_model_settings(), revision="e" * 40))
    monkeypatch.setattr(dry_run.gpu, "run_nvidia_smi", lambda: TABLE)
    seen = {}

    def fake_run(cfg):
        seen["cfg"] = cfg
        return result

    monkeypatch.setattr(dry_run, "run", fake_run)
    return seen


def test_main_success_writes_evidence(tmp_path, monkeypatch):
    ok = dry_run.DryRunResult(ok=True, peak_reserved_mib=4321.0, peak_smi_mib=7000, total_mib=10240, trainable_params=10, total_params=100, losses=[1.0], seconds=2.0)
    seen = _patch_main(monkeypatch, ok)
    env_md, rj = tmp_path / "ENV.md", tmp_path / "r.json"
    assert dry_run.main(["--per-device-batch", "4", "--grad-accum", "4"], env_md=env_md, result_json=rj) == 0
    assert seen["cfg"].per_device_batch_size == 4 and seen["cfg"].grad_accum == 4 and seen["cfg"].revision == "e" * 40
    text = env_md.read_text(encoding="utf-8")
    for needle in ("**10240 MiB**", "PASSED", "4321 MiB", "7000 MiB", "Sequence length: 256", "Per-device batch size: 4", "e" * 40, STAMP.commit):
        assert needle in text
    data = json.loads(rj.read_text(encoding="utf-8"))
    assert data["result"]["ok"] is True and data["stamp"]["commit"] == STAMP.commit
    assert data["config"]["lora"]["target_modules"][0] == "q_proj"


def test_main_over_budget_exits_nonzero(tmp_path, monkeypatch, capsys):
    bad = dry_run.DryRunResult(ok=False, failure="VRAM over budget: nvidia-smi peak 11000 MiB > total 10240 MiB", total_mib=10240)
    _patch_main(monkeypatch, bad)
    env_md = tmp_path / "ENV.md"
    assert dry_run.main([], env_md=env_md, result_json=tmp_path / "r.json") == 1
    assert "11000 MiB > total 10240 MiB" in capsys.readouterr().err
    assert "FAILED: VRAM over budget" in env_md.read_text(encoding="utf-8")
