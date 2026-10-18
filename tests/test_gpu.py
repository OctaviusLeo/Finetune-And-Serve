"""Unit tests for fts.gpu with nvidia-smi faked."""

from __future__ import annotations

import subprocess

import pytest

from fts import gpu

TABLE = "|  0%   45C    P8    30W / 320W |   2767MiB / 10240MiB |      3%      Default |"


def test_parse_memory_csv():
    assert gpu.parse_memory_csv("2767, 10240\n") == gpu.GpuMemory(2767, 10240)
    assert gpu.parse_memory_csv("garbage\n 100 MiB, 200 MiB\n") == gpu.GpuMemory(100, 200)
    with pytest.raises(ValueError):
        gpu.parse_memory_csv("no numbers")


def test_parse_total_from_table():
    assert gpu.parse_total_mib(TABLE) == 10240
    assert gpu.parse_total_mib("nothing") is None


def test_within_budget():
    assert gpu.within_budget(10240, 10240)
    assert not gpu.within_budget(10241, 10240)
    assert not gpu.within_budget(1, 0)


def test_query_memory_passes_query_args():
    seen = []

    def runner(args):
        seen.append(args)
        return "10, 20\n"

    assert gpu.query_memory(runner) == gpu.GpuMemory(10, 20)
    assert seen == [gpu.QUERY_ARGS]


def test_run_nvidia_smi_success_and_failure(monkeypatch):
    def ok(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(cmd, 0, stdout="out\n", stderr="")

    monkeypatch.setattr(gpu.subprocess, "run", ok)
    assert gpu.run_nvidia_smi(["-L"]) == "out\n"

    def missing(cmd, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="nvidia-smi failed"):
        gpu.run_nvidia_smi()


def test_sampler_records_peak_and_skips_errors():
    readings = iter([gpu.GpuMemory(100, 1000), RuntimeError("x"), gpu.GpuMemory(300, 1000)])
    seen = []

    def query():
        r = next(readings, gpu.GpuMemory(200, 1000))
        if isinstance(r, Exception):
            raise r
        return r

    s = gpu.MemorySampler(interval_s=0.01, query=query, on_sample=lambda t, m: seen.append(m))
    s.sample_once()
    s.sample_once()
    s.sample_once()
    assert s.errors == 1 and s.peak_mib == 300 and s.total_mib == 1000 and len(seen) == 2


def test_sampler_thread_takes_readings():
    s = gpu.MemorySampler(interval_s=0.01, query=lambda: gpu.GpuMemory(5, 10))
    with s:
        pass
    assert len(s.samples) >= 2 and s.peak_mib == 5


def test_empty_sampler_reports_zero():
    s = gpu.MemorySampler(query=lambda: gpu.GpuMemory(0, 0))
    assert s.peak_mib == 0 and s.total_mib == 0
