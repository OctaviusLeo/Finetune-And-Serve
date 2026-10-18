"""Unit tests for the GPU-free wait (fts.gpu.wait_until_at_most) and scripts/gpu_check.py."""

from __future__ import annotations

from fts import gpu
from scripts import gpu_check


def _readings(*values):
    it = iter(values)
    last = values[-1]
    return lambda: gpu.GpuMemory(next(it, last), 10240)


def test_wait_returns_immediately_when_free():
    sleeps = []
    r = gpu.wait_until_at_most(3072, query=_readings(2700), sleep=sleeps.append, log=lambda m: None)
    assert r.ok and r.used_mib == 2700 and sleeps == [] and r.waited_s == 0


def test_wait_polls_in_steps_until_free():
    sleeps, logs = [], []
    r = gpu.wait_until_at_most(3072, query=_readings(4800, 4800, 2900), sleep=sleeps.append, step_s=300, max_wait_s=1200, log=logs.append)
    assert r.ok and r.readings == (4800, 4800, 2900) and sleeps == [300, 300] and len(logs) == 2


def test_wait_gives_up_after_max_wait():
    sleeps = []
    r = gpu.wait_until_at_most(3072, query=_readings(5000), sleep=sleeps.append, step_s=300, max_wait_s=1200, log=lambda m: None)
    assert not r.ok and r.waited_s == 1200 and len(sleeps) == 4 and len(r.readings) == 5


def _waiter(ok, used=2800):
    calls = []

    def waiter(limit):
        calls.append(limit)
        return gpu.WaitResult(ok, used, limit, 0.0 if ok else 1200.0, (used,))

    return waiter, calls


def test_record_idle_writes_reading(tmp_path):
    idle = tmp_path / "gpu_idle_mib.txt"
    waiter, calls = _waiter(True, used=2767)
    assert gpu_check.main(["--record-idle"], idle_file=idle, waiter=waiter) == 0
    assert calls == [gpu.IDLE_LIMIT_MIB] and gpu.read_idle_mib(idle) == 2767


def test_record_idle_blocked(tmp_path, capsys):
    waiter, _ = _waiter(False, used=4858)
    assert gpu_check.main(["--record-idle"], idle_file=tmp_path / "i.txt", waiter=waiter) == 1
    assert "BLOCKED" in capsys.readouterr().err and not (tmp_path / "i.txt").exists()


def test_check_uses_idle_plus_margin(tmp_path, capsys):
    idle = tmp_path / "i.txt"
    idle.write_text("2767\n", encoding="utf-8")
    waiter, calls = _waiter(True)
    assert gpu_check.main([], idle_file=idle, waiter=waiter) == 0
    assert calls == [2767 + gpu.BUSY_MARGIN_MIB]
    waiter, _ = _waiter(False)
    assert gpu_check.main([], idle_file=idle, waiter=waiter) == 1
    assert "BLOCKED" in capsys.readouterr().err


def test_check_without_idle_file_fails(tmp_path):
    assert gpu_check.main([], idle_file=tmp_path / "missing.txt") == 1
