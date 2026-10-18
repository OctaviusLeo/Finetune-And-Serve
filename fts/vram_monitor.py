"""GPU memory monitor for serving (Req 6.4).

Samples ``nvidia-smi --query-gpu=memory.used,memory.total`` once per second to a
CSV file (``unix_time,used_mib,total_mib``) while the Serving_Endpoint is up and
checks that the peak never exceeds the total. ``nvidia-smi`` reports the whole
GPU, so the peak is an upper bound on what the server used.

    with VramMonitor("runs/vram/serve_q4.csv") as mon:
        ...                      # quality evaluation or load test
    mon.summary()                # {"peak_mib", "total_mib", "within_budget", "samples", ...}

    python -m fts.vram_monitor --out runs/vram/manual.csv --seconds 60
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from fts import gpu


class VramBudgetExceeded(RuntimeError):
    def __init__(self, peak_mib: int, total_mib: int) -> None:
        self.peak_mib, self.total_mib = peak_mib, total_mib
        super().__init__(f"peak GPU memory {peak_mib} MiB exceeds the nvidia-smi total {total_mib} MiB")


def assert_within_budget(peak_mib: int, total_mib: int) -> None:
    if not gpu.within_budget(peak_mib, total_mib):
        raise VramBudgetExceeded(peak_mib, total_mib)


class VramMonitor:
    def __init__(self, out_path: Path | str, interval_s: float = 1.0, query: Callable[[], gpu.GpuMemory] = gpu.query_memory) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self.sampler = gpu.MemorySampler(interval_s=interval_s, query=query, on_sample=self._write)

    def _write(self, t: float, m: gpu.GpuMemory) -> None:
        if self._fh is not None:
            self._fh.write(f"{t:.3f},{m.used_mib},{m.total_mib}\n")
            self._fh.flush()

    def __enter__(self) -> VramMonitor:
        self._fh = self.out_path.open("w", encoding="utf-8")
        self._fh.write("unix_time,used_mib,total_mib\n")
        self.sampler.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self.sampler.__exit__(*exc)
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def peak_mib(self) -> int:
        return self.sampler.peak_mib

    @property
    def total_mib(self) -> int:
        return self.sampler.total_mib

    def summary(self) -> dict:
        return {
            "peak_mib": self.peak_mib,
            "total_mib": self.total_mib,
            "within_budget": gpu.within_budget(self.peak_mib, self.total_mib),
            "samples": len(self.sampler.samples),
            "failed_samples": self.sampler.errors,
            "csv": self.out_path.as_posix(),
        }

    def check(self) -> None:
        assert_within_budget(self.peak_mib, self.total_mib)


def main(argv: list[str] | None = None, query: Callable[[], gpu.GpuMemory] = gpu.query_memory, sleep: Callable[[float], None] | None = None) -> int:
    import time

    parser = argparse.ArgumentParser(prog="python -m fts.vram_monitor", description="Sample GPU memory once per second.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    with VramMonitor(args.out, query=query) as mon:
        (sleep or time.sleep)(args.seconds)
    s = mon.summary()
    print(f"peak {s['peak_mib']} MiB of {s['total_mib']} MiB over {s['samples']} samples -> {'within' if s['within_budget'] else 'OVER'} budget")
    return 0 if s["within_budget"] else 1


if __name__ == "__main__":
    sys.exit(main())
