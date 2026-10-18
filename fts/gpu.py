"""nvidia-smi queries and a once-per-second GPU memory sampler.

``nvidia-smi`` reports memory for the whole GPU, including other programs (the
Windows desktop holds some), so its peak is an upper bound on what this
project used. Every call goes through :func:`run_nvidia_smi`, which tests fake.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

QUERY_ARGS = ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]


@dataclass(frozen=True)
class GpuMemory:
    used_mib: int
    total_mib: int


def run_nvidia_smi(args: list[str] | None = None, timeout_s: float = 30.0) -> str:
    """Run nvidia-smi and return its stdout; raises RuntimeError when it is missing or fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", *(args or [])], capture_output=True, text=True, check=True, timeout=timeout_s
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}") from exc
    return out.stdout


def parse_memory_csv(text: str) -> GpuMemory:
    """Parse the first GPU's ``used, total`` line of the CSV query output."""
    for line in text.splitlines():
        m = re.fullmatch(r"\s*(\d+)\s*(?:MiB)?\s*,\s*(\d+)\s*(?:MiB)?\s*", line)
        if m:
            return GpuMemory(used_mib=int(m.group(1)), total_mib=int(m.group(2)))
    raise ValueError(f"cannot parse nvidia-smi memory output: {text!r}")


def parse_total_mib(text: str) -> int | None:
    """Total MiB from the human-readable ``nvidia-smi`` table (``used MiB / total MiB``)."""
    m = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", text)
    return int(m.group(2)) if m else None


def query_memory(runner: Callable[[list[str]], str] = run_nvidia_smi) -> GpuMemory:
    return parse_memory_csv(runner(QUERY_ARGS))


def within_budget(peak_mib: float, total_mib: float) -> bool:
    """True when the peak does not exceed the total; no budget (total <= 0) never fits."""
    return total_mib > 0 and peak_mib <= total_mib


class MemorySampler:
    """Samples GPU memory every ``interval_s`` seconds on a daemon thread.

    Use as a context manager; ``samples`` holds every reading and ``peak_mib`` the largest.
    A failed reading is counted in ``errors`` and skipped.
    """

    def __init__(
        self,
        interval_s: float = 1.0,
        query: Callable[[], GpuMemory] = query_memory,
        on_sample: Callable[[float, GpuMemory], None] | None = None,
    ) -> None:
        self.interval_s = interval_s
        self._query = query
        self._on_sample = on_sample
        self.samples: list[GpuMemory] = []
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        try:
            reading = self._query()
        except Exception:  # noqa: BLE001 - a missed sample must not stop the job being measured
            self.errors += 1
            return
        self.samples.append(reading)
        if self._on_sample is not None:
            self._on_sample(time.time(), reading)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> MemorySampler:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10 * self.interval_s + 5)
        self.sample_once()  # one final reading after the job

    @property
    def peak_mib(self) -> int:
        return max((s.used_mib for s in self.samples), default=0)

    @property
    def total_mib(self) -> int:
        return max((s.total_mib for s in self.samples), default=0)
