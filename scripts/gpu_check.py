"""GPU-free check run before every GPU job (design.md "One GPU job at a time").

  python scripts/gpu_check.py --record-idle   # task 1.6: wait until usage <= 3072 MiB, write runs/gpu_idle_mib.txt
  python scripts/gpu_check.py                 # before a GPU job: wait until usage <= idle + 1024 MiB

Both forms wait in 5-minute steps for up to 20 minutes and exit 1 if the GPU is
still busy (the caller records a blocker). Nothing is stopped to free the GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts import gpu  # noqa: E402

IDLE_FILE = Path(__file__).resolve().parent.parent / "runs" / "gpu_idle_mib.txt"


def main(argv: list[str] | None = None, idle_file: Path = IDLE_FILE, waiter=gpu.wait_until_at_most) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-idle", action="store_true", help="record the idle reading (task 1.6)")
    args = parser.parse_args(argv)

    if args.record_idle:
        result = waiter(gpu.IDLE_LIMIT_MIB)
        if not result.ok:
            print(f"BLOCKED: GPU memory used {result.used_mib} MiB stayed above {result.limit_mib} MiB for "
                  f"{result.waited_s:.0f} s (readings {list(result.readings)})", file=sys.stderr)
            return 1
        idle_file.parent.mkdir(parents=True, exist_ok=True)
        idle_file.write_text(f"{result.used_mib}\n", encoding="utf-8")
        print(f"idle GPU memory: {result.used_mib} MiB (written to {idle_file})")
        return 0

    if not idle_file.exists():
        print(f"no idle reading at {idle_file}; run with --record-idle first", file=sys.stderr)
        return 1
    idle = gpu.read_idle_mib(idle_file)
    result = waiter(idle + gpu.BUSY_MARGIN_MIB)
    if not result.ok:
        print(f"BLOCKED: GPU memory used {result.used_mib} MiB stayed above idle {idle} + "
              f"{gpu.BUSY_MARGIN_MIB} MiB for {result.waited_s:.0f} s", file=sys.stderr)
        return 1
    print(f"GPU free: {result.used_mib} MiB used (idle {idle} MiB, limit {result.limit_mib} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
