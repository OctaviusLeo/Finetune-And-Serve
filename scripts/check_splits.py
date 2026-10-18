"""M1 gate step: rebuild the splits under runs/rebuild/ and confirm they match data/massive_en (logic in fts/massive.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.massive import rebuild_and_compare  # noqa: E402


def main() -> int:
    failures = rebuild_and_compare()
    for f in failures:
        print(f"FAIL: {f}")
    print("splits: rebuilt membership and targets identical, no overlap" if not failures else f"splits: {len(failures)} problem(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
