"""Single-command load-test entry point (Req 6.6; task 5.4).

A thin wrapper so the load test is invocable as ``python scripts/loadtest.py``
in addition to ``python -m fts.loadtest``. Runs non-interactively; all driving
logic lives in :mod:`fts.loadtest` (GPU-free; the live sender uses httpx only).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the subproject root importable when run as `python scripts/loadtest.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.loadtest import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
