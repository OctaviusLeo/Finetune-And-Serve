"""Entry point: environment check and VRAM Dry_Run (logic in fts/dry_run.py).

Run from the repository root inside WSL: python scripts/dry_run.py [--per-device-batch N --grad-accum M]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.dry_run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
