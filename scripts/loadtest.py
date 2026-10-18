"""Single-command Load_Test: python scripts/loadtest.py (logic in fts/loadtest.py; schedule in config/loadtest.yaml)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.loadtest import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
