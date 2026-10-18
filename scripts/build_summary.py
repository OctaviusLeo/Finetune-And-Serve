"""Entry point: generate results/summary.md and results/plots/ from results/*.json (logic in fts/report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
