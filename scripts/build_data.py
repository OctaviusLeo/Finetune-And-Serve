"""Entry point: build data/massive_en/ and results/data_counts.json from MASSIVE 1.1 (logic in fts/massive.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.massive import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
