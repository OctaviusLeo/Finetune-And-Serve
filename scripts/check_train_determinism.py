"""Entry point: train the Final_Configuration twice for 30 steps and compare the losses (logic in fts/determinism.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.determinism import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
