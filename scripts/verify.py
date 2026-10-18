"""Entry point: milestone gate checks (logic in fts/verify.py). Usage: python scripts/verify.py m0|m1|..."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.verify import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
