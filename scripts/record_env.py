"""Entry point: record library versions in docs/ENVIRONMENT.md and licenses in THIRD_PARTY.md (logic in fts/envrecord.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.envrecord import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
