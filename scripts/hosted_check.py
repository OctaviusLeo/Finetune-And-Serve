"""Entry point: hosted API proof, one request to the Hosted_Model through ModelClient and the ledger (logic in fts/hosted.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.hosted import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
