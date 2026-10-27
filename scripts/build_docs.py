"""Entry point: generate README.md, MODEL_CARD.md and results/doc_facts.json, or check them with --check (logic in fts/docs.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts.docs import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
