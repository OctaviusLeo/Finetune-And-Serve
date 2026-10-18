"""Select the 10 few-shot exemplars from the Train_Split (seed 0) into config/few_shot_exemplars.json.

Usage: python scripts/select_exemplars.py [--k 10] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts import data, exemplars  # noqa: E402


def main(argv: list[str] | None = None, data_dir: Path = exemplars.REPO_ROOT / "data" / "massive_en", out: Path = exemplars.EXEMPLARS_JSON) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=exemplars.DEFAULT_K)
    parser.add_argument("--seed", type=int, default=exemplars.DEFAULT_SEED)
    args = parser.parse_args(argv)
    train = data.read_split(data_dir / "train.jsonl")
    chosen = exemplars.select_exemplars(train, args.k, args.seed)
    exemplars.save_exemplars(chosen, out, args.seed)
    for e in chosen:
        print(f"{e.example_id}\t{exemplars.scenario_of(e)}\t{e.inputs}")
    print(f"wrote {len(chosen)} exemplars to {out}")
    return 0 if len(chosen) == args.k else 1


if __name__ == "__main__":
    sys.exit(main())
