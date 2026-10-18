"""Few-shot exemplar selection from the Train_Split only (Req 3.5, 3.6).

Shuffle the Train_Split with ``random.Random(seed)`` and take the first example
of each of ``k`` distinct scenarios, in shuffled order. A MASSIVE intent is
``{scenario}_{intent}`` and scenario names contain no underscore, so the
scenario is the intent's prefix before the first ``_``. The result is stored in
config/few_shot_exemplars.json and used unchanged by both few-shot baselines.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from fts.data import Example

REPO_ROOT = Path(__file__).resolve().parent.parent
EXEMPLARS_JSON = REPO_ROOT / "config" / "few_shot_exemplars.json"
DEFAULT_K = 10
DEFAULT_SEED = 0


def scenario_of(example: Example) -> str:
    return json.loads(example.target)["intent"].split("_", 1)[0]


def select_exemplars(train: Sequence[Example], k: int = DEFAULT_K, seed: int = DEFAULT_SEED) -> list[Example]:
    """First example of each of ``k`` distinct scenarios after a seeded shuffle of the Train_Split."""
    order = list(train)
    random.Random(seed).shuffle(order)
    chosen: list[Example] = []
    seen: set[str] = set()
    for ex in order:
        scenario = scenario_of(ex)
        if scenario not in seen:
            seen.add(scenario)
            chosen.append(ex)
            if len(chosen) == k:
                break
    return chosen


def save_exemplars(exemplars: Sequence[Example], path: Path | str = EXEMPLARS_JSON, seed: int = DEFAULT_SEED) -> Path:
    payload = {
        "seed": seed,
        "method": "random.Random(seed).shuffle of the Train_Split; first example of each distinct scenario (intent prefix)",
        "exemplars": [
            {"id": e.example_id, "partition": e.partition, "scenario": scenario_of(e), "utt": e.inputs, "target": e.target}
            for e in exemplars
        ],
    }
    out = Path(path)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_exemplars(path: Path | str = EXEMPLARS_JSON) -> list[Example]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Example(e["id"], e["partition"], e["utt"], e["target"]) for e in data["exemplars"]]
