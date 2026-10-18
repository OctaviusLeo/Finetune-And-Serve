"""Grouped error analysis of the serving seed's Test failures (Req 5.4, 5.5).

A failure is a Test example whose field-level F1 is below 1.0. Every failure is
analyzed and assigned to exactly one group, the first that applies in this order:

1. ``invalid_output``   - no valid JSON object of the required shape;
2. ``wrong_intent``     - the predicted intent differs from the gold intent;
3. ``missing_slot``     - for some slot type, fewer predicted slots than gold;
4. ``extra_slot``       - for some slot type, more predicted slots than gold;
5. ``wrong_slot_value`` - everything else: same intent, same slot types and counts,
   but at least one value differs.

The last group catches every remaining failure, so the group counts always sum
to the number analyzed. Five examples per group (first in dataset order) are kept
for the Model_Card. Strings are compared after the metric's normalization.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from fts.data import Example
from fts.infer import Prediction
from fts.metric import normalize, parse_prediction, score_example

GROUPS = ("invalid_output", "wrong_intent", "missing_slot", "extra_slot", "wrong_slot_value")
EXAMPLES_PER_GROUP = 5
MIN_ANALYZED = 50


@dataclass(frozen=True)
class FailureGroup:
    group: str
    count: int


@dataclass(frozen=True)
class ErrorAnalysis:
    analyzed: int  # every failure of the serving seed (>= 50 when available)
    groups: tuple[FailureGroup, ...]  # sum(counts) == analyzed
    examples: dict[str, list[dict]] = field(default_factory=dict)
    available_failures: int = 0

    def to_dict(self) -> dict:
        return {
            "analyzed": self.analyzed,
            "available_failures": self.available_failures,
            "groups": [{"group": g.group, "count": g.count} for g in self.groups],
            "taxonomy_order": list(GROUPS),
            "examples": self.examples,
        }


def classify(prediction: str, gold: str) -> str:
    """The failure group of one prediction whose F1 against ``gold`` is below 1."""
    pred = parse_prediction(prediction)
    if pred is None:
        return "invalid_output"
    ref = parse_prediction(gold)
    if ref is None:
        raise ValueError(f"gold is not valid task JSON: {gold!r}")
    if normalize(pred.intent) != normalize(ref.intent):
        return "wrong_intent"
    pred_types = Counter(normalize(t) for t, _ in pred.slots)
    gold_types = Counter(normalize(t) for t, _ in ref.slots)
    if any(pred_types[t] < n for t, n in gold_types.items()):
        return "missing_slot"
    if any(n > gold_types[t] for t, n in pred_types.items()):
        return "extra_slot"
    return "wrong_slot_value"


def error_analysis(pairs: Sequence[tuple[Example, str]], examples_per_group: int = EXAMPLES_PER_GROUP) -> ErrorAnalysis:
    """Analyze every failure (F1 < 1) among ``(example, prediction)`` pairs in dataset order."""
    counts = {g: 0 for g in GROUPS}
    kept: dict[str, list[dict]] = {g: [] for g in GROUPS}
    failures = 0
    for ex, pred in pairs:
        f1 = score_example(pred, ex.target)
        if f1 >= 1.0:
            continue
        failures += 1
        group = classify(pred, ex.target)
        counts[group] += 1
        if len(kept[group]) < examples_per_group:
            kept[group].append({"id": ex.example_id, "utterance": ex.inputs, "gold": ex.target, "prediction": pred, "f1": f1})
    groups = tuple(FailureGroup(g, counts[g]) for g in GROUPS)
    return ErrorAnalysis(analyzed=failures, groups=groups, examples=kept, available_failures=failures)


def pairs_from_predictions(test: Sequence[Example], predictions: Sequence[Prediction]) -> list[tuple[Example, str]]:
    """Test examples (with scorable gold) joined to their predictions, in dataset order."""
    by_id = {p.example_id: p.text for p in predictions}
    return [(ex, by_id[ex.example_id]) for ex in test if ex.example_id in by_id and parse_prediction(ex.target) is not None]
