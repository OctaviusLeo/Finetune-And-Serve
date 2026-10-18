"""Task_Metric: field-level F1 for intent-and-slot JSON, range [0, 1] (Req 2.1).

For one example, the gold and predicted fields are multisets of pairs:
``("intent", intent)`` plus ``(slot_type, value)`` for every slot, with every
string lowercased, trimmed, and internal whitespace collapsed. Matches are
counted with multiplicity:

    precision = matches / predicted pairs
    recall    = matches / gold pairs
    F1        = 2PR / (P + R), and 0 when there are no matches.

A prediction is invalid, and scores 0, when no JSON object with a string
``intent`` and a list ``slots`` of ``{"type": str, "value": str}`` objects can be
parsed from the first ``{...}`` block of the output. The aggregate is the mean
per-example F1, also in [0, 1].

Hand-computed example: gold {alarm_set; time: five am; date: this week},
prediction {alarm_set; time: five am}: P = 2/2, R = 2/3, F1 = 0.8.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

METRIC_RANGE = (0.0, 1.0)
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Parsed:
    intent: str
    slots: tuple[tuple[str, str], ...]  # (type, value) in output order


def first_json_object(text: str) -> str | None:
    """The first balanced ``{...}`` block in ``text`` (braces inside strings ignored), or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # the first "{" never closes


def parse_prediction(text: str) -> Parsed | None:
    """Parse the first ``{...}`` block as {"intent": str, "slots": [{"type": str, "value": str}]}; None if invalid."""
    block = first_json_object(text)
    if block is None:
        return None
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("intent"), str) or not isinstance(obj.get("slots"), list):
        return None
    slots = []
    for slot in obj["slots"]:
        if not isinstance(slot, dict) or not isinstance(slot.get("type"), str) or not isinstance(slot.get("value"), str):
            return None
        slots.append((slot["type"], slot["value"]))
    return Parsed(intent=obj["intent"], slots=tuple(slots))


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def field_pairs(parsed: Parsed) -> Counter:
    pairs = Counter({("intent", normalize(parsed.intent)): 1})
    pairs.update((normalize(t), normalize(v)) for t, v in parsed.slots)
    return pairs


def f1_from_pairs(predicted: Counter, gold: Counter) -> float:
    matches = sum((predicted & gold).values())
    if matches == 0:
        return 0.0
    precision = matches / sum(predicted.values())
    recall = matches / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def score_example(prediction: str, gold: str) -> float:
    """Field-level F1 of ``prediction`` against the canonical ``gold`` JSON; invalid prediction -> 0.0."""
    gold_parsed = parse_prediction(gold)
    if gold_parsed is None:
        raise ValueError(f"gold target is not valid task JSON: {gold!r}")
    pred_parsed = parse_prediction(prediction)
    if pred_parsed is None:
        return 0.0
    return f1_from_pairs(field_pairs(pred_parsed), field_pairs(gold_parsed))


def aggregate(scores: Sequence[float]) -> float:
    """Mean per-example F1 over the scored examples, in [0, 1]."""
    if len(scores) == 0:
        raise ValueError("aggregate needs at least one score")
    return sum(scores) / len(scores)
