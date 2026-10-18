"""Task_Metric: deterministic field-level F1 for JSON extraction (Req 2.1, 0.13).

The default Task_Metric is a field-level F1 over the key/value pairs of a JSON
object extracted by the model, compared against the gold JSON object. It is a
deterministic pure function in ``[0, 1]`` behind a clean, swappable boundary
(``score_example``) so the owner can replace the metric without touching callers.
``aggregate`` returns the arithmetic mean of per-example scores, also in
``[0, 1]``.

Range contract
--------------
``score_example(prediction, gold)`` returns a float in ``[0.0, 1.0]`` and is
``1.0`` exactly when ``prediction`` and ``gold`` parse to the same set of
field/value pairs. ``aggregate(scores)`` returns a float in ``[0.0, 1.0]`` (the
mean), or ``0.0`` for an empty input.
"""

from __future__ import annotations

import json
from collections.abc import Sequence


def _parse_fields(text: str) -> dict[str, str] | None:
    """Parse ``text`` into a flat ``{field: stringified value}`` mapping.

    Returns ``None`` when the text is not a JSON object. Values are normalized
    to their JSON string form so comparison is type- and whitespace-stable.
    """
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    return {str(k): json.dumps(v, sort_keys=True) for k, v in obj.items()}


def score_example(prediction: str, gold: str) -> float:
    """Field-level F1 between the predicted and gold JSON objects, in ``[0, 1]``.

    A predicted field is a true positive when the gold object has the same key
    with an equal (JSON-normalized) value. Precision is TP / predicted fields,
    recall is TP / gold fields, and the score is their harmonic mean. When
    either object fails to parse as a JSON object, the score is ``1.0`` only if
    both are byte-equal after stripping, else ``0.0``. Two empty JSON objects
    score ``1.0``.
    """
    pred_fields = _parse_fields(prediction)
    gold_fields = _parse_fields(gold)

    if pred_fields is None or gold_fields is None:
        return 1.0 if prediction.strip() == gold.strip() else 0.0

    if not pred_fields and not gold_fields:
        return 1.0

    true_positives = sum(
        1 for k, v in pred_fields.items() if k in gold_fields and gold_fields[k] == v
    )
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(pred_fields)
    recall = true_positives / len(gold_fields)
    return 2.0 * precision * recall / (precision + recall)


def aggregate(scores: Sequence[float]) -> float:
    """Return the mean of per-example ``scores`` in ``[0, 1]`` (``0.0`` if empty)."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
