"""Unit tests for the error-analysis classifier and harness (Req 5.4, 5.5)."""

from __future__ import annotations

import json

import pytest

from fts import error_analysis as ea
from fts.data import Example
from fts.infer import Prediction


def t(intent, *slots):
    return json.dumps({"intent": intent, "slots": [{"type": a, "value": b} for a, b in slots]})


GOLD = t("alarm_set", ("time", "five am"), ("date", "friday"))


@pytest.mark.parametrize(
    ("prediction", "group"),
    [
        ("not json", "invalid_output"),
        ('{"intent": "alarm_set"}', "invalid_output"),
        (t("alarm_query", ("time", "five am"), ("date", "friday")), "wrong_intent"),
        (t("alarm_query"), "wrong_intent"),  # intent is checked before slots
        (t("alarm_set", ("time", "five am")), "missing_slot"),
        (t("alarm_set", ("time", "five am"), ("time", "six am")), "missing_slot"),  # date missing comes first
        (t("alarm_set", ("time", "five am"), ("date", "friday"), ("place_name", "home")), "extra_slot"),
        (t("alarm_set", ("time", "five pm"), ("date", "friday")), "wrong_slot_value"),
        (t("ALARM_SET", ("Time", "six am"), ("date", "friday")), "wrong_slot_value"),
    ],
)
def test_classify_first_matching_group(prediction, group):
    assert ea.classify(prediction, GOLD) == group


def test_classify_rejects_invalid_gold():
    with pytest.raises(ValueError):
        ea.classify(t("x"), "bad gold")


def test_error_analysis_counts_examples_and_sum():
    pairs = [
        (Example("1", "test", "u1", GOLD), GOLD),  # correct: not a failure
        (Example("2", "test", "u2", GOLD), "oops"),
        (Example("3", "test", "u3", GOLD), t("alarm_query")),
        (Example("4", "test", "u4", GOLD), t("alarm_set", ("time", "five am"))),
        (Example("5", "test", "u5", GOLD), t("alarm_set", ("time", "five am"), ("date", "monday"))),
    ] + [(Example(str(10 + i), "test", f"x{i}", GOLD), "nope") for i in range(7)]
    result = ea.error_analysis(pairs, examples_per_group=5)
    counts = {g.group: g.count for g in result.groups}
    assert counts == {"invalid_output": 8, "wrong_intent": 1, "missing_slot": 1, "extra_slot": 0, "wrong_slot_value": 1}
    assert result.analyzed == sum(counts.values()) == 11 == result.available_failures
    assert len(result.examples["invalid_output"]) == 5 and result.examples["extra_slot"] == []
    first = result.examples["wrong_slot_value"][0]
    assert first["id"] == "5" and first["utterance"] == "u5" and 0 < first["f1"] < 1
    d = result.to_dict()
    assert d["taxonomy_order"] == list(ea.GROUPS) and d["groups"][0] == {"group": "invalid_output", "count": 8}


def test_pairs_from_predictions_skips_unscorable_gold_and_missing_predictions():
    test = [Example("1", "test", "a", GOLD), Example("2", "test", "b", "bad"), Example("3", "test", "c", GOLD)]
    preds = [Prediction("1", "x", 1, 1), Prediction("2", "y", 1, 1)]
    assert [(e.example_id, p) for e, p in ea.pairs_from_predictions(test, preds)] == [("1", "x")]
