"""Unit tests for the Task_Metric (task 2.6; Req 2.1, 9.1, 9.5)."""

from __future__ import annotations

import json

import pytest

from fts import metric


def target(intent, *slots):
    return json.dumps({"intent": intent, "slots": [{"type": t, "value": v} for t, v in slots]})


GOLD = target("alarm_set", ("time", "five am"), ("date", "this week"))


def test_hand_computed_example_is_0_8():
    # gold {alarm_set; time: five am; date: this week}, prediction {alarm_set; time: five am}
    # P = 2/2, R = 2/3, F1 = 2 * 1 * (2/3) / (1 + 2/3) = 0.8
    prediction = target("alarm_set", ("time", "five am"))
    assert abs(metric.score_example(prediction, GOLD) - 0.8) <= 1e-6


def test_exact_match_and_permuted_slots_score_one():
    assert metric.score_example(GOLD, GOLD) == 1.0
    permuted = target("alarm_set", ("date", "this week"), ("time", "five am"))
    assert metric.score_example(permuted, GOLD) == 1.0


def test_values_are_normalised():
    messy = json.dumps({"intent": " Alarm_Set ", "slots": [{"type": "TIME", "value": "  Five   AM"}, {"type": "date", "value": "this week"}]})
    assert metric.score_example(messy, GOLD) == 1.0


def test_invalid_outputs_score_zero():
    for bad in ["", "no json here", "{not json}", '{"intent": 3, "slots": []}', '{"intent": "x"}',
                '{"intent": "x", "slots": "none"}', '{"intent": "x", "slots": [{"type": "t"}]}',
                '{"intent": "x", "slots": ["a"]}', "[1, 2]", '{"intent": "alarm_set", "slots": [']:
        assert metric.score_example(bad, GOLD) == 0.0, bad


def test_duplicate_slots_count_with_multiplicity():
    gold = target("play_music", ("artist_name", "abba"), ("artist_name", "abba"))
    one = target("play_music", ("artist_name", "abba"))
    # matches = 2 of predicted 2, gold 3 -> P = 1, R = 2/3, F1 = 0.8
    assert metric.score_example(one, gold) == pytest.approx(0.8)
    three = target("play_music", ("artist_name", "abba"), ("artist_name", "abba"), ("artist_name", "abba"))
    # matches = 3, predicted 4, gold 3 -> P = 0.75, R = 1, F1 = 6/7
    assert metric.score_example(three, gold) == pytest.approx(6 / 7)


def test_wrong_intent_partial_credit_from_slots():
    pred = target("alarm_query", ("time", "five am"), ("date", "this week"))
    # matches 2 of 3 each way -> F1 = 2/3
    assert metric.score_example(pred, GOLD) == pytest.approx(2 / 3)


def test_no_matches_scores_zero():
    assert metric.score_example(target("weather_query", ("place_name", "paris")), GOLD) == 0.0


def test_first_json_block_is_used():
    text = 'Sure! {"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}, {"type": "date", "value": "this week"}]} and {"intent": "x", "slots": []}'
    assert metric.score_example(text, GOLD) == 1.0
    assert metric.first_json_object('pre {"a": "}{", "b": {"c": 1}} post') == '{"a": "}{", "b": {"c": 1}}'
    assert metric.first_json_object('{"a": "esc \\" brace }"}') == '{"a": "esc \\" brace }"}'
    assert metric.first_json_object("nothing") is None
    assert metric.first_json_object("{ never closed") is None


def test_parse_prediction_shape():
    assert metric.parse_prediction(GOLD) == metric.Parsed("alarm_set", (("time", "five am"), ("date", "this week")))
    assert metric.parse_prediction('{"intent": "x", "slots": [], "extra": 1}') == metric.Parsed("x", ())


def test_invalid_gold_raises():
    with pytest.raises(ValueError):
        metric.score_example(GOLD, "not gold")


def test_aggregate_mean_and_range():
    assert metric.aggregate([1.0, 0.8, 0.0]) == pytest.approx(0.6)
    assert metric.METRIC_RANGE == (0.0, 1.0)
    with pytest.raises(ValueError):
        metric.aggregate([])
