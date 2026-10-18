"""Unit tests for fts.jev_extract (candidate spans, questions, decoding)."""

from __future__ import annotations

import json

from fts import jev_extract
from fts.data import Example

LABELS = {"intents": ["alarm_set", "weather_query"], "slot_types": ["date", "place_name", "time"]}
T = jev_extract.load_templates()


def answer(choice, probs):
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": max(probs.values())}


def test_templates_load_from_repository():
    assert "`request`" in T.intent and "{span}" in T.span and "not a complete slot value" in T.none


def test_candidate_spans_cover_contiguous_runs_up_to_the_limit():
    spans = jev_extract.candidate_spans("tell me the weather", 2)
    assert [s.text for s in spans] == ["tell", "tell me", "me", "me the", "the", "the weather", "weather"]
    assert spans[1] == jev_extract.Span(0, 2, "tell me") and spans[1].qid == "span_0_2"
    assert len(jev_extract.candidate_spans("a b c d e f g h", 6)) == 8 * 6 - 15
    assert jev_extract.candidate_spans("", 6) == []


def test_state_and_questions():
    shot = Example("1", "train", "wake me at five", json.dumps({"intent": "alarm_set", "slots": [{"type": "time", "value": "five"}]}))
    state = jev_extract.build_state("weather in paris", [shot])
    assert state == {"request": "weather in paris", "examples": [{"request": "wake me at five", "answer": {"intent": "alarm_set", "slots": [{"type": "time", "value": "five"}]}}]}
    q = jev_extract.intent_question(T, LABELS)
    assert q["type"] == "choice" and list(q["criteria"]) == LABELS["intents"]
    sq = jev_extract.span_question(T, LABELS, jev_extract.Span(2, 3, "paris"))
    assert 'Span: "paris"' in sq["instructions"] and list(sq["criteria"]) == [*LABELS["slot_types"], "none"] and sq["criteria"]["none"] == T.none


def test_batches_split_and_mark_repeated_text():
    spans, batches = jev_extract.question_batches("play play jazz", LABELS, T, 2, per_request=3)
    assert len(spans) == 5 and [len(b) for b in batches] == [4, 2]
    assert "intent" in batches[0] and "intent" not in batches[1]
    play = [q for b in batches for qid, q in b.items() if qid in ("span_0_1", "span_1_2")]
    assert "(occurrence 1 of 2" in play[0]["instructions"] and "(occurrence 2 of 2" in play[1]["instructions"]
    _, empty = jev_extract.question_batches("", LABELS, T, 2, per_request=3)
    assert empty == [{"intent": jev_extract.intent_question(T, LABELS)}]


def test_decode_keeps_most_probable_non_overlapping_spans_in_order():
    spans = jev_extract.candidate_spans("weather in paris in two days", 3)
    answers = {"intent": answer("weather_query", {"weather_query": 0.99, "alarm_set": 0.01})}
    for s in spans:
        answers[s.qid] = answer("none", {"none": 0.9, "date": 0.05, "place_name": 0.05, "time": 0.0})
    answers["span_2_3"] = answer("place_name", {"place_name": 0.94, "none": 0.06, "date": 0.0, "time": 0.0})  # "paris"
    answers["span_1_3"] = answer("place_name", {"place_name": 0.55, "none": 0.45, "date": 0.0, "time": 0.0})  # "in paris" overlaps
    answers["span_4_6"] = answer("date", {"date": 0.93, "none": 0.07, "place_name": 0.0, "time": 0.0})  # "two days"
    answers["span_5_6"] = answer("date", {"date": 0.93, "none": 0.07, "place_name": 0.0, "time": 0.0})  # "days": tie -> longer wins
    out = json.loads(jev_extract.decode(answers, spans))
    assert out == {"intent": "weather_query", "slots": [{"type": "place_name", "value": "paris"}, {"type": "date", "value": "two days"}]}


def test_decode_without_slots_and_missing_answers():
    spans = jev_extract.candidate_spans("hello there", 2)
    answers = {"intent": answer("alarm_set", {"alarm_set": 1.0, "weather_query": 0.0})}
    assert jev_extract.decode(answers, spans) == '{"intent": "alarm_set", "slots": []}'
