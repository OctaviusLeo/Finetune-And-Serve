"""Unit tests: hand-computed F1 (Req 9.5). Task 2.6 rewrites these for the MASSIVE metric."""

from __future__ import annotations

from fts.metric import score_example


def test_hand_computed_field_f1_within_1e_6():
    # prediction has 3 fields {a:1, b:2, c:3}; gold has 3 fields {a:1, b:9, d:4}.
    # True positives (key present with equal value): only "a" -> TP = 1.
    # precision = 1/3, recall = 1/3, F1 = 2*p*r/(p+r) = 1/3.
    prediction = '{"a": 1, "b": 2, "c": 3}'
    gold = '{"a": 1, "b": 9, "d": 4}'
    expected = 1.0 / 3.0
    assert abs(score_example(prediction, gold) - expected) < 1e-6


def test_perfect_match_scores_one():
    obj = '{"name": "x", "qty": 2}'
    assert score_example(obj, obj) == 1.0
