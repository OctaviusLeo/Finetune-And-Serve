"""Feature: finetune-and-serve, Property 6: Error-analysis group counts sum to the analyzed total.

For any set of failures, the classifier assigns each failure to exactly one
documented group, the reported group counts sum exactly to the number of
analyzed failures, and the analyzed total is at least 50 when at least 50
failures exist.

Validates: Requirements 5.4, 5.5
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from fts.data import Example
from fts.error_analysis import GROUPS, MIN_ANALYZED, classify, error_analysis
from fts.metric import score_example

name = st.sampled_from(["alarm_set", "alarm_query", "play_music", "weather_query"])
slot_type = st.sampled_from(["time", "date", "place_name", "artist_name"])
value = st.sampled_from(["five am", "friday", "paris", "abba", "six pm"])
slots = st.lists(st.tuples(slot_type, value), max_size=4)


def enc(intent, ss):
    return json.dumps({"intent": intent, "slots": [{"type": a, "value": b} for a, b in ss]})


prediction = st.one_of(st.builds(enc, name, slots), st.text(max_size=20), st.just('{"intent": 1}'))


@settings(max_examples=100, derandomize=True, deadline=None)
@given(rows=st.lists(st.tuples(name, slots, prediction), min_size=0, max_size=120))
def test_group_counts_sum_to_analyzed_total(rows):
    """Feature: finetune-and-serve, Property 6: Error-analysis group counts sum to the analyzed total."""
    pairs = [(Example(str(i), "test", f"u{i}", enc(intent, ss)), pred) for i, (intent, ss, pred) in enumerate(rows)]
    failures = [(ex, p) for ex, p in pairs if score_example(p, ex.target) < 1.0]
    result = error_analysis(pairs)
    assert {g.group for g in result.groups} == set(GROUPS)
    assert sum(g.count for g in result.groups) == result.analyzed == len(failures)
    assert result.analyzed >= min(MIN_ANALYZED, len(failures))
    for ex, p in failures:
        assert classify(p, ex.target) in GROUPS
    assert all(len(v) <= 5 for v in result.examples.values())
