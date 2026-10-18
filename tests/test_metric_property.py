"""Feature: finetune-and-serve, Property 4: Task_Metric range and identity.

For any prediction/gold pair, score_example lies in [0.0, 1.0]; when the
prediction encodes the same intent and slot multiset as the gold the score is
1.0; aggregate of scores in [0, 1] lies in [0, 1].

Validates: Requirements 2.1, 3.2, 5.1
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from fts.metric import aggregate, score_example

name = st.from_regex(r"[a-z]{1,6}(_[a-z]{1,6})?", fullmatch=True)
value = st.from_regex(r"[a-z0-9]{1,6}( [a-z0-9]{1,6}){0,2}", fullmatch=True)
slot = st.tuples(name, value)


def encode(intent, slots):
    return json.dumps({"intent": intent, "slots": [{"type": t, "value": v} for t, v in slots]})


prediction_text = st.one_of(
    st.text(max_size=40),  # arbitrary (mostly invalid) output
    st.builds(encode, name, st.lists(slot, max_size=5)),  # valid JSON, unrelated content
)


@settings(max_examples=100, derandomize=True, deadline=None)
@given(
    intent=name,
    gold_slots=st.lists(slot, max_size=6),
    data=st.data(),
)
def test_metric_range_and_identity(intent, gold_slots, data):
    """Feature: finetune-and-serve, Property 4: Task_Metric range and identity."""
    gold = encode(intent, gold_slots)
    # Identity: the same intent and slot multiset, in any order, scores 1.0.
    permuted = data.draw(st.permutations(gold_slots))
    assert score_example(encode(intent, permuted), gold) == 1.0

    # Partial overlap: a sub-multiset of the gold plus extra slots stays in range.
    kept = data.draw(st.lists(st.sampled_from(gold_slots), max_size=len(gold_slots))) if gold_slots else []
    extra = data.draw(st.lists(slot, max_size=3))
    partial = score_example(encode(data.draw(st.sampled_from([intent, intent + "x"])), kept + extra), gold)
    assert 0.0 <= partial <= 1.0

    other = score_example(data.draw(prediction_text), gold)
    assert 0.0 <= other <= 1.0

    scores = [1.0, partial, other]
    assert 0.0 <= aggregate(scores) <= 1.0
