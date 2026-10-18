"""Property-based test for the Task_Metric (task 2.5).

Feature: finetune-and-serve, Property 4: Task_Metric range + identity
"""

from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from fts.metric import aggregate, score_example

_key = st.from_regex(r"[a-e]", fullmatch=True)
_value = st.one_of(
    st.integers(min_value=-5, max_value=5),
    st.text(alphabet="xyz", min_size=0, max_size=3),
    st.booleans(),
)
_json_obj = st.dictionaries(_key, _value, min_size=0, max_size=5).map(
    lambda d: json.dumps(d, sort_keys=True)
)


@given(prediction=_json_obj, gold=_json_obj)
def test_metric_range_and_identity(prediction, gold):
    """Feature: finetune-and-serve, Property 4: Task_Metric range + identity."""
    score = score_example(prediction, gold)
    assert 0.0 <= score <= 1.0

    # Identity: prediction == gold scores exactly 1.0.
    assert score_example(gold, gold) == 1.0

    # aggregate stays in [0, 1].
    agg = aggregate([score, score_example(gold, gold)])
    assert 0.0 <= agg <= 1.0


@given(scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=0, max_size=20))
def test_aggregate_in_unit_interval(scores):
    """Aggregate always lands in the unit interval (plain unit assertion)."""
    assert 0.0 <= aggregate(scores) <= 1.0
