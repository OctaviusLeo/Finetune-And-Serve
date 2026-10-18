"""Feature: finetune-and-serve, Property 8: Few-shot exemplars never leak Test/Validation.

For any Splits and selected few-shot exemplar set, every exemplar id is a member
of the Train_Split and none is a member of the Validation_Split or Test_Split.

Validates: Requirements 3.5, 4.7
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from fts.data import Example, Splits
from fts.exemplars import scenario_of, select_exemplars

intents = st.sampled_from(["alarm_set", "alarm_query", "iot_cleaning", "play_music", "qa_factoid", "weather_query",
                           "news_query", "email_sendemail", "lists_query", "takeaway_order", "transport_ticket", "general_joke"])


@st.composite
def splits_strategy(draw):
    n = draw(st.integers(min_value=0, max_value=60))
    ids = draw(st.lists(st.integers(min_value=0, max_value=10**6), min_size=n, max_size=n, unique=True))
    utterances = draw(st.lists(st.sampled_from(["a", "b", "c", "set an alarm"]), min_size=n, max_size=n))  # texts may repeat across splits
    parts = draw(st.lists(st.sampled_from(["train", "dev", "test"]), min_size=n, max_size=n))
    exs = [Example(str(i), p, u, json.dumps({"intent": draw(intents), "slots": []})) for i, u, p in zip(ids, utterances, parts)]
    return Splits(
        train=tuple(e for e in exs if e.partition == "train"),
        validation=tuple(e for e in exs if e.partition == "dev"),
        test=tuple(e for e in exs if e.partition == "test"),
    )


@settings(max_examples=100, derandomize=True, deadline=None)
@given(splits=splits_strategy(), k=st.integers(min_value=0, max_value=12), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_exemplars_never_leak_validation_or_test(splits, k, seed):
    """Feature: finetune-and-serve, Property 8: Few-shot exemplars never leak Test/Validation."""
    chosen = select_exemplars(splits.train, k=k, seed=seed)
    ids = {e.example_id for e in chosen}
    assert ids <= {e.example_id for e in splits.train}
    assert ids.isdisjoint({e.example_id for e in splits.validation})
    assert ids.isdisjoint({e.example_id for e in splits.test})
    assert len(chosen) <= k and len({scenario_of(e) for e in chosen}) == len(chosen)
