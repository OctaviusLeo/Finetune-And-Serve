"""Feature: finetune-and-serve, Property 3: Deterministic transform and malformed exclusion.

For any raw record and any seed, transform_example applied twice with the same
seed produces the identical Example, and a malformed record (unbalanced
brackets, missing " : ", empty intent, or markup that does not reproduce utt)
is excluded (returns None) deterministically.

Validates: Requirements 2.5, 2.6
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from fts.data import transform_example

word = st.from_regex(r"[a-z']{1,8}", fullmatch=True)
label = st.from_regex(r"[a-z]{1,8}(_[a-z]{1,8})?", fullmatch=True)
value = st.lists(word, min_size=1, max_size=3).map(" ".join)


@st.composite
def well_formed(draw):
    """A record built from random plain words and slot spans, with its expected slots."""
    pieces = draw(st.lists(st.one_of(word.map(lambda w: ("w", w)), st.tuples(label, value).map(lambda s: ("s", s))), min_size=1, max_size=8))
    utt_parts, annot_parts, slots = [], [], []
    for kind, item in pieces:
        if kind == "w":
            utt_parts.append(item)
            annot_parts.append(item)
        else:
            lab, val = item
            utt_parts.append(val)
            annot_parts.append(f"[{lab} : {val}]")
            slots.append({"type": lab, "value": val})
    record = {
        "id": str(draw(st.integers(min_value=0, max_value=10**6))),
        "partition": draw(st.sampled_from(["train", "dev", "test"])),
        "intent": draw(label),
        "utt": " ".join(utt_parts),
        "annot_utt": " ".join(annot_parts),
    }
    return record, slots


def corrupt(record: dict, how: str) -> dict:
    bad = dict(record)
    if how == "unbalanced":
        bad["annot_utt"] = "[" + bad["annot_utt"]
    elif how == "no_colon":
        bad["annot_utt"] = bad["annot_utt"] + " [loose]"
        bad["utt"] = bad["utt"] + " loose"
    elif how == "empty_intent":
        bad["intent"] = " "
    elif how == "utt_mismatch":
        bad["utt"] = bad["utt"] + " extra"
    return bad


@settings(max_examples=100, derandomize=True, deadline=None)
@given(rec=well_formed(), seed=st.integers(min_value=0, max_value=2**31 - 1), how=st.sampled_from(["unbalanced", "no_colon", "empty_intent", "utt_mismatch"]))
def test_deterministic_transform_and_malformed_exclusion(rec, seed, how):
    """Feature: finetune-and-serve, Property 3: Deterministic transform and malformed exclusion."""
    record, slots = rec
    first, second = transform_example(record, seed), transform_example(record, seed)
    assert first is not None and first == second
    assert json.loads(first.target) == {"intent": record["intent"], "slots": slots}
    assert first.target == json.dumps({"intent": record["intent"], "slots": slots}, ensure_ascii=False)

    bad = corrupt(record, how)
    assert transform_example(bad, seed) is None
    assert transform_example(bad, seed) is None
