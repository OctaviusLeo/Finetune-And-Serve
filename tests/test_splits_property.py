"""Property tests for the splitter (tasks 2.2 and 2.3).

Feature: finetune-and-serve, Property 1: Split disjointness and overlap reporting.
Validates: Requirements 2.3, 2.4
Feature: finetune-and-serve, Property 2: Deterministic split membership under a fixed seed.
Validates: Requirements 2.3, 2.5
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from fts.data import Example, Splits, build_splits, check_overlap, overlap_total

PARTITIONS = ("train", "dev", "test")


@st.composite
def examples(draw, min_size=0):
    ids = draw(st.lists(st.from_regex(r"[0-9]{1,6}", fullmatch=True), min_size=min_size, max_size=40, unique=True))
    return [
        Example(example_id=i, partition=draw(st.sampled_from(PARTITIONS)), inputs=f"utt {i}", target='{"intent": "x", "slots": []}')
        for i in ids
    ]


@settings(max_examples=100, derandomize=True, deadline=None)
@given(exs=examples(min_size=1), data=st.data())
def test_split_disjointness_and_overlap_reporting(exs, data):
    """Feature: finetune-and-serve, Property 1: Split disjointness and overlap reporting."""
    splits = build_splits(exs, seed=0)
    ids = [{e.example_id for e in part} for part in (splits.train, splits.validation, splits.test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert check_overlap(splits) == {}
    assert sum(len(i) for i in ids) == len(exs)

    # Inject known examples into a second split and check the report is exact.
    injected = data.draw(st.lists(st.sampled_from(exs), min_size=1, max_size=len(exs), unique_by=lambda e: e.example_id))
    names = ("train", "validation", "test")
    parts = {n: list(splits.by_name(n)) for n in names}
    expected: dict[str, set[str]] = {}
    for e in injected:
        home = {"train": "train", "dev": "validation", "test": "test"}[e.partition]
        other = data.draw(st.sampled_from([n for n in names if n != home]))
        parts[other].append(e)
        pair = "-".join(sorted((home, other), key=names.index))
        expected.setdefault(pair, set()).add(e.example_id)
    overlaps = check_overlap(Splits(*(tuple(parts[n]) for n in names)))
    assert {k: set(v) for k, v in overlaps.items()} == expected
    assert all(v == sorted(v) for v in overlaps.values())
    assert overlap_total(overlaps) == len(injected)


@settings(max_examples=100, derandomize=True, deadline=None)
@given(exs=examples(), seed=st.integers(min_value=0, max_value=2**31 - 1), other=st.integers(min_value=0, max_value=2**31 - 1))
def test_deterministic_split_membership(exs, seed, other):
    """Feature: finetune-and-serve, Property 2: Deterministic split membership under a fixed seed."""
    first, second, different = build_splits(exs, seed), build_splits(exs, seed), build_splits(exs, other)
    assert first == second == different
    official = {"train": "train", "dev": "validation", "test": "test"}
    for name in ("train", "validation", "test"):
        assert [e.example_id for e in first.by_name(name)] == [e.example_id for e in exs if official[e.partition] == name]
