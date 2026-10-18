"""Property-based tests for the data pipeline (tasks 2.2, 2.3).

Feature: finetune-and-serve, Property 1: Split disjointness + exact overlap reporting
Feature: finetune-and-serve, Property 2: Deterministic split membership under a fixed seed
Feature: finetune-and-serve, Property 3: Deterministic transform + deterministic malformed exclusion
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from fts.data import (
    Example,
    Splits,
    build_splits,
    check_overlap,
    transform_batch,
    transform_example,
)

_example_id = st.from_regex(r"id-[0-9a-f]{1,8}", fullmatch=True)
_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=20,
)


@st.composite
def _examples(draw):
    n = draw(st.integers(min_value=0, max_value=40))
    ids = draw(st.lists(_example_id, min_size=n, max_size=n, unique=True))
    return [Example(example_id=i, inputs=draw(_text), target=draw(_text)) for i in ids]


@given(examples=_examples(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_split_disjointness_and_overlap_reporting(examples, seed):
    """Feature: finetune-and-serve, Property 1: Split disjointness + exact overlap reporting."""
    splits = build_splits(examples, seed=seed)

    # Splits produced by build_splits are pairwise disjoint.
    assert check_overlap(splits) == {}

    train_ids = {e.example_id for e in splits.train}
    val_ids = {e.example_id for e in splits.validation}
    test_ids = {e.example_id for e in splits.test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Inject a known overlap and confirm it is reported exactly, with count.
    if splits.test:
        leaked = splits.test[0]
        bad = Splits(
            train=splits.train + (leaked,),
            validation=splits.validation,
            test=splits.test,
        )
        overlaps = check_overlap(bad)
        assert overlaps == {"train-test": [leaked.example_id]}
        assert len(overlaps["train-test"]) == 1


@given(examples=_examples(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_deterministic_split_membership(examples, seed):
    """Feature: finetune-and-serve, Property 2: Deterministic split membership under a fixed seed."""
    first = build_splits(examples, seed=seed)
    # Re-run with the SAME seed but shuffled input order: membership is identical.
    reordered = list(reversed(examples))
    second = build_splits(reordered, seed=seed)

    def ids(sp):
        return (
            {e.example_id for e in sp.train},
            {e.example_id for e in sp.validation},
            {e.example_id for e in sp.test},
        )

    assert ids(first) == ids(second)


@given(
    example_id=_example_id,
    inputs=_text,
    target=_text,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_deterministic_transform_and_malformed_exclusion(example_id, inputs, target, seed):
    """Feature: finetune-and-serve, Property 3: Deterministic transform + deterministic malformed exclusion."""
    raw = {"id": example_id, "inputs": inputs, "target": target}
    first = transform_example(raw, seed)
    second = transform_example(raw, seed)
    assert first == second
    assert first is not None

    # Malformed records (missing fields / wrong type) are excluded deterministically.
    malformed = [
        {"id": example_id, "inputs": inputs},  # missing target
        {"inputs": inputs, "target": target},  # missing id
        {"id": example_id, "inputs": "", "target": target},  # empty inputs
        "not-a-dict",
    ]
    batch = [raw] + malformed
    report_a = transform_batch(batch, seed)
    report_b = transform_batch(batch, seed)
    assert report_a.excluded_count == report_b.excluded_count == len(malformed)
    assert [e.example_id for e in report_a.examples] == [
        e.example_id for e in report_b.examples
    ]
    assert [x.example_id for x in report_a.exclusions] == [
        x.example_id for x in report_b.exclusions
    ]
