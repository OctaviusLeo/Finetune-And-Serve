"""Property-based determinism test across the pure components (task 4.7).

Feature: finetune-and-serve, Property 10: Deterministic components reproduce identically under a fixed seed
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from fts.data import build_splits, split_counts, transform_example
from fts.eval_core import bootstrap_ci
from fts.metric import score_example

_json_value = st.integers(min_value=-5, max_value=5)


@st.composite
def _raw_records(draw):
    n = draw(st.integers(min_value=1, max_value=30))
    ids = draw(st.lists(st.from_regex(r"id-[0-9a-f]{1,6}", fullmatch=True),
                        min_size=n, max_size=n, unique=True))
    records = []
    for i, rid in enumerate(ids):
        records.append(
            {"id": rid, "inputs": f"text {i}", "target": f'{{"n": {draw(_json_value)}}}'}
        )
    return records


@given(
    records=_raw_records(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=30,
    ),
)
def test_components_reproduce_identically_under_fixed_seed(records, seed, scores):
    """Feature: finetune-and-serve, Property 10: Deterministic components reproduce identically under a fixed seed."""
    # transform_example: exact equality across two same-seed runs.
    examples_a = [transform_example(r, seed) for r in records]
    examples_b = [transform_example(r, seed) for r in records]
    assert examples_a == examples_b

    present = [e for e in examples_a if e is not None]

    # Splitter: identical membership (categorical ids) + identical integer counts.
    splits_a = build_splits(present, seed)
    splits_b = build_splits(present, seed)
    assert [e.example_id for e in splits_a.train] == [e.example_id for e in splits_b.train]
    assert [e.example_id for e in splits_a.validation] == [
        e.example_id for e in splits_b.validation
    ]
    assert [e.example_id for e in splits_a.test] == [e.example_id for e in splits_b.test]
    assert split_counts(splits_a) == split_counts(splits_b)  # exact int equality

    # Metric: deterministic float equality within 1e-6.
    for example in present:
        s_a = score_example(example.target, example.target)
        s_b = score_example(example.target, example.target)
        assert abs(s_a - s_b) <= 1e-6

    # Bootstrap: same seed -> identical interval within 1e-6.
    ci_a = bootstrap_ci(scores, seed=seed)
    ci_b = bootstrap_ci(scores, seed=seed)
    assert abs(ci_a.point - ci_b.point) <= 1e-6
    assert abs(ci_a.lower - ci_b.lower) <= 1e-6
    assert abs(ci_a.upper - ci_b.upper) <= 1e-6
