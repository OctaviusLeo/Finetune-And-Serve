"""Property-based tests for baselines (task 2.7).

Feature: finetune-and-serve, Property 7: Spend equals the formula over completed requests and never exceeds the cap
Feature: finetune-and-serve, Property 8: Few-shot exemplars subset of Train, disjoint from Val/Test
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from fts.baselines import accumulate_spend, select_few_shot_exemplars
from fts.data import Example, Splits

_example_id = st.from_regex(r"id-[0-9a-f]{1,8}", fullmatch=True)


@st.composite
def _splits(draw):
    n = draw(st.integers(min_value=0, max_value=30))
    ids = draw(st.lists(_example_id, min_size=n, max_size=n, unique=True))
    examples = [Example(example_id=i, inputs="in", target="out") for i in ids]
    # Partition deterministically into train/val/test.
    n_train = len(examples) * 6 // 10
    n_val = len(examples) * 2 // 10
    return Splits(
        train=tuple(examples[:n_train]),
        validation=tuple(examples[n_train : n_train + n_val]),
        test=tuple(examples[n_train + n_val :]),
    )


@given(
    splits=_splits(),
    k=st.integers(min_value=0, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_exemplars_never_leak_val_or_test(splits, k, seed):
    """Feature: finetune-and-serve, Property 8: Few-shot exemplars subset of Train, disjoint from Val/Test."""
    exemplars = select_few_shot_exemplars(splits, k, seed)
    exemplar_ids = {e.example_id for e in exemplars}
    train_ids = {e.example_id for e in splits.train}
    val_ids = {e.example_id for e in splits.validation}
    test_ids = {e.example_id for e in splits.test}

    assert exemplar_ids <= train_ids
    assert exemplar_ids.isdisjoint(val_ids)
    assert exemplar_ids.isdisjoint(test_ids)
    assert len(exemplars) <= min(k, len(splits.train))


@given(
    token_counts=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=1000),
            st.integers(min_value=0, max_value=1000),
        ),
        min_size=0,
        max_size=30,
    ),
    price_prompt=st.floats(min_value=0.0, max_value=0.01, allow_nan=False),
    price_completion=st.floats(min_value=0.0, max_value=0.01, allow_nan=False),
    cap=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
)
def test_spend_matches_formula_and_respects_cap(
    token_counts, price_prompt, price_completion, cap
):
    """Feature: finetune-and-serve, Property 7: Spend equals the formula over completed requests and never exceeds the cap."""
    result = accumulate_spend(token_counts, price_prompt, price_completion, cap)

    # Reported spend never exceeds the cap.
    assert result.spend_usd <= cap + 1e-12

    # Reported spend equals the exact formula summed over completed requests.
    expected = sum(
        p * price_prompt + c * price_completion
        for p, c in token_counts[: result.completed]
    )
    assert abs(result.spend_usd - expected) < 1e-9

    # When capped, the capping (next) request was NOT completed: adding it would
    # breach the cap.
    if result.capped:
        assert result.completed < len(token_counts)
        p, c = token_counts[result.completed]
        next_cost = p * price_prompt + c * price_completion
        assert result.spend_usd + next_cost > cap
    else:
        assert result.completed == len(token_counts)
