"""Property-based test for the earlier exemplar selector (task 2.10 replaces it).

Property 7 (spend cap) is tested in tests/test_spend_property.py against fts/spend.py.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from fts.baselines import select_few_shot_exemplars
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
    exemplars = select_few_shot_exemplars(splits, k, seed)
    exemplar_ids = {e.example_id for e in exemplars}
    train_ids = {e.example_id for e in splits.train}
    val_ids = {e.example_id for e in splits.validation}
    test_ids = {e.example_id for e in splits.test}

    assert exemplar_ids <= train_ids
    assert exemplar_ids.isdisjoint(val_ids)
    assert exemplar_ids.isdisjoint(test_ids)
    assert len(exemplars) <= min(k, len(splits.train))
