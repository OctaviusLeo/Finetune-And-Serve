"""Unit tests for the baselines orchestrator + bootstrap CI (task 2.6)."""

from __future__ import annotations

import pytest

from fts.baselines import SpendCapExceeded, run_baselines
from fts.data import Example, Splits
from fts.eval_core import bootstrap_ci


def _splits() -> Splits:
    train = tuple(
        Example(example_id=f"tr-{i}", partition="train", inputs=f"in{i}", target=f'{{"n": {i}}}')
        for i in range(5)
    )
    test = tuple(
        Example(example_id=f"te-{i}", partition="test", inputs=f"in{i}", target=f'{{"n": {i}}}')
        for i in range(4)
    )
    return Splits(train=train, validation=(), test=test)


def _perfect_predict(_template: str, example: Example) -> str:
    return example.target


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    samples = [0.2, 0.4, 0.6, 0.8, 1.0]
    a = bootstrap_ci(samples, seed=123)
    b = bootstrap_ci(samples, seed=123)
    assert a == b
    assert a.lower <= a.point <= a.upper


def test_run_baselines_produces_three_results_with_perfect_predictor():
    splits = _splits()
    results = run_baselines(
        splits,
        few_shot_exemplars=2,
        spend_cap_usd=10.0,
        seed=7,
        predict=_perfect_predict,
        hosted_token_counts=[(10, 5), (10, 5)],
        price_per_prompt_token=0.001,
        price_per_completion_token=0.002,
    )
    assert [r.name for r in results] == [
        "base-zero-shot",
        "base-few-shot",
        "hosted-few-shot",
    ]
    for r in results:
        assert r.metric.point == pytest.approx(1.0)
        assert r.prompt_template
    hosted = results[-1]
    # spend = 2 requests * (10*0.001 + 5*0.002) = 2 * 0.02 = 0.04
    assert hosted.spend_usd == pytest.approx(0.04)


def test_run_baselines_raises_when_spend_cap_breached():
    splits = _splits()
    with pytest.raises(SpendCapExceeded):
        run_baselines(
            splits,
            few_shot_exemplars=1,
            spend_cap_usd=0.01,
            seed=1,
            predict=_perfect_predict,
            hosted_token_counts=[(100, 100)],
            price_per_prompt_token=0.001,
            price_per_completion_token=0.001,
        )
