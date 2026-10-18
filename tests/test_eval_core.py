"""Unit tests for the Finetuned_Model evaluator (task 4.3).

Covers the malformed-Test 'not scored' path (Req 5.8) and seed-stable metric
values, using a dependency-injected fake predictor so no GPU is required.
"""

from __future__ import annotations

from dataclasses import dataclass

from fts.eval_core import evaluate


@dataclass(frozen=True)
class _Ex:
    example_id: str
    inputs: str
    target: object  # str for well-formed, non-str to force the 'not scored' path


@dataclass(frozen=True)
class _Splits:
    test: tuple


def _perfect_predict(example):
    """Echo the gold target so each well-formed example scores 1.0."""
    return example.target


def test_malformed_test_example_reported_not_scored_and_continues():
    """A malformed Test example is reported 'not scored' with a cause; others still score (Req 5.8)."""
    test = (
        _Ex("good-1", "a", '{"n": 1}'),
        _Ex("bad-1", "b", None),  # malformed target -> cannot be scored
        _Ex("good-2", "c", '{"n": 2}'),
    )
    result = evaluate(_perfect_predict, _Splits(test=test), seed=7)

    assert result.scored == 2
    assert len(result.not_scored) == 1
    assert result.not_scored[0].example_id == "bad-1"
    assert "target" in result.not_scored[0].reason
    # The two well-formed examples were scored perfectly.
    assert abs(result.metric.point - 1.0) <= 1e-6
    assert result.metric.lower <= result.metric.point <= result.metric.upper


def test_predictor_exception_reported_not_scored():
    """A predictor error is captured as 'not scored' and evaluation continues."""

    def flaky_predict(example):
        if example.example_id == "boom":
            raise ValueError("inference blew up")
        return example.target

    test = (
        _Ex("ok", "a", '{"n": 1}'),
        _Ex("boom", "b", '{"n": 2}'),
    )
    result = evaluate(flaky_predict, _Splits(test=test), seed=1)

    assert result.scored == 1
    assert len(result.not_scored) == 1
    assert result.not_scored[0].example_id == "boom"
    assert "prediction failed" in result.not_scored[0].reason


def test_same_seed_gives_equal_metric_within_tolerance():
    """Re-running with the same seed yields equal metric values within 1e-6."""
    test = tuple(
        _Ex(f"ex-{i}", f"in-{i}", f'{{"n": {i}}}') for i in range(12)
    )
    splits = _Splits(test=test)

    def noisy_predict(example):
        # Deterministic but imperfect: half the examples get a wrong field.
        n = int(example.example_id.split("-")[1])
        return example.target if n % 2 == 0 else '{"n": -1}'

    first = evaluate(noisy_predict, splits, seed=123)
    second = evaluate(noisy_predict, splits, seed=123)

    assert abs(first.metric.point - second.metric.point) <= 1e-6
    assert abs(first.metric.lower - second.metric.lower) <= 1e-6
    assert abs(first.metric.upper - second.metric.upper) <= 1e-6
    assert first.scored == second.scored == 12


def test_no_scorable_examples_yields_degenerate_interval():
    """When nothing can be scored, the metric is a degenerate zero interval."""
    test = (_Ex("bad-1", "a", None), _Ex("bad-2", "b", 42))
    result = evaluate(_perfect_predict, _Splits(test=test), seed=0)

    assert result.scored == 0
    assert len(result.not_scored) == 2
    assert result.metric.point == 0.0
    assert result.metric.lower == 0.0
    assert result.metric.upper == 0.0
