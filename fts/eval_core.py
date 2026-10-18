"""Evaluator core: bootstrap confidence intervals + Finetuned_Model evaluation
(Req 5.1, 5.2, 5.7, 5.8).

This module owns the shared bootstrap-CI primitive used across the project.
FEAT-001 (baselines) defined :class:`BootstrapCI` and :func:`bootstrap_ci` here,
minimal, and FEAT-003 reuses them (no redefinition) while adding the full
:func:`evaluate` of the Finetuned_Model on the Test_Split.

Pure logic only — no heavy-ML imports at module scope. ``bootstrap_ci`` is
deterministic under a fixed seed and always brackets the point estimate. The
model-inference predictor used by :func:`evaluate` is injected by the caller
(production wires it to a lazily-imported Finetuned_Model), so this module and
every test that imports it stay GPU-free.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from fts.metric import aggregate, score_example


@dataclass(frozen=True)
class BootstrapCI:
    """A point estimate with its lower/upper confidence bounds.

    Invariant: ``lower <= point <= upper``.
    """

    point: float
    lower: float
    upper: float


def bootstrap_ci(
    samples: Sequence[float],
    seed: int,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of ``samples``.

    Deterministic under a fixed ``seed``: the same inputs always produce the
    same interval. Uses ``resamples`` (>=1000 by default) NumPy resamples. The
    returned interval always brackets the point estimate (the sample mean) by
    construction: the bounds are clamped so ``lower <= point <= upper`` even in
    degenerate cases (e.g. a single sample).

    Raises ``ValueError`` for empty ``samples`` or ``resamples < 1``.
    """
    if len(samples) == 0:
        raise ValueError("bootstrap_ci requires at least one sample")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")

    data = np.asarray(samples, dtype=float)
    point = float(data.mean())

    rng = np.random.default_rng(seed)
    n = data.shape[0]
    idx = rng.integers(0, n, size=(resamples, n))
    means = data[idx].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))

    # Guarantee the point estimate is bracketed even in degenerate cases.
    lower = min(lower, point)
    upper = max(upper, point)
    return BootstrapCI(point=point, lower=lower, upper=upper)


@dataclass(frozen=True)
class NotScored:
    """A Test example that could not be scored, with the cause (Req 5.8)."""

    example_id: str
    reason: str


@dataclass(frozen=True)
class EvalResult:
    """Outcome of evaluating the Finetuned_Model on the Test_Split (Req 5.7, 5.8).

    ``metric`` is the Task_Metric point estimate with its 95% bootstrap CI over
    the scored Test examples. ``scored`` is the number of Test examples that were
    scored and ``not_scored`` lists every Test example skipped (malformed or
    whose prediction raised), each with its cause. The evaluation continues past
    a malformed/failing example rather than aborting.
    """

    metric: BootstrapCI
    scored: int
    not_scored: tuple[NotScored, ...] = field(default_factory=tuple)


# A predictor maps a Test example to the model's prediction string. It is
# injected so ``evaluate`` needs no GPU; production passes a predictor that
# lazily loads the Finetuned_Model + Adapter.
Predictor = Callable[[object], str]


def _example_id(example: object, index: int) -> str:
    """Best-effort id for a Test example used in 'not scored' reporting."""
    raw = getattr(example, "example_id", None)
    return raw if isinstance(raw, str) and raw else f"<index:{index}>"


def _example_target(example: object) -> str | None:
    """Return the gold target string, or ``None`` when it is missing/malformed."""
    target = getattr(example, "target", None)
    return target if isinstance(target, str) else None


def evaluate(
    predict: Predictor,
    splits: object,
    seed: int,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> EvalResult:
    """Evaluate the Finetuned_Model on the Test_Split with a 95% CI (Req 5.7, 5.8).

    For each example in ``splits.test`` the injected ``predict`` is called and
    the prediction is scored against the gold target with :func:`fts.metric.score_example`.
    A missing/malformed Test example (no string target) or one whose predictor
    raises is reported as 'not scored' with its cause and the evaluation
    continues (Req 5.8). The Task_Metric point estimate (the mean of the scored
    examples) is reported with a deterministic percentile bootstrap CI under
    ``seed``. Re-running with the same ``seed`` yields identical metric values.

    When no Test example can be scored, ``metric`` is a degenerate
    ``BootstrapCI(0.0, 0.0, 0.0)`` and every example appears in ``not_scored``.
    """
    test = tuple(getattr(splits, "test"))
    scores: list[float] = []
    not_scored: list[NotScored] = []

    for index, example in enumerate(test):
        example_id = _example_id(example, index)
        target = _example_target(example)
        if target is None:
            not_scored.append(
                NotScored(example_id=example_id, reason="missing or malformed target")
            )
            continue
        try:
            prediction = predict(example)
        except Exception as exc:  # noqa: BLE001 - report cause and keep going (Req 5.8)
            not_scored.append(
                NotScored(example_id=example_id, reason=f"prediction failed: {exc}")
            )
            continue
        scores.append(score_example(prediction, target))

    if not scores:
        metric = BootstrapCI(point=0.0, lower=0.0, upper=0.0)
    else:
        point = aggregate(scores)
        ci = bootstrap_ci(scores, seed=seed, resamples=resamples, confidence=confidence)
        # Report the Task_Metric point as the aggregate mean, keeping the CI
        # bounds from the bootstrap (which bracket the same mean).
        metric = BootstrapCI(point=point, lower=min(ci.lower, point), upper=max(ci.upper, point))

    return EvalResult(metric=metric, scored=len(scores), not_scored=tuple(not_scored))
