"""Evaluation core: bootstrap CI, Test-split evaluation and the seed summary (Req 3.2, 5.1-5.3, 5.7, 5.8).

``bootstrap_ci`` resamples the per-example scores with replacement at least
1,000 times, takes the mean of each resample, and reports the 2.5th and 97.5th
percentiles of those means as the 95% interval around the sample mean. A fixed
seed makes it deterministic. In degenerate cases (for example one sample, or a
heavily skewed tiny sample) the percentile bounds are clamped so that
``lower <= point <= upper`` always holds.

``evaluate`` scores one Run's Test_Split predictions with the same metric and
Test_Split as the baselines (a malformed Test example is reported "not scored"
with its cause and the rest still count), and ``seed_summary`` gives the mean
and the sample standard deviation of the Test F1 across the final seeds.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

MIN_RESAMPLES = 1000


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float  # lower <= point <= upper

    def as_dict(self) -> dict:
        return {"point": self.point, "lower": self.lower, "upper": self.upper}


def bootstrap_ci(samples: Sequence[float], seed: int, resamples: int = MIN_RESAMPLES, confidence: float = 0.95) -> BootstrapCI:
    """95% (by default) percentile bootstrap CI of the mean of ``samples``."""
    if len(samples) == 0:
        raise ValueError("bootstrap_ci needs at least one sample")
    if resamples < MIN_RESAMPLES:
        raise ValueError(f"bootstrap_ci needs at least {MIN_RESAMPLES} resamples, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    data = np.asarray(samples, dtype=float)
    point = float(data.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.shape[0], size=(resamples, data.shape[0]))
    means = data[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return BootstrapCI(point=point, lower=min(lower, point), upper=max(upper, point))


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    seed: int
    metric: BootstrapCI
    scored: int
    n_test: int
    test_ids_sha256: str
    not_scored: list[dict] = field(default_factory=list)  # {"id", "cause"}
    per_example: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "metric": self.metric.as_dict(),
            "scored": self.scored,
            "n_test": self.n_test,
            "not_scored": len(self.not_scored),
            "not_scored_examples": self.not_scored,
            "test_ids_sha256": self.test_ids_sha256,
        }


@dataclass(frozen=True)
class SeedSummary:
    per_seed: dict[int, float]  # seed -> Test F1
    mean: float
    spread: float  # sample standard deviation across seeds


def split_digest(examples: Sequence) -> str:
    """Digest of the Test ids in dataset order (the same digest the baselines record)."""
    return hashlib.sha256("\n".join(e.example_id for e in examples).encode("utf-8")).hexdigest()


def evaluate_predictions(run_id: str, seed: int, test: Sequence, predictions: Sequence, bootstrap_seed: int) -> EvalResult:
    """Score predictions on the Test_Split with a 95% CI; malformed Test examples are not scored."""
    from fts.infer import score

    report = score(test, predictions)
    if not report.scores:
        raise ValueError(f"{run_id}: no Test example could be scored")
    return EvalResult(run_id=run_id, seed=seed, metric=bootstrap_ci(report.scores, seed=bootstrap_seed), scored=len(report.scores),
                      n_test=len(test), test_ids_sha256=split_digest(test), not_scored=report.not_scored, per_example=report.per_example)


def evaluate(run, splits, seed: int, predict: Callable[[object, Sequence], Sequence]) -> EvalResult:
    """Evaluate one Run (a RunRecord) on the Test_Split; ``predict(run, test)`` returns its predictions."""
    test = splits.test
    return evaluate_predictions(run.run_id, run.config.seed, test, predict(run, test), bootstrap_seed=seed)


def seed_summary(per_seed: Mapping[int, float]) -> SeedSummary:
    """Mean and sample standard deviation (n - 1) of the Test F1 across at least two seeds."""
    if len(per_seed) < 2:
        raise ValueError("a seed summary needs at least two seeds")
    values = [per_seed[s] for s in sorted(per_seed)]
    return SeedSummary(per_seed={s: per_seed[s] for s in sorted(per_seed)}, mean=statistics.fmean(values), spread=statistics.stdev(values))
