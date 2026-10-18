"""Evaluation core: the bootstrap confidence interval (Req 3.2, 5.1, 5.7).

``bootstrap_ci`` resamples the per-example scores with replacement at least
1,000 times, takes the mean of each resample, and reports the 2.5th and 97.5th
percentiles of those means as the 95% interval around the sample mean. A fixed
seed makes it deterministic. In degenerate cases (for example one sample, or a
heavily skewed tiny sample) the percentile bounds are clamped so that
``lower <= point <= upper`` always holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
