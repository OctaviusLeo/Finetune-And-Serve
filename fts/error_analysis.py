"""Error analysis: seed summary + grouped Test failures (Req 6.1-6.4).

Two pure, GPU-free pieces:

* :func:`seed_summary` aggregates a per-seed Task_Metric into a mean and a
  spread (the max-minus-min range), the stability signal reported across the
  >=3 final-seed Runs.
* :func:`error_analysis` selects at least 50 Test failures (lowest-scoring Test
  examples by the Task_Metric, or every available failure when fewer than 50
  exist) and assigns each to a documented failure group via an injected
  ``classify`` callable, returning per-group counts that always sum to the
  analyzed total.

Model inference is injected (the ``predict`` callable), so this module and its
tests never import the heavy ML stack.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from fts.metric import score_example

#: Minimum number of Test failures to analyze when that many are available
#: (Req 6.2). Fewer are analyzed only when fewer failures exist.
MIN_FAILURES: int = 50

#: A Test example scoring at or below this threshold counts as a failure.
FAILURE_THRESHOLD: float = 1.0


@dataclass(frozen=True)
class SeedSummary:
    """Per-seed Task_Metric with its mean and spread across seeds (Req 6.4).

    ``per_seed`` maps each seed to its Task_Metric point estimate, ``mean`` is
    the arithmetic mean over the seeds, and ``spread`` is the max-minus-min
    range (``0.0`` for a single seed).
    """

    per_seed: Mapping[int, float]
    mean: float
    spread: float


@dataclass(frozen=True)
class FailureGroup:
    """A documented failure category with the count of failures assigned to it."""

    group: str
    count: int


@dataclass(frozen=True)
class ErrorAnalysis:
    """Grouped error analysis over the selected Test failures (Req 6.2, 6.3).

    ``analyzed`` is the number of Test failures examined and ``groups`` lists the
    per-group counts, which always sum to ``analyzed``.
    """

    analyzed: int
    groups: tuple[FailureGroup, ...]


def seed_summary(per_seed: Mapping[int, float]) -> SeedSummary:
    """Summarize a per-seed Task_Metric with its mean and spread (Req 6.4).

    Pure. ``mean`` is the arithmetic mean of the values and ``spread`` is
    ``max - min`` (``0.0`` when a single seed is given). Raises ``ValueError``
    for an empty mapping.
    """
    if not per_seed:
        raise ValueError("seed_summary requires at least one seed")
    values = list(per_seed.values())
    mean = sum(values) / len(values)
    spread = max(values) - min(values)
    return SeedSummary(per_seed=dict(per_seed), mean=mean, spread=spread)


def _select_failures(
    predict: Callable[[object], str],
    test: Sequence[object],
) -> list[object]:
    """Return the Test examples that are failures, worst (lowest score) first.

    An example is a failure when its Task_Metric score is below
    :data:`FAILURE_THRESHOLD`. Selection is deterministic: failures are ordered
    by ascending score, ties broken by their original index.
    """
    scored: list[tuple[float, int, object]] = []
    for index, example in enumerate(test):
        target = getattr(example, "target", None)
        if not isinstance(target, str):
            # A missing/malformed target cannot be scored; treat as a failure at
            # the worst score so it is prioritized for analysis.
            scored.append((0.0, index, example))
            continue
        score = score_example(predict(example), target)
        if score < FAILURE_THRESHOLD:
            scored.append((score, index, example))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [example for _score, _index, example in scored]


def group_failures(
    failures: Sequence[object],
    classify: Callable[[object], str],
) -> ErrorAnalysis:
    """Assign each failure to a documented group and count per group (Req 6.3).

    Pure aggregation: ``classify(example)`` returns the group name for a failure
    and this tallies the per-group counts. Groups appear in first-seen order and
    their counts always sum to ``len(failures)``.
    """
    counts: dict[str, int] = {}
    order: list[str] = []
    for example in failures:
        group = classify(example)
        if group not in counts:
            counts[group] = 0
            order.append(group)
        counts[group] += 1
    groups = tuple(FailureGroup(group=name, count=counts[name]) for name in order)
    return ErrorAnalysis(analyzed=len(failures), groups=groups)


def error_analysis(
    predict: Callable[[object], str],
    splits: object,
    classify: Callable[[object], str],
) -> ErrorAnalysis:
    """Select >=50 Test failures and group them by cause (Req 6.2, 6.3).

    Scores every Test example with the injected ``predict`` + the Task_Metric,
    selects the failures (lowest-scoring first) up to :data:`MIN_FAILURES`, or
    every available failure when fewer than 50 exist, and assigns each selected
    failure to a documented group via ``classify``. The returned per-group
    counts always sum to the analyzed total, and ``analyzed`` is
    ``min(MIN_FAILURES, available_failures)``.
    """
    test = tuple(getattr(splits, "test"))
    failures = _select_failures(predict, test)
    selected = failures[:MIN_FAILURES]
    return group_failures(selected, classify)
