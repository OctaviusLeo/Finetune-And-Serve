"""Deterministic data pipeline: transform, split, and overlap check (Req 2.3-2.7).

Pure logic, no heavy-ML imports. ``transform_example`` is deterministic and
returns ``None`` for malformed input while reporting the id and reason;
``build_splits`` produces reproducible seeded membership; ``check_overlap``
reports any example id shared across splits. Each split's example count is a
Measured_Value (see :func:`split_counts`).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Example:
    """A single transformed training/eval example."""

    example_id: str
    inputs: str
    target: str


@dataclass(frozen=True)
class Splits:
    """Train / Validation / Test partition of :class:`Example` objects."""

    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]


@dataclass(frozen=True)
class Exclusion:
    """Record of a malformed raw record excluded by :func:`transform_example`."""

    example_id: str
    reason: str


@dataclass
class TransformReport:
    """Diagnostics collected while transforming a batch of raw records."""

    examples: list[Example] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)

    @property
    def excluded_count(self) -> int:
        """Total number of malformed records excluded."""
        return len(self.exclusions)


def transform_example(raw: dict, seed: int) -> Example | None:
    """Deterministically transform one raw record into an :class:`Example`.

    Returns ``None`` for malformed input. A record is malformed when it is not
    a mapping, is missing a non-empty ``id``/``inputs``/``target`` field, or any
    of those fields is not a string. The transform is pure and deterministic:
    the same ``raw`` and ``seed`` always yield the same result. ``seed`` is
    accepted for interface symmetry with the rest of the pipeline; the transform
    itself does not use randomness, so exclusion is deterministic too.
    """
    if not isinstance(raw, dict):
        return None

    example_id = raw.get("id")
    inputs = raw.get("inputs")
    target = raw.get("target")

    if not isinstance(example_id, str) or not example_id:
        return None
    if not isinstance(inputs, str) or not inputs:
        return None
    if not isinstance(target, str):
        return None

    return Example(
        example_id=example_id,
        inputs=inputs.strip(),
        target=target.strip(),
    )


def transform_batch(raws: list[dict], seed: int) -> TransformReport:
    """Transform a batch, collecting successes and malformed-record diagnostics.

    Deterministic: identical input yields an identical report, including the
    excluded count and the id+reason of every exclusion.
    """
    report = TransformReport()
    for index, raw in enumerate(raws):
        example = transform_example(raw, seed)
        if example is None:
            raw_id = raw.get("id") if isinstance(raw, dict) else None
            example_id = raw_id if isinstance(raw_id, str) and raw_id else f"<index:{index}>"
            report.exclusions.append(
                Exclusion(example_id=example_id, reason="malformed raw record")
            )
        else:
            report.examples.append(example)
    return report


def _seeded_sort_key(example_id: str, seed: int) -> str:
    """Deterministic shuffle key: hash of (seed, example_id)."""
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    return digest


def build_splits(
    examples: list[Example],
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Splits:
    """Partition ``examples`` into train/validation/test with reproducible membership.

    Membership is a deterministic function of ``seed`` and each example's id:
    examples are ordered by a seeded hash of their id (stable, independent of
    input order) and then sliced by ``ratios``. The same ``examples`` (in any
    order) and ``seed`` always produce the same membership.
    """
    train_r, val_r, test_r = ratios
    if min(ratios) < 0 or sum(ratios) <= 0:
        raise ValueError(f"invalid split ratios: {ratios!r}")

    # Deduplicate by id (keep first occurrence) so membership is well defined.
    unique: dict[str, Example] = {}
    for ex in examples:
        unique.setdefault(ex.example_id, ex)

    ordered = sorted(unique.values(), key=lambda ex: _seeded_sort_key(ex.example_id, seed))

    n = len(ordered)
    total = train_r + val_r + test_r
    n_train = int(n * (train_r / total))
    n_val = int(n * (val_r / total))

    train = tuple(ordered[:n_train])
    validation = tuple(ordered[n_train : n_train + n_val])
    test = tuple(ordered[n_train + n_val :])
    return Splits(train=train, validation=validation, test=test)


def check_overlap(splits: Splits) -> dict[str, list[str]]:
    """Return the shared example ids per split pair; empty dict when disjoint.

    Keys are ``"train-validation"``, ``"train-test"``, and
    ``"validation-test"``. Only pairs with a non-empty intersection appear, each
    mapping to the sorted list of shared ids.
    """
    train_ids = {ex.example_id for ex in splits.train}
    val_ids = {ex.example_id for ex in splits.validation}
    test_ids = {ex.example_id for ex in splits.test}

    overlaps: dict[str, list[str]] = {}
    pairs = (
        ("train-validation", train_ids & val_ids),
        ("train-test", train_ids & test_ids),
        ("validation-test", val_ids & test_ids),
    )
    for name, shared in pairs:
        if shared:
            overlaps[name] = sorted(shared)
    return overlaps


def split_counts(splits: Splits) -> dict[str, int]:
    """Return each split's example count as a Measured_Value mapping."""
    return {
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }
