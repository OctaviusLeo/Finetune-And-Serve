"""CPU Smoke_Evaluation (Req 9.3; task 6.5).

Invocable as ``python -m fts.smoke_eval``. Runs the data-processing transform
(:func:`fts.data.transform_example`) and the Task_Metric
(:func:`fts.metric.score_example` / :func:`fts.metric.aggregate`) end to end on
a tiny fixed CPU-only input, producing a deterministic result and exiting
non-zero on any failure.

GPU-free by construction: this module imports ONLY the pure-logic
:mod:`fts.data` and :mod:`fts.metric` helpers and never imports
``torch``/``transformers`` or requires a GPU, so it is safe to run in CPU-only
CI. The fixed input and the expected aggregate score are the deterministic
contract the smoke test asserts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from fts.data import transform_example
from fts.metric import aggregate, score_example

#: Tiny fixed CPU-only input: raw records paired with the model "prediction" the
#: smoke run scores against each record's gold target. Chosen so the end-to-end
#: aggregate score is a stable, hand-checkable value (see EXPECTED_SCORE).
SMOKE_RECORDS: tuple[tuple[dict, str], ...] = (
    # Exact match -> field-F1 1.0.
    ({"id": "s1", "inputs": "amount 10 to Bob", "target": '{"amount": 10, "to": "Bob"}'},
     '{"amount": 10, "to": "Bob"}'),
    # One of two fields correct -> field-F1 0.5.
    ({"id": "s2", "inputs": "amount 20 to Ann", "target": '{"amount": 20, "to": "Ann"}'},
     '{"amount": 20, "to": "Eve"}'),
    # Fully wrong -> field-F1 0.0.
    ({"id": "s3", "inputs": "amount 30 to Cat", "target": '{"amount": 30, "to": "Cat"}'},
     '{"amount": 99, "to": "Dog"}'),
)

#: Deterministic expected aggregate Task_Metric over SMOKE_RECORDS: mean of
#: (1.0, 0.5, 0.0) = 0.5. Asserted exactly by the smoke run and its test.
EXPECTED_SCORE: float = 0.5

#: Seed passed to the deterministic transform (the transform ignores it, but it
#: is threaded through for interface symmetry with the rest of the pipeline).
SMOKE_SEED: int = 0


class SmokeEvalError(Exception):
    """Raised when the Smoke_Evaluation fails (malformed input or mismatch)."""


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of the Smoke_Evaluation."""

    scored: int
    aggregate_score: float


def run_smoke_eval(
    records: tuple[tuple[dict, str], ...] = SMOKE_RECORDS,
    seed: int = SMOKE_SEED,
) -> SmokeResult:
    """Run the transform + Task_Metric end to end on the fixed input.

    For each ``(raw, prediction)`` pair the raw record is transformed via
    :func:`fts.data.transform_example`; a malformed record (transform returns
    ``None``) raises :class:`SmokeEvalError`. Each surviving example's gold
    target is scored against ``prediction`` with the Task_Metric and the scores
    are aggregated. Deterministic: identical ``records``/``seed`` always yield
    the same :class:`SmokeResult`.
    """
    scores: list[float] = []
    for raw, prediction in records:
        example = transform_example(raw, seed)
        if example is None:
            raise SmokeEvalError(f"malformed smoke input: {raw!r}")
        scores.append(score_example(prediction, example.target))
    return SmokeResult(scored=len(scores), aggregate_score=aggregate(scores))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exits 0 on success, non-zero on any failure (Req 9.3)."""
    try:
        result = run_smoke_eval()
    except SmokeEvalError as exc:
        print(f"FAIL: smoke evaluation error: {exc}", file=sys.stderr)
        return 1

    if abs(result.aggregate_score - EXPECTED_SCORE) > 1e-9:
        print(
            "FAIL: smoke evaluation score "
            f"{result.aggregate_score} != expected {EXPECTED_SCORE}",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: smoke evaluation scored {result.scored} examples; "
        f"aggregate Task_Metric {result.aggregate_score}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
