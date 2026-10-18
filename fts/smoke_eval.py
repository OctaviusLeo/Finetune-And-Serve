"""CPU Smoke_Evaluation (Req 9.3, 9.4): ``python -m fts.smoke_eval``.

Runs the data transform, the splitter, the overlap check and the Task_Metric end
to end on a tiny fixed input in tests/fixtures/ (six MASSIVE-format records, one
malformed, and fixed predictions for the Test examples). It needs no GPU, model
or network, prints a deterministic result, and exits 1 on any failure or on any
difference from the expected result. Expected: splits 2 / 1 / 2, 1 excluded, no
overlap, and mean Test F1 0.9 (design.md's hand-computed 0.8 example plus one
exact match).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fts import data
from fts.metric import aggregate, score_example

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SEED = 0
EXPECTED = {"counts": {"train": 2, "validation": 1, "test": 2}, "excluded": 1, "overlap": 0, "scores": {"s4": 0.8, "s5": 1.0}, "mean_f1": 0.9}
TOLERANCE = 1e-9


class SmokeError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SmokeError(f"{path.name}: invalid JSON line: {exc}") from exc
    return rows


def run(fixtures: Path = FIXTURES) -> dict:
    report = data.transform_all(read_jsonl(fixtures / "smoke_massive.jsonl"), SEED)
    splits = data.build_splits(report.examples, SEED)
    overlaps = data.check_overlap(splits)
    predictions = {row["id"]: row["prediction"] for row in read_jsonl(fixtures / "smoke_predictions.jsonl")}
    scores = {}
    for ex in splits.test:
        if ex.example_id not in predictions:
            raise SmokeError(f"no prediction for Test example {ex.example_id}")
        scores[ex.example_id] = score_example(predictions[ex.example_id], ex.target)
    if not scores:
        raise SmokeError("no Test example to score")
    return {
        "counts": data.split_counts(splits),
        "excluded": report.excluded_count,
        "overlap": data.overlap_total(overlaps),
        "scores": scores,
        "mean_f1": aggregate(list(scores.values())),
    }


def mismatches(result: dict, expected: dict = EXPECTED) -> list[str]:
    out = [f"{k}: got {result[k]!r}, expected {expected[k]!r}" for k in ("counts", "excluded", "overlap") if result[k] != expected[k]]
    if set(result["scores"]) != set(expected["scores"]) or any(abs(result["scores"][i] - v) > TOLERANCE for i, v in expected["scores"].items()):
        out.append(f"scores: got {result['scores']}, expected {expected['scores']}")
    if abs(result["mean_f1"] - expected["mean_f1"]) > TOLERANCE:
        out.append(f"mean_f1: got {result['mean_f1']}, expected {expected['mean_f1']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fts.smoke_eval", description="CPU smoke evaluation of the data pipeline and metric.")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = parser.parse_args(argv)
    try:
        result = run(args.fixtures)
    except (SmokeError, OSError, KeyError, ValueError) as exc:
        print(f"FAIL: smoke evaluation error: {exc}", file=sys.stderr)
        return 1
    print(f"splits {result['counts']}, excluded {result['excluded']}, overlap {result['overlap']}, "
          f"Test F1 {', '.join(f'{k}={v:.4f}' for k, v in sorted(result['scores'].items()))}, mean {result['mean_f1']:.4f}")
    problems = mismatches(result)
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    if not problems:
        print("smoke evaluation OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
