"""Feature: finetune-and-serve, Property 10: Determinism under fixed seed.

For any seed and any valid input, running a deterministic component (splitter,
transform, metric, bootstrap, cost ledger) twice with the same seed and
configuration produces floats equal within 1e-6 and integers and categorical
values exactly equal (timestamps and absolute paths excluded).

Validates: Requirements 0.3, 5.7
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from fts.data import build_splits, transform_all
from fts.eval_core import bootstrap_ci
from fts.metric import score_example
from fts.spend import Prices, SpendCapExceeded, SpendLedger

word = st.from_regex(r"[a-z]{1,6}", fullmatch=True)


@st.composite
def raw_records(draw):
    n = draw(st.integers(min_value=0, max_value=25))
    out = []
    for i in range(n):
        words = draw(st.lists(word, min_size=1, max_size=4))
        val = draw(word)
        good = draw(st.booleans())
        annot = " ".join(words) + (f" [time : {val}]" if good else f" [time {val}]")
        out.append({"id": str(i), "partition": draw(st.sampled_from(["train", "dev", "test"])), "intent": draw(st.sampled_from(["a_b", "c_d"])),
                    "utt": " ".join(words) + f" {val}", "annot_utt": annot})
    return out


def run_pipeline(records, seed, predictions, token_counts, cap):
    report = transform_all(records, seed)
    splits = build_splits(report.examples, seed)
    gold = [e.target for e in report.examples]
    scores = [score_example(p, g) for p, g in zip(predictions, gold)]
    ci = bootstrap_ci(scores, seed) if scores else None
    with tempfile.TemporaryDirectory() as tmp:
        ledger = SpendLedger(Path(tmp) / "l.jsonl", cap, Prices(0.125, 0.5), clock=lambda: 0.0)
        sent = 0
        for p, c in token_counts:
            try:
                ledger.check_tokens(p, 128)
            except SpendCapExceeded:
                break
            ledger.record(p, c, "m")
            sent += 1
        total = ledger.total_usd
    return {
        "ids": [[e.example_id for e in getattr(splits, n)] for n in ("train", "validation", "test")],
        "targets": gold,
        "excluded": [(x.example_id, x.reason) for x in report.exclusions],
        "scores": scores,
        "ci": (ci.point, ci.lower, ci.upper) if ci else None,
        "sent": sent,
        "total": total,
    }


@settings(max_examples=100, derandomize=True, deadline=None)
@given(records=raw_records(), seed=st.integers(min_value=0, max_value=2**31 - 1), data=st.data(),
       token_counts=st.lists(st.tuples(st.integers(0, 5000), st.integers(0, 128)), max_size=20), cap=st.floats(0.0, 0.01))
def test_determinism_under_fixed_seed(records, seed, data, token_counts, cap):
    """Feature: finetune-and-serve, Property 10: Determinism under fixed seed."""
    n_good = sum("[time :" in r["annot_utt"] for r in records)
    predictions = data.draw(st.lists(st.sampled_from(['{"intent": "a_b", "slots": []}', "junk", '{"intent": "c_d", "slots": [{"type": "time", "value": "x"}]}']),
                                     min_size=n_good, max_size=n_good))
    first = run_pipeline(records, seed, predictions, token_counts, cap)
    second = run_pipeline(records, seed, predictions, token_counts, cap)
    for key in ("ids", "targets", "excluded", "sent"):
        assert first[key] == second[key]
    assert all(abs(a - b) <= 1e-6 for a, b in zip(first["scores"], second["scores"])) and len(first["scores"]) == len(second["scores"])
    if first["ci"] is not None:
        assert all(abs(a - b) <= 1e-6 for a, b in zip(first["ci"], second["ci"]))
    assert abs(first["total"] - second["total"]) <= 1e-6
    assert json.dumps(first["targets"]) == json.dumps(second["targets"])
