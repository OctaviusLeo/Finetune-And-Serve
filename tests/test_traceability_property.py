"""Feature: finetune-and-serve, Property 11: README/Model_Card number traceability.

For any number appearing in the README or Model_Card, the traceability check
passes if and only if that number can be traced to the Results_Table or another
generated file; an untraceable number is reported and causes the check to fail.

Validates: Requirements 10.4, 10.5
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.check_traceability import check_traceability

number = st.one_of(st.integers(min_value=0, max_value=10**6).map(str),
                   st.tuples(st.integers(0, 999), st.integers(0, 9999)).map(lambda t: f"{t[0]}.{t[1]:04d}"))
filler = st.sampled_from(["F1", "is", "and", "P95", "Q4_K_M", "ms", "tokens/s", "seed", "jev-1.13.0", "(95% CI)"])


@settings(max_examples=100, derandomize=True, deadline=None)
@given(traceable=st.lists(number, max_size=8, unique=True), untraceable=st.lists(number, max_size=4, unique=True), words=st.lists(filler, max_size=6), data=st.data())
def test_check_flags_exactly_the_untraceable_numbers(traceable, untraceable, words, data):
    """Feature: finetune-and-serve, Property 11: README/Model_Card number traceability."""
    untraceable = [u for u in untraceable if u not in traceable and u != "95"]
    source_text = " ".join(traceable) + " 95"  # "95" from the "(95% CI)" filler is generated too
    tokens = data.draw(st.permutations(traceable + untraceable + words))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "summary.md"
        src.write_text(source_text, encoding="utf-8")
        doc = Path(tmp) / "README.md"
        doc.write_text(" ".join(tokens) + "\n", encoding="utf-8")
        ok, bad = check_traceability(doc, [src])
    flagged = sorted(b.split(": ", 1)[1] for b in bad)
    assert flagged == sorted(untraceable)
    assert ok is (not untraceable)
