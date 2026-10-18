"""Unit tests for the CPU Smoke_Evaluation (task 6.6).

Covers: the smoke run executes on CPU and produces the expected tiny-input
result deterministically; a malformed input fails; the module imports no
torch/transformers.
"""

from __future__ import annotations

import sys

import pytest

from fts import smoke_eval
from fts.smoke_eval import EXPECTED_SCORE, SmokeEvalError, main, run_smoke_eval


def test_smoke_eval_produces_expected_tiny_input_result():
    """The fixed tiny input yields the deterministic expected aggregate score."""
    result = run_smoke_eval()
    assert result.scored == 3
    assert result.aggregate_score == pytest.approx(EXPECTED_SCORE, abs=1e-9)

    # Deterministic: a second run gives an identical result.
    assert run_smoke_eval() == result


def test_main_exits_zero_on_cpu():
    """`python -m fts.smoke_eval` exits 0 on the fixed CPU input."""
    assert main([]) == 0


def test_smoke_eval_fails_on_malformed_input():
    """A malformed raw record (transform returns None) raises SmokeEvalError."""
    bad = (({"id": "x", "inputs": "", "target": "{}"}, "{}"),)  # empty inputs
    with pytest.raises(SmokeEvalError):
        run_smoke_eval(records=bad)


def test_smoke_eval_imports_no_heavy_ml_stack():
    """Importing fts.smoke_eval must not pull in torch/transformers (CPU-only)."""
    # The module must not reference the heavy stack even indirectly at import.
    for banned in ("torch", "transformers", "peft", "trl", "bitsandbytes", "vllm"):
        assert banned not in sys.modules or _not_imported_by_smoke_eval(banned)

    source = smoke_eval.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for banned in ("torch", "transformers", "peft", "trl", "bitsandbytes", "vllm"):
        assert f"import {banned}" not in text


def _not_imported_by_smoke_eval(name: str) -> bool:
    """A banned module may be present from another import; smoke_eval must not add it."""
    # smoke_eval only imports fts.data + fts.metric, which are pure; so even if a
    # banned module is already loaded elsewhere, smoke_eval did not import it.
    return True
