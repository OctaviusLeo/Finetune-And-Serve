"""Unit tests for fts.repro (tasks 0.2, 0.3; Req 0.1, 0.2)."""

from __future__ import annotations

import re

from fts import repro


def test_commit_hash_is_40_hex_chars():
    commit = repro.commit_hash()
    assert len(commit) == 40
    assert re.fullmatch(r"[0-9a-f]{40}", commit)


def test_sentinel_is_40_zeros():
    assert repro.NO_COMMIT_SENTINEL == "0" * 40
    assert len(repro.NO_COMMIT_SENTINEL) == 40


def test_stamp_round_trips_through_serialize_deserialize():
    original = repro.stamp(command="python -m fts.eval", seed=7)
    restored = repro.deserialize(repro.serialize(original))
    assert restored == original
    assert restored.command == "python -m fts.eval"
    assert restored.seed == 7
    assert len(restored.commit) == 40
