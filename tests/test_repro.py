"""Unit tests for fts.repro (Req 0.1). Task 0.3 extends these."""

from __future__ import annotations

import re

from fts import repro


def test_commit_hash_is_40_hex_chars():
    assert re.fullmatch(r"[0-9a-f]{40}", repro.commit_hash())


def test_stamp_round_trips_through_serialize_deserialize():
    original = repro.stamp(command="python -m fts.eval", seed=7)
    assert repro.deserialize(repro.serialize(original)) == original
