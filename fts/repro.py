"""Reproducibility helper: stamp every reported number with its generating
command, seed, and the full 40-character git commit hash (Req 0.1).

Sentinel behavior
-----------------
`commit_hash()` shells out to ``git rev-parse HEAD``. When the repository has
no commits yet (``git rev-parse`` exits non-zero) or git is otherwise
unavailable, there is no commit to record. Rather than crashing, this module
returns a documented sentinel of 40 zero characters (``"0" * 40``). This keeps
the 40-hex *shape* contract stable for downstream serialization and tests while
making it obvious that no real commit was captured.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass

#: Documented sentinel returned when no git commit is available (e.g. a repo
#: with no commits yet). Preserves the 40-character shape of a real hash.
NO_COMMIT_SENTINEL: str = "0" * 40


def commit_hash() -> str:
    """Return the current 40-hex git commit hash.

    Returns :data:`NO_COMMIT_SENTINEL` (40 zeros) when ``git rev-parse HEAD``
    fails because the repo has no commits yet or git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return NO_COMMIT_SENTINEL

    commit = out.stdout.strip()
    if len(commit) != 40:
        return NO_COMMIT_SENTINEL
    return commit


@dataclass(frozen=True)
class Stamp:
    """Provenance stamp for a Measured_Value."""

    command: str
    seed: int
    commit: str


def stamp(command: str, seed: int) -> Stamp:
    """Capture the generating ``command`` and ``seed`` with the current commit."""
    return Stamp(command=command, seed=seed, commit=commit_hash())


def serialize(stamp: Stamp) -> str:
    """Serialize a :class:`Stamp` to a JSON string."""
    return json.dumps(asdict(stamp), sort_keys=True)


def deserialize(data: str) -> Stamp:
    """Reconstruct a :class:`Stamp` from a JSON string produced by :func:`serialize`."""
    obj = json.loads(data)
    return Stamp(command=obj["command"], seed=int(obj["seed"]), commit=obj["commit"])
