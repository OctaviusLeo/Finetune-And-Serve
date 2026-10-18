"""Reproducibility helper: stamp every reported result with command, seed and commit (Req 0.1, 0.2).

Every script that writes a reported number calls :func:`start_run` before it
writes anything. It refuses to continue when the working tree has changes
outside the paths a run is allowed to write (so the number really comes from the
committed code), then returns a :class:`Stamp` holding the generating command,
the seed, and the full 40-character commit hash.

The only Git commands this module runs against the repository are the two
read-only ones: ``git rev-parse HEAD`` and ``git --no-optional-locks status --porcelain``.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Paths a run may change without making the tree "dirty" for stamping purposes:
#: generated outputs, local run data, docs evidence, generated docs, and Kiro's task list.
ALLOWED_DIRTY_PREFIXES: tuple[str, ...] = ("results/", "runs/", "docs/", ".kiro/")
ALLOWED_DIRTY_FILES: tuple[str, ...] = ("README.md", "MODEL_CARD.md")

_HASH_RE = re.compile(r"^[0-9a-f]{40}$")


class ReproError(RuntimeError):
    """The commit cannot be determined, or the tree has uncommitted source changes."""


@dataclass(frozen=True)
class Stamp:
    """Provenance of a Measured_Value."""

    command: str
    seed: int
    commit: str  # full 40-character hash


def _git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise ReproError("git is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise ReproError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return out.stdout


def commit_hash(cwd: Path = REPO_ROOT) -> str:
    """The 40-character hash of HEAD; raises ReproError when there is no commit."""
    commit = _git(["rev-parse", "HEAD"], cwd).strip()
    if not _HASH_RE.match(commit):
        raise ReproError(f"git rev-parse HEAD returned {commit!r}, not a 40-character hash")
    return commit


def _porcelain_path(line: str) -> str:
    # Porcelain v1: "XY path", or "XY old -> new" for a rename; paths with spaces are quoted.
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip().strip('"')


def is_allowed_dirty(path: str) -> bool:
    return path in ALLOWED_DIRTY_FILES or any(path.startswith(p) for p in ALLOWED_DIRTY_PREFIXES)


def dirty_source_paths(cwd: Path = REPO_ROOT) -> list[str]:
    """Changed or untracked paths outside the allowed generated-output locations."""
    status = _git(["--no-optional-locks", "status", "--porcelain"], cwd)
    paths = [_porcelain_path(line) for line in status.splitlines() if line.strip()]
    return sorted(p for p in paths if not is_allowed_dirty(p))


def require_clean_tree(cwd: Path = REPO_ROOT) -> None:
    """Raise ReproError naming every uncommitted source path, if any."""
    dirty = dirty_source_paths(cwd)
    if dirty:
        listed = "\n  ".join(dirty)
        raise ReproError("refusing to stamp a reported result; commit these changes first:\n  " + listed)


def current_command(argv: list[str] | None = None, root: Path = REPO_ROOT) -> str:
    """The generating command, e.g. ``python scripts/dry_run.py --seed 0`` or ``python -m fts.eval ...``.

    ``python -m fts.eval`` puts the module's file path in ``argv[0]``, so a file
    inside the ``fts`` package is reported in its ``-m`` form.
    """
    args = list(sys.argv if argv is None else argv)
    if not args or not args[0]:
        return "python"
    first = Path(args[0])
    if first.is_absolute():
        try:
            first = first.resolve().relative_to(root.resolve())
        except ValueError:
            pass
    parts = first.with_suffix("").parts
    if first.suffix == ".py" and len(parts) >= 2 and parts[0] == "fts":
        return shlex.join(["python", "-m", ".".join(parts), *args[1:]])
    return shlex.join(["python", first.as_posix(), *args[1:]])


def make_stamp(command: str, seed: int, cwd: Path = REPO_ROOT) -> Stamp:
    return Stamp(command=command, seed=int(seed), commit=commit_hash(cwd))


def stamp(command: str, seed: int) -> Stamp:
    """Stamp without the clean-tree check (for tests and in-memory reports)."""
    return make_stamp(command, seed)


def start_run(seed: int, command: str | None = None, cwd: Path = REPO_ROOT) -> Stamp:
    """Check the tree is clean, then stamp. Call before writing any output."""
    require_clean_tree(cwd)
    return make_stamp(current_command() if command is None else command, seed, cwd)


def to_dict(s: Stamp) -> dict:
    return asdict(s)


def from_dict(data: dict) -> Stamp:
    commit = str(data["commit"])
    if not _HASH_RE.match(commit):
        raise ReproError(f"stamp commit {commit!r} is not a 40-character hash")
    return Stamp(command=str(data["command"]), seed=int(data["seed"]), commit=commit)


def serialize(s: Stamp) -> str:
    return json.dumps(to_dict(s), sort_keys=True)


def deserialize(text: str) -> Stamp:
    return from_dict(json.loads(text))


def write_json(path: Path | str, payload: dict, s: Stamp) -> Path:
    """Write ``payload`` plus its stamp as indented, key-sorted UTF-8 JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["stamp"] = to_dict(s)
    out.write_text(json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
