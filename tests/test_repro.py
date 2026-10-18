"""Unit tests for fts.repro (task 0.3; Req 0.1)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from fts import repro


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture()
def temp_repo(tmp_path: Path) -> Path:
    """A throwaway repository with one commit."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "fts").mkdir()
    (tmp_path / "fts" / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_commit_hash_is_40_hex_chars():
    assert re.fullmatch(r"[0-9a-f]{40}", repro.commit_hash())


def test_commit_hash_of_temp_repo(temp_repo):
    assert re.fullmatch(r"[0-9a-f]{40}", repro.commit_hash(temp_repo))


def test_commit_hash_without_commits_raises(tmp_path):
    _git(tmp_path, "init", "-q")
    with pytest.raises(repro.ReproError):
        repro.commit_hash(tmp_path)


def test_clean_tree_passes_and_stamps(temp_repo):
    stamp = repro.start_run(seed=3, command="python scripts/x.py", cwd=temp_repo)
    assert stamp == repro.Stamp(command="python scripts/x.py", seed=3, commit=repro.commit_hash(temp_repo))


def test_dirty_source_file_is_refused(temp_repo):
    (temp_repo / "fts" / "code.py").write_text("x = 2\n", encoding="utf-8")
    (temp_repo / "new_module.py").write_text("y = 1\n", encoding="utf-8")
    with pytest.raises(repro.ReproError) as excinfo:
        repro.start_run(seed=0, command="c", cwd=temp_repo)
    message = str(excinfo.value)
    assert "fts/code.py" in message and "new_module.py" in message


def test_changes_in_generated_locations_are_allowed(temp_repo):
    for rel in ("results/a.json", "runs/logs/x.log", "docs/ENVIRONMENT.md", ".kiro/specs/t.md", "README.md", "MODEL_CARD.md"):
        p = temp_repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("generated\n", encoding="utf-8")
    assert repro.dirty_source_paths(temp_repo) == []
    repro.require_clean_tree(temp_repo)


def test_rename_reports_the_new_path(temp_repo):
    _git(temp_repo, "mv", "fts/code.py", "fts/renamed.py")
    assert repro.dirty_source_paths(temp_repo) == ["fts/renamed.py"]


def test_stamp_round_trips_through_serialization():
    original = repro.Stamp(command="python -m fts.eval --seed 1", seed=1, commit="a" * 40)
    assert repro.deserialize(repro.serialize(original)) == original
    assert repro.from_dict(repro.to_dict(original)) == original
    current = repro.stamp(command="python -m fts.eval", seed=7)
    assert repro.deserialize(repro.serialize(current)) == current


def test_deserialize_rejects_short_commit():
    with pytest.raises(repro.ReproError):
        repro.deserialize(json.dumps({"command": "c", "seed": 0, "commit": "abc"}))


def test_write_json_embeds_stamp(tmp_path):
    stamp = repro.Stamp(command="c", seed=2, commit="b" * 40)
    out = repro.write_json(tmp_path / "sub" / "r.json", {"value": 0.5}, stamp)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"value": 0.5, "stamp": {"command": "c", "seed": 2, "commit": "b" * 40}}


def test_current_command_forms(tmp_path):
    root = repro.REPO_ROOT
    assert repro.current_command(["scripts/dry_run.py", "--seed", "0"]) == "python scripts/dry_run.py --seed 0"
    assert repro.current_command([str(root / "fts" / "eval.py"), "--seed", "1"]) == "python -m fts.eval --seed 1"
    assert repro.current_command([str(root / "scripts" / "a.py"), "x y"]) == "python scripts/a.py 'x y'"
    outside = tmp_path / "tool.py"
    assert repro.current_command([str(outside)]) == f"python {outside.as_posix()}"
    assert repro.current_command([]) == "python"


def test_start_run_uses_current_command_by_default(temp_repo, monkeypatch):
    monkeypatch.setattr(repro.sys, "argv", ["scripts/build.py", "--seed", "4"])
    assert repro.start_run(seed=4, cwd=temp_repo).command == "python scripts/build.py --seed 4"


def test_git_missing_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(repro.subprocess, "run", boom)
    with pytest.raises(repro.ReproError, match="git is not installed"):
        repro.commit_hash()


def test_non_hash_output_raises(monkeypatch):
    monkeypatch.setattr(repro, "_git", lambda args, cwd: "not-a-hash\n")
    with pytest.raises(repro.ReproError, match="not a 40-character hash"):
        repro.commit_hash()
