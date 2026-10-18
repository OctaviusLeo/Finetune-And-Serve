"""Unit tests for scripts/check_coverage.py (the per-module coverage rule)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import check_coverage as cc

TRIVIAL = "def only():\n    return 1\n"
TWO_PUBLIC = "def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef _private():\n    return 3\n"
CLASS_METHODS = "class K:\n    def a(self):\n        pass\n\n    def b(self):\n        pass\n\n    def _c(self):\n        pass\n"


def _entry(statements: int, covered: int) -> dict:
    return {"summary": {"num_statements": statements, "covered_lines": covered}}


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_counts_public_functions_and_methods():
    assert cc.count_public_functions(TRIVIAL) == 1
    assert cc.count_public_functions(TWO_PUBLIC) == 2
    assert cc.count_public_functions(CLASS_METHODS) == 2
    assert cc.count_public_functions("class _Hidden:\n    def a(self):\n        pass\n") == 0


def test_source_lines_skip_blank_and_comment_lines():
    assert cc.count_source_lines("# c\n\nx = 1\n  # indented comment\ny = 2\n") == 2


def test_statement_percent_edges():
    assert cc.statement_percent(None) == 0.0
    assert cc.statement_percent(_entry(0, 0)) == 100.0
    assert cc.statement_percent(_entry(10, 8)) == 80.0


def test_trivial_module_passes_even_uncovered(tmp_path):
    root = _repo(tmp_path, {"fts/tiny.py": TRIVIAL, "scripts/.keep": ""})
    modules = cc.collect({"files": {}}, root)
    assert [m.path for m in modules] == ["fts/tiny.py"]
    assert cc.failures(modules) == []


def test_non_trivial_module_under_threshold_fails(tmp_path):
    root = _repo(tmp_path, {"fts/two.py": TWO_PUBLIC, "scripts/long.py": "x = 1\n" * 51})
    report = {"files": {"fts/two.py": _entry(10, 7), str(root / "scripts/long.py"): _entry(51, 51)}}
    bad = cc.failures(cc.collect(report, root))
    assert [m.path for m in bad] == ["fts/two.py"]


def test_unmeasured_non_trivial_module_counts_as_zero(tmp_path):
    root = _repo(tmp_path, {"scripts/never_imported.py": TWO_PUBLIC})
    bad = cc.failures(cc.collect({"files": {}}, root))
    assert len(bad) == 1 and bad[0].percent == 0.0


def test_main_exit_codes_and_listing(tmp_path, capsys):
    root = _repo(tmp_path, {"fts/two.py": TWO_PUBLIC})
    report = tmp_path / "cov.json"
    report.write_text(json.dumps({"files": {"fts/two.py": _entry(10, 9)}}), encoding="utf-8")
    assert cc.main([str(report), "--root", str(root)]) == 0
    assert "coverage rule OK" in capsys.readouterr().out

    report.write_text(json.dumps({"files": {"fts/two.py": _entry(10, 5)}}), encoding="utf-8")
    assert cc.main([str(report), "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "fts/two.py: 50.0%" in out


def test_absolute_path_outside_root_is_kept(tmp_path):
    other = tmp_path.parent / "elsewhere" / "x.py"
    assert cc._normalise(str(other), tmp_path) == other.as_posix()
