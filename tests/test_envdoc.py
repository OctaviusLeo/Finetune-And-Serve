"""Unit tests for fts.envdoc (Markdown evidence sections)."""

from __future__ import annotations

from fts import envdoc
from fts.repro import Stamp

DOC = "# Title\n\nIntro.\n\n## A\n\nold a\n\n## B\n\nold b\n"


def test_replace_existing_section_keeps_others():
    out = envdoc.replace_section_text(DOC, "A", "new a\nline 2")
    assert "new a\nline 2" in out and "old a" not in out
    assert "## B\n\nold b" in out and out.startswith("# Title")


def test_replace_last_section_and_append_missing():
    out = envdoc.replace_section_text(DOC, "B", "new b")
    assert out.endswith("## B\n\nnew b\n")
    out2 = envdoc.replace_section_text(DOC, "C", "c body")
    assert out2.endswith("## C\n\nc body\n") and "old b" in out2


def test_append_to_empty_text():
    assert envdoc.replace_section_text("", "X", "x") == "## X\n\nx\n"


def test_replace_and_read_section_on_disk(tmp_path):
    p = tmp_path / "docs" / "ENV.md"
    assert envdoc.read_section("A", p) is None
    envdoc.replace_section("A", "first", p)
    envdoc.replace_section("A", "second", p)
    assert envdoc.read_section("A", p) == "second"
    assert envdoc.read_section("Missing", p) is None
    assert p.read_text(encoding="utf-8").count("## A") == 1


def test_stamp_lines_and_code_block():
    lines = envdoc.stamp_lines(Stamp(command="python x.py", seed=0, commit="c" * 40))
    assert lines == ["- Command: `python x.py`", "- Seed: `0`", f"- Commit: `{'c' * 40}`"]
    assert envdoc.code_block("a\nb\n") == ["```text", "a", "b", "```"]
