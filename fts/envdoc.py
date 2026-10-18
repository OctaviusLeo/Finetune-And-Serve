"""Write evidence sections into Markdown documents such as docs/ENVIRONMENT.md.

A section is a ``## Heading`` line and everything up to the next ``## `` heading.
:func:`replace_section` swaps one section's body and leaves the rest untouched,
so each script owns its own section and re-running it replaces stale evidence.
"""

from __future__ import annotations

from pathlib import Path

from fts.repro import Stamp

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENT_MD = REPO_ROOT / "docs" / "ENVIRONMENT.md"


def replace_section_text(text: str, heading: str, body: str) -> str:
    """Return ``text`` with the body under ``## heading`` replaced (appended if absent)."""
    lines = text.splitlines()
    target = f"## {heading}"
    new_body = body.strip("\n").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == target)
    except StopIteration:
        out = lines + ([""] if lines and lines[-1].strip() else []) + [target, ""] + new_body
        return "\n".join(out) + "\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    out = lines[: start + 1] + [""] + new_body + [""] + lines[end:]
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def replace_section(heading: str, body: str, path: Path | str = ENVIRONMENT_MD) -> Path:
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(replace_section_text(text, heading, body), encoding="utf-8")
    return p


def read_section(heading: str, path: Path | str = ENVIRONMENT_MD) -> str | None:
    """The body under ``## heading``, or None when the section is missing."""
    p = Path(path)
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    target = f"## {heading}"
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == target)
    except StopIteration:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1 : end]).strip("\n")


def stamp_lines(stamp: Stamp) -> list[str]:
    """The three provenance bullets every evidence section carries."""
    return [
        f"- Command: `{stamp.command}`",
        f"- Seed: `{stamp.seed}`",
        f"- Commit: `{stamp.commit}`",
    ]


def code_block(text: str) -> list[str]:
    return ["```text", *text.rstrip("\n").splitlines(), "```"]
