"""README / Model_Card number traceability check (Req 10.4, 10.5).

Every numeric token in README.md and MODEL_CARD.md must appear in a file a
repository script generated: results/summary.md, results/*.json,
results/doc_facts.json, or the evidence in docs/ENVIRONMENT.md. URLs are
ignored. A number glued to letters or joined by "-" or "/" (P95, F1, Q4_K_M,
gpt-6-luna, Ubuntu-24.04, Qwen2.5-1.5B) is part of a name, not a number.
Thousands separators are ignored ("2,974" == "2974").

    python scripts/check_traceability.py [doc ...]   (default: README.md MODEL_CARD.md)

Prints every untraceable number with its line and exits 1 if there is any.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
URL = re.compile(r"https?://[^\s)>\]]+")
DOC_NUMBER = re.compile(r"(?<![\w.\-/])\d+(?:[.,]\d+)*(?![\w\-/])")
SOURCE_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def normalize(token: str) -> str:
    token = re.sub(r"(?<=\d),(?=\d{3}\b)", "", token)
    return token.rstrip(".,")


def doc_numbers(text: str) -> list[tuple[int, str]]:
    """(line number, normalized token) for every number in a document, URLs removed."""
    found = []
    for i, line in enumerate(text.splitlines(), start=1):
        for m in DOC_NUMBER.finditer(URL.sub(" ", line)):
            found.append((i, normalize(m.group(0))))
    return found


def source_numbers(texts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for text in texts:
        for m in SOURCE_NUMBER.finditer(text):
            tok = normalize(m.group(0))
            out.add(tok)
    return out


def default_sources(root: Path = REPO_ROOT) -> list[Path]:
    results = root / "results"
    paths = [results / "summary.md", root / "docs" / "ENVIRONMENT.md"]
    paths += sorted(results.glob("*.json")) + sorted((results / "baselines").glob("*.json")) + sorted((results / "runs").glob("*.json"))
    return [p for p in paths if p.exists()]


def check_traceability(doc_path: Path | str, generated_sources: Sequence[Path | str]) -> tuple[bool, list[str]]:
    """(ok, untraceable numbers as 'line N: token'); ok is False when any number cannot be traced."""
    known = source_numbers(Path(p).read_text(encoding="utf-8") for p in generated_sources)
    text = Path(doc_path).read_text(encoding="utf-8")
    bad = [f"line {line}: {tok}" for line, tok in doc_numbers(text) if tok not in known]
    return not bad, bad


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    docs = [Path(a) for a in (argv if argv else ["README.md", "MODEL_CARD.md"])]
    sources = default_sources(root)
    failed = False
    for doc in docs:
        path = doc if doc.is_absolute() else root / doc
        if not path.exists():
            print(f"FAIL: {doc} does not exist")
            failed = True
            continue
        ok, bad = check_traceability(path, sources)
        for b in bad:
            print(f"UNTRACEABLE {doc} {b}")
        print(f"{doc}: {'every number traced' if ok else f'{len(bad)} untraceable number(s)'} ({len(sources)} generated sources)")
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
