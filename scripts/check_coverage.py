"""Enforce the per-module coverage rule on a coverage.py JSON report.

A module in ``fts/`` or ``scripts/`` is non-trivial when it has more than one
public function or more than 50 source lines. Every non-trivial module must
reach at least 80% statement coverage. A module that coverage never saw counts
as 0% covered. Usage::

    python scripts/check_coverage.py runs/coverage.json [--root .] [--threshold 80]

Exits 0 when every non-trivial module passes, 1 otherwise (listing each one).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("fts", "scripts")
MAX_TRIVIAL_LINES = 50
THRESHOLD_PERCENT = 80.0


@dataclass(frozen=True)
class ModuleCoverage:
    path: str  # repository-relative, forward slashes
    public_functions: int
    source_lines: int
    percent: float

    @property
    def non_trivial(self) -> bool:
        return self.public_functions > 1 or self.source_lines > MAX_TRIVIAL_LINES


def count_source_lines(text: str) -> int:
    """Lines that are neither blank nor comment-only."""
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def count_public_functions(text: str) -> int:
    """Module-level public functions plus public methods of module-level public classes."""
    tree = ast.parse(text)
    count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            count += 1
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    count += 1
    return count


def statement_percent(file_entry: dict | None) -> float:
    """Statement coverage of one coverage.py JSON file entry; 0 when absent, 100 when empty."""
    if file_entry is None:
        return 0.0
    summary = file_entry["summary"]
    statements = summary["num_statements"]
    if statements == 0:
        return 100.0
    return 100.0 * summary["covered_lines"] / statements


def _normalise(path: str, root: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(root.resolve())
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def collect(report: dict, root: Path) -> list[ModuleCoverage]:
    """One ModuleCoverage per Python file under the measured packages."""
    files = {_normalise(name, root): entry for name, entry in report.get("files", {}).items()}
    modules: list[ModuleCoverage] = []
    for package in PACKAGES:
        for py in sorted((root / package).rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            rel = py.relative_to(root).as_posix()
            text = py.read_text(encoding="utf-8")
            modules.append(
                ModuleCoverage(
                    path=rel,
                    public_functions=count_public_functions(text),
                    source_lines=count_source_lines(text),
                    percent=statement_percent(files.get(rel)),
                )
            )
    return modules


def failures(modules: list[ModuleCoverage], threshold: float = THRESHOLD_PERCENT) -> list[ModuleCoverage]:
    return [m for m in modules if m.non_trivial and m.percent < threshold]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="coverage.py JSON report")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_PERCENT)
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    modules = collect(report, args.root)
    bad = failures(modules, args.threshold)
    checked = sum(1 for m in modules if m.non_trivial)
    if bad:
        print(f"coverage rule FAILED: {len(bad)} of {checked} non-trivial modules under {args.threshold:.0f}%")
        for m in bad:
            print(f"  {m.path}: {m.percent:.1f}% ({m.public_functions} public functions, {m.source_lines} source lines)")
        return 1
    print(f"coverage rule OK: {checked} non-trivial modules at or above {args.threshold:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
