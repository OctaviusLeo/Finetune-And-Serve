"""Milestone gate checks (``python scripts/verify.py m0|m1|...``), run by scripts/verify_mN.sh.

Each ``check_mN(root)`` returns a list of failure messages (empty = pass). The
shell gates run the test commands first and then this script, so a gate exits 0
only when the tests pass and every recorded artifact holds what the milestone requires.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from fts import envdoc
from fts.settings import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HASH = re.compile(r"[0-9a-f]{40}")

GITIGNORE_ENTRIES = (".env", "*.env", "runs/", ".venv/", "__pycache__/", ".pytest_cache/", ".hypothesis/", ".coverage", "htmlcov/",
                     "*.safetensors", "*.gguf", "*.bin", "adapters/")
GITATTRIBUTES_LINES = ("* text=auto eol=lf", "*.png binary", "*.gif binary")


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.failures: list[str] = []

    def require(self, ok: bool, message: str) -> bool:
        if not ok:
            self.failures.append(message)
        return ok

    def json(self, rel: str) -> dict | None:
        p = self.root / rel
        if not self.require(p.exists(), f"{rel} is missing"):
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.failures.append(f"{rel} is not valid JSON: {exc}")
            return None

    def section(self, heading: str, doc: str = "docs/ENVIRONMENT.md") -> str:
        body = envdoc.read_section(heading, self.root / doc)
        self.require(body is not None and "Filled in" not in body, f"{doc}: section '{heading}' is missing or not filled in")
        return body or ""

    def stamped_section(self, heading: str, doc: str = "docs/ENVIRONMENT.md") -> str:
        body = self.section(heading, doc)
        self.require(bool(re.search(r"- Commit: `[0-9a-f]{40}`", body)) and "- Command: `" in body and "- Seed: `" in body,
                     f"{doc}: section '{heading}' lacks its command, seed and 40-character commit")
        return body

    def stamp(self, rel: str, payload: dict | None) -> None:
        s = (payload or {}).get("stamp", {})
        self.require(bool(s.get("command")) and isinstance(s.get("seed"), int) and bool(HASH.fullmatch(str(s.get("commit", "")))),
                     f"{rel} lacks a stamp with command, seed and 40-character commit")


def _mib(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def ledger_total(root: Path) -> tuple[float, float]:
    """(ledger total in USD by the configured prices, spend cap) from config/hosted.yaml."""
    from fts.spend import SpendLedger, prices_from_config

    cfg = load_yaml(root / "config" / "hosted.yaml")
    ledger = SpendLedger(root / cfg["ledger_path"], cap_usd=float(cfg["spend_cap_usd"]), prices=prices_from_config(cfg))
    return ledger.total_usd, ledger.cap_usd


def check_ci(c: Checker, label: str, metric: dict | None) -> None:
    ok = isinstance(metric, dict) and all(isinstance(metric.get(k), (int, float)) for k in ("point", "lower", "upper"))
    if c.require(ok, f"{label}: metric with point, lower and upper is missing"):
        c.require(0.0 <= metric["lower"] <= metric["point"] <= metric["upper"] <= 1.0, f"{label}: CI {metric} is not ordered inside [0, 1]")


def check_m0(root: Path = REPO_ROOT) -> list[str]:
    c = Checker(root)
    machine = c.stamped_section("Machine and GPU")
    total = _mib(r"GPU total VRAM reported by `nvidia-smi`: \*\*(\d+) MiB\*\*", machine)
    c.require(total is not None and "NVIDIA-SMI" in machine, "Machine and GPU: nvidia-smi capture with total MiB is missing")

    dry = c.stamped_section("Base model and VRAM Dry_Run")
    c.require("Result: **PASSED**" in dry, "Dry_Run did not pass")
    reserved = _mib(r"max_memory_reserved\(\)`: \*\*(\d+) MiB\*\*", dry)
    smi_peak = _mib(r"sampled once per second \(whole GPU\): \*\*(\d+) MiB\*\*", dry)
    dry_total = _mib(r"`nvidia-smi` total: \*\*(\d+) MiB\*\*", dry)
    if c.require(None not in (reserved, smi_peak, dry_total), "Dry_Run: peak VRAM values or total are missing"):
        c.require(reserved <= dry_total and smi_peak <= dry_total, f"Dry_Run: peak {reserved}/{smi_peak} MiB exceeds total {dry_total} MiB")
    for needle, what in (("PEFT method:", "PEFT method"), ("Sequence length:", "sequence length"), ("Per-device batch size:", "batch size"),
                         ("license Apache-2.0", "Base_Model license")):
        c.require(needle in dry, f"Dry_Run: {what} is missing")
    c.require(bool(re.search(r"at revision `[0-9a-f]{40}`", dry)), "Dry_Run: Base_Model revision is missing")
    dr = c.json("results/dry_run.json")
    if dr is not None:
        c.stamp("results/dry_run.json", dr)
        c.require(dr.get("result", {}).get("ok") is True, "results/dry_run.json: Dry_Run not ok")

    env = c.stamped_section("Python environment")
    for pkg in ("torch", "transformers", "peft", "accelerate"):
        c.require(bool(re.search(rf"\| {pkg} \| [0-9][^|]* \|", env)), f"Python environment: {pkg} version is missing")
    c.require("uv pip show" in env and "built for CUDA" in env, "Python environment: package-manager artifact or CUDA version is missing")

    hosted = c.stamped_section("Hosted model and Spend_Cap")
    c.require(bool(re.search(r"\*\*Spend_Cap: \$\d+\.\d\d\*\*", hosted)), "Hosted model section: Spend_Cap is missing")
    c.require("`model` field was" in hosted, "Hosted model section: API check result is missing")
    hc = c.json("results/hosted_check.json")
    if hc is not None:
        c.stamp("results/hosted_check.json", hc)

    spent, cap = ledger_total(root)
    c.require(spent <= cap, f"hosted ledger total ${spent:.6f} exceeds the Spend_Cap ${cap:.2f}")

    gitignore = (root / ".gitignore").read_text(encoding="utf-8").splitlines() if (root / ".gitignore").exists() else []
    for entry in GITIGNORE_ENTRIES:
        c.require(entry in gitignore, f".gitignore lacks {entry}")
    attrs = (root / ".gitattributes").read_text(encoding="utf-8").splitlines() if (root / ".gitattributes").exists() else []
    for line in GITATTRIBUTES_LINES:
        c.require(line in attrs, f".gitattributes lacks '{line}'")
    return c.failures


def check_m1(root: Path = REPO_ROOT) -> list[str]:
    c = Checker(root)
    counts = c.json("results/data_counts.json")
    if counts is not None:
        c.stamp("results/data_counts.json", counts)
        n = counts.get("counts", {})
        c.require(all(isinstance(n.get(k), int) and n[k] > 0 for k in ("train", "validation", "test")), "data_counts: split counts missing")
        c.require(isinstance(counts.get("excluded"), int), "data_counts: excluded count missing")
        c.require(counts.get("overlap", {}).get("total") == 0, "data_counts: splits overlap")
    base = c.json("results/baselines.json")
    if base is not None:
        c.stamp("results/baselines.json", base)
        records = base.get("baselines", [])
        c.require([r.get("name") for r in records] == ["base_zero_shot", "base_few_shot", "hosted_few_shot"], "baselines.json: expected three baselines")
        for r in records:
            check_ci(c, f"baseline {r.get('name')}", r.get("metric"))
            c.stamp(f"results/baselines/{r.get('name')}.json", r)
            c.require(bool(r.get("prompt_template")), f"baseline {r.get('name')}: prompt template not recorded")
        c.require(len({r.get("test_ids_sha256") for r in records}) == 1, "baselines: different Test_Splits")
        hs = base.get("hosted_spend", {})
        c.require(hs.get("within_cap") is True, f"baselines: hosted spend {hs} not within cap")
    spent, cap = ledger_total(root)
    c.require(spent <= cap, f"hosted ledger total ${spent:.6f} exceeds the Spend_Cap ${cap:.2f}")
    return c.failures


CHECKS: dict[str, Callable[[Path], list[str]]] = {"m0": check_m0, "m1": check_m1}


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description="Milestone gate checks.")
    parser.add_argument("milestone", choices=sorted(CHECKS))
    args = parser.parse_args(argv)
    failures = CHECKS[args.milestone](root)
    for f in failures:
        print(f"FAIL: {f}")
    print(f"{args.milestone}: {'PASSED' if not failures else f'{len(failures)} check(s) failed'}")
    return 0 if not failures else 1
