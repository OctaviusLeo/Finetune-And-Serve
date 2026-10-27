"""Milestone gate checks (``python scripts/verify.py m0|m1|...``), run by scripts/verify_mN.sh.

Each ``check_mN(root)`` returns a list of failure messages (empty = pass). The
shell gates run the test commands first and then this script, so a gate exits 0
only when the tests pass and every recorded artifact holds what the milestone requires.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

from fts import envdoc
from fts.settings import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HASH = re.compile(r"[0-9a-f]{40}")

GITIGNORE_ENTRIES = (".env", "*.env", "/runs/", ".venv/", "__pycache__/", ".pytest_cache/", ".hypothesis/", ".coverage", "htmlcov/",
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


def default_artifacts() -> Path:
    import os

    return Path(os.environ.get("FTS_ARTIFACTS", Path.home() / "fts-artifacts"))


def check_m2(root: Path = REPO_ROOT, artifacts: Path | None = None) -> list[str]:
    artifacts = artifacts or default_artifacts()
    c = Checker(root)
    sweep = c.json("results/sweep.json")
    if sweep is not None:
        c.stamp("results/sweep.json", sweep)
        metrics = sweep.get("val_metrics", {})
        c.require(len(sweep.get("configs", [])) >= 3 and len(metrics) >= 3 and all(isinstance(v, float) for v in metrics.values()),
                  "sweep.json: needs three configs with Validation F1")
        c.require(sweep.get("selected", {}).get("name") in metrics, "sweep.json: no selected Final_Configuration")
        if metrics and sweep.get("selected", {}).get("name") in metrics:
            c.require(metrics[sweep["selected"]["name"]] == max(metrics.values()), "sweep.json: selection is not the best Validation F1")
        c.require((root / str(sweep.get("rationale_path", ""))).is_file(), "sweep rationale file is missing")
    final = c.json("results/final_runs.json")
    if final is not None:
        c.stamp("results/final_runs.json", final)
        runs = final.get("runs", [])
        c.require(len({r.get("seed") for r in runs}) >= 3, "final_runs.json: needs three distinct seeds")
        for r in runs:
            rid = r.get("run_id")
            c.require((artifacts / str(r.get("adapter_path")) / "adapter_config.json").is_file(), f"{rid}: Adapter missing under {artifacts}")
            c.require(len(r.get("loss_curve", [])) > 0, f"{rid}: loss curve missing")
            c.require(isinstance(r.get("config"), dict) and r["config"].get("seed") == r.get("seed"), f"{rid}: config or seed missing")
            c.require(bool(HASH.fullmatch(str(r.get("commit", "")))), f"{rid}: 40-character commit missing")
        c.require(final.get("serving_seed") in {r.get("seed") for r in runs}, "final_runs.json: serving seed missing")
    det = c.json("results/train_determinism.json")
    if det is not None:
        c.stamp("results/train_determinism.json", det)
        c.require(det.get("status") == "identical" or (det.get("status") == "documented" and det.get("nondeterministic_kernels")),
                  f"train_determinism: status {det.get('status')} is neither identical nor documented with a named kernel")
    runs_dir = root / "results" / "runs"
    records = sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []
    c.require(bool(records), "results/runs/ has no Run records")
    for p in records:
        text = p.read_text(encoding="utf-8")
        rec = json.loads(text)
        c.require(rec.get("splits_used") == ["train", "validation"] and "test.jsonl" not in text and '"test"' not in text,
                  f"{p.name} references the Test_Split")
    return c.failures


def check_m3(root: Path = REPO_ROOT) -> list[str]:
    c = Checker(root)
    ft = c.json("results/finetuned.json")
    base = c.json("results/baselines.json")
    if ft is not None:
        c.stamp("results/finetuned.json", ft)
        seeds = ft.get("seeds", [])
        c.require(len({s.get("seed") for s in seeds}) >= 3, "finetuned.json: needs three seeds")
        for s in seeds:
            check_ci(c, f"finetuned seed {s.get('seed')}", s.get("metric"))
        c.require(isinstance(ft.get("mean"), float) and isinstance(ft.get("spread"), float), "finetuned.json: mean or spread missing")
        if base is not None:
            c.require(ft.get("test_ids_sha256") == base.get("test_ids_sha256") and all(s.get("test_ids_sha256") == base.get("test_ids_sha256") for s in seeds),
                      "finetuned.json Test ids differ from baselines.json")
    ea = c.json("results/error_analysis.json")
    if ea is not None:
        c.stamp("results/error_analysis.json", ea)
        total = sum(g.get("count", 0) for g in ea.get("groups", []))
        c.require(total == ea.get("analyzed"), f"error analysis: group counts sum to {total}, not {ea.get('analyzed')}")
        c.require(ea.get("analyzed", 0) >= min(50, ea.get("available_failures", 0)), "error analysis: fewer than 50 failures analyzed")
    return c.failures


def check_m4(root: Path = REPO_ROOT) -> list[str]:
    c = Checker(root)
    q = c.json("results/quantization.json")
    if q is not None:
        c.stamp("results/quantization.json", q)
        got = {(m.get("method"), m.get("bit_width")) for m in q.get("models", [])}
        c.require(got == {("gguf-f16", 16), ("gguf-q4_k_m", 4)}, f"quantization.json: expected F16 (16) and Q4_K_M (4), got {sorted(got)}")
    qq = c.json("results/quant_quality.json")
    if qq is not None:
        c.stamp("results/quant_quality.json", qq)
        variants = {v.get("method"): v for v in qq.get("variants", [])}
        for method in ("gguf-f16", "gguf-q4_k_m"):
            v = variants.get(method)
            if c.require(v is not None, f"quant_quality.json: {method} missing"):
                check_ci(c, f"quant_quality {method}", v.get("metric"))
                vram = v.get("vram", {})
                c.require(vram.get("total_mib", 0) > 0 and vram.get("peak_mib", 10**9) <= vram.get("total_mib", 0),
                          f"quant_quality {method}: peak serving VRAM {vram.get('peak_mib')} MiB exceeds total {vram.get('total_mib')} MiB")
    lt = c.json("results/loadtest.json")
    if lt is not None:
        c.stamp("results/loadtest.json", lt)
        levels = lt.get("levels", [])
        c.require(len(levels) > 0, "loadtest.json: no levels")
        for lv in levels:
            n = lv.get("concurrency_level")
            c.require(lv.get("p50_latency_ms", 1) <= lv.get("p95_latency_ms", 0), f"loadtest level {n}: P50 > P95")
            c.require(isinstance(lv.get("tokens_per_second"), (int, float)) and isinstance(lv.get("failed_requests"), int), f"loadtest level {n}: tokens/s or failed count missing")
            s = lv.get("stamp") or {}
            c.require(s.get("concurrency") == n and bool(HASH.fullmatch(str(s.get("commit", "")))) and bool(s.get("command")) and isinstance(s.get("seed"), int),
                      f"loadtest level {n}: stamp missing")
        c.require(isinstance(lt.get("max_concurrent_requests"), int), "loadtest.json: maximum sustained concurrency missing")
        vram = lt.get("vram", {})
        c.require(vram.get("total_mib", 0) > 0 and vram.get("peak_mib", 10**9) <= vram.get("total_mib", 0),
                  f"loadtest: peak serving VRAM {vram.get('peak_mib')} MiB exceeds total {vram.get('total_mib')} MiB")
    return c.failures




MODEL_PATTERNS = ("*.safetensors", "*.gguf", "*.bin", "*.pt")
SKIP_DIRS = {".git", "runs", ".venv", "__pycache__", ".pytest_cache", ".hypothesis"}


def model_files(root: Path) -> list[str]:
    """Model-weight files anywhere in the working tree outside ignored run folders (none may be tracked)."""
    import fnmatch
    import os

    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = Path(dirpath).relative_to(root)
        if rel.parts[:1] == ("adapters",):
            found.append(rel.as_posix())
            continue
        found += [(rel / f).as_posix() for f in filenames if any(fnmatch.fnmatch(f, p) for p in MODEL_PATTERNS)]
    return sorted(found)


def check_m5(root: Path = REPO_ROOT) -> list[str]:
    from fts.report import PLOTS, SECTIONS

    c = Checker(root)
    summary_path = root / "results" / "summary.md"
    if c.require(summary_path.exists(), "results/summary.md is missing"):
        text = summary_path.read_text(encoding="utf-8")
        c.require("_Missing" not in text, "results/summary.md has missing sources")
        for title in SECTIONS:
            body = envdoc.read_section(title, summary_path)
            if c.require(body is not None, f"summary.md: section '{title}' missing"):
                c.require(bool(re.search(r"commit `[0-9a-f]{40}`", body)), f"summary.md: section '{title}' lacks its stamp")
        test_section = envdoc.read_section(SECTIONS[1], summary_path) or ""
        c.require(bool(re.search(r"\d\.\d{4} \[\d\.\d{4}, \d\.\d{4}\]", test_section)), "summary.md: Test F1 section has no CIs")
        quant_section = envdoc.read_section(SECTIONS[3], summary_path) or ""
        c.require(len(re.findall(r"\d\.\d{4} \[\d\.\d{4}, \d\.\d{4}\]", quant_section)) >= 3, "summary.md: quantization section lacks CIs")
    for plot in PLOTS:
        c.require((root / "results" / "plots" / plot).is_file(), f"results/plots/{plot} is missing")
    for doc in ("README.md", "MODEL_CARD.md"):
        c.require((root / doc).is_file(), f"{doc} is missing")
    spent, cap = ledger_total(root)
    c.require(spent <= cap, f"hosted ledger total ${spent:.6f} exceeds the Spend_Cap ${cap:.2f}")
    files = model_files(root)
    c.require(not files, f"model files in the repository tree: {files}")
    return c.failures


CHECKS: dict[str, Callable[[Path], list[str]]] = {"m0": check_m0, "m1": check_m1, "m2": check_m2, "m3": check_m3, "m4": check_m4, "m5": check_m5}


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description="Milestone gate checks.")
    parser.add_argument("milestone", choices=sorted(CHECKS))
    args = parser.parse_args(argv)
    failures = CHECKS[args.milestone](root)
    for f in failures:
        print(f"FAIL: {f}")
    print(f"{args.milestone}: {'PASSED' if not failures else f'{len(failures)} check(s) failed'}")
    return 0 if not failures else 1
