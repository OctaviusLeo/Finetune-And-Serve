"""Unit tests for the milestone gate checks (fts.verify), using the real section renderers."""

from __future__ import annotations

import json
import shutil

import pytest

from fts import dry_run, envdoc, envrecord, hosted, repro, settings, verify
from scripts import verify as verify_script

STAMP = repro.Stamp(command="python x.py", seed=0, commit="4" * 40)
SMI = "| NVIDIA-SMI 615.78 |\n| 0 RTX 3080 | 2410MiB / 10240MiB |\n"


def write_m0_root(root, *, passed=True, peak=9000, spent_tokens=1000):
    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(verify.REPO_ROOT / "config" / "hosted.yaml", root / "config" / "hosted.yaml")
    shutil.copy(verify.REPO_ROOT / ".gitignore", root / ".gitignore")
    shutil.copy(verify.REPO_ROOT / ".gitattributes", root / ".gitattributes")
    env = root / "docs" / "ENVIRONMENT.md"
    cfg = dry_run.config_from_settings(settings.ModelSettings("Qwen/Qwen2.5-1.5B-Instruct", "a" * 40, "Apache-2.0", "bfloat16"), settings.load_train_settings())
    result = dry_run.DryRunResult(ok=passed, failure=None if passed else "VRAM over budget", peak_reserved_mib=peak, peak_smi_mib=peak, total_mib=10240,
                                  trainable_params=1, total_params=2, idle_mib=2410)
    envdoc.replace_section("Machine and GPU", dry_run.render_machine_section(SMI, STAMP), env)
    envdoc.replace_section("Base model and VRAM Dry_Run", dry_run.render_dry_run_section(cfg, result, STAMP), env)
    rows = [(p, "1.0", "Apache-2.0") for p in ("torch", "transformers", "peft", "accelerate")]
    envdoc.replace_section("Python environment", envrecord.render_environment(
        rows, "Name: torch", {"torch": "2.14.1", "cuda_runtime": "13.0", "cudnn": 1, "cuda_available": True}, "uv 0.13", "3.12", STAMP), env)
    hcfg = settings.load_yaml(root / "config" / "hosted.yaml")
    check = {"response_model": "gpt-6-luna-x", "text": "OK", "accepted_params": ["seed"], "dropped_params": [], "prompt_tokens": 30,
             "completion_tokens": 2, "cost_usd": 1e-5}
    summary = {"prompt_price_per_mtok": 0.125, "completion_price_per_mtok": 0.5, "spend_usd": 1e-5, "requests": 1}
    envdoc.replace_section("Hosted model and Spend_Cap", hosted.render_section(hcfg, check, summary, STAMP), env)
    repro.write_json(root / "results" / "dry_run.json", dry_run.result_payload(cfg, result), STAMP)
    repro.write_json(root / "results" / "hosted_check.json", {"check": check}, STAMP)
    (root / "results" / "hosted_ledger.jsonl").write_text(json.dumps({"prompt_tokens": spent_tokens, "completion_tokens": 0}) + "\n", encoding="utf-8")
    return root


def test_m0_passes_on_complete_evidence(tmp_path):
    assert verify.check_m0(write_m0_root(tmp_path)) == []


def test_m0_flags_failed_dry_run_over_cap_ledger_and_git_files(tmp_path):
    root = write_m0_root(tmp_path, passed=False, peak=11000, spent_tokens=10**8)
    (root / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    failures = "\n".join(verify.check_m0(root))
    assert "Dry_Run did not pass" in failures and "exceeds total 10240" in failures
    assert "exceeds the Spend_Cap" in failures and "'* text=auto eol=lf'" in failures and "results/dry_run.json: Dry_Run not ok" in failures


def test_m0_flags_missing_sections_and_files(tmp_path):
    root = write_m0_root(tmp_path)
    envdoc.replace_section("Hosted model and Spend_Cap", "Filled in at task 1.6.", root / "docs" / "ENVIRONMENT.md")
    (root / "results" / "hosted_check.json").unlink()
    (root / "results" / "dry_run.json").write_text("{not json", encoding="utf-8")
    failures = "\n".join(verify.check_m0(root))
    assert "'Hosted model and Spend_Cap' is missing or not filled in" in failures
    assert "results/hosted_check.json is missing" in failures and "results/dry_run.json is not valid JSON" in failures


def baseline(name, point=0.5, digest="d"):
    return {"name": name, "metric": {"point": point, "lower": point - 0.01, "upper": point + 0.01}, "prompt_template": "config/prompts/x.txt",
            "test_ids_sha256": digest, "stamp": repro.to_dict(STAMP)}


def write_m1_root(root, *, records=None, within_cap=True):
    write_m0_root(root)
    repro.write_json(root / "results" / "data_counts.json", {"counts": {"train": 10, "validation": 2, "test": 3}, "excluded": 0, "overlap": {"total": 0}}, STAMP)
    records = records or [baseline(n) for n in ("base_zero_shot", "base_few_shot", "hosted_few_shot")]
    repro.write_json(root / "results" / "baselines.json", {"baselines": records, "hosted_spend": {"within_cap": within_cap}}, STAMP)
    return root


def test_m1_passes(tmp_path):
    assert verify.check_m1(write_m1_root(tmp_path)) == []


def test_m1_flags_bad_ci_split_mismatch_and_spend(tmp_path):
    records = [baseline("base_zero_shot", point=1.2), baseline("base_few_shot", digest="other"), baseline("hosted_few_shot")]
    failures = "\n".join(verify.check_m1(write_m1_root(tmp_path, records=records, within_cap=False)))
    assert "not ordered inside [0, 1]" in failures and "different Test_Splits" in failures and "not within cap" in failures


def test_m1_flags_missing_stamp_and_counts(tmp_path):
    root = write_m1_root(tmp_path)
    repro.write_json(root / "results" / "data_counts.json", {"counts": {"train": 0}, "overlap": {"total": 2}}, STAMP)
    data = json.loads((root / "results" / "baselines.json").read_text(encoding="utf-8"))
    del data["stamp"]
    data["baselines"][0]["metric"] = None
    (root / "results" / "baselines.json").write_text(json.dumps(data), encoding="utf-8")
    failures = "\n".join(verify.check_m1(root))
    assert "split counts missing" in failures and "excluded count missing" in failures and "splits overlap" in failures
    assert "results/baselines.json lacks a stamp" in failures and "metric with point, lower and upper is missing" in failures


def test_main_exit_codes(tmp_path, capsys):
    root = write_m0_root(tmp_path)
    assert verify.main(["m0"], root=root) == 0 and "m0: PASSED" in capsys.readouterr().out
    (root / "results" / "hosted_check.json").unlink()
    assert verify.main(["m0"], root=root) == 1 and "FAIL:" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        verify.main(["m9"], root=root)


def test_script_entry_point_imports():
    assert verify_script.main is verify.main
