"""Unit tests for fts.envrecord (versions and licenses)."""

from __future__ import annotations

from email.message import Message
from importlib import metadata

from fts import envrecord, repro, settings

STAMP = repro.Stamp(command="python scripts/record_env.py", seed=0, commit="1" * 40)


class FakeDist:
    def __init__(self, version="1.0", **headers):
        self.version = version
        msg = Message()
        for key, value in headers.items():
            if isinstance(value, list):
                for v in value:
                    msg[key.replace("_", "-")] = v
            else:
                msg[key.replace("_", "-")] = value
        self.metadata = msg


def test_license_of_prefers_expression_then_field_then_classifiers():
    assert envrecord.license_of(FakeDist(License_Expression="MIT", License="BSD")) == "MIT"
    assert envrecord.license_of(FakeDist(License="BSD-3-Clause")) == "BSD-3-Clause"
    assert envrecord.license_of(FakeDist(License="UNKNOWN", Classifier=["License :: OSI Approved :: MIT License"])) == "MIT License"
    long_text = "Copyright (c) someone\nPermission is hereby granted"
    assert envrecord.license_of(FakeDist(License=long_text)) == "Copyright (c) someone"
    assert envrecord.license_of(FakeDist()) == "not declared in package metadata"


def test_license_rows_handle_missing_packages():
    def lookup(name):
        if name == "gone":
            raise metadata.PackageNotFoundError(name)
        return FakeDist(version="2.0", License_Expression="Apache-2.0")

    rows = envrecord.license_rows([("present", "2.0"), ("gone", "0.1")], lookup)
    assert rows == [("present", "2.0", "Apache-2.0"), ("gone", "0.1", "not installed in this environment")]


def test_locked_and_direct_dependencies_from_repository():
    locked = dict(envrecord.locked_packages())
    direct = dict(envrecord.direct_dependencies())
    assert direct["torch"] == "2.14.1" and direct["transformers"] == "5.20.1"
    assert locked["torch"] == "2.14.1" and "finetune-and-serve" not in locked
    assert len(locked) == len(envrecord.locked_packages())  # one row per name


def test_render_tables():
    table = envrecord.render_license_table([("torch", "2.14.1", "BSD|x"), ("six", "1.0", "MIT")], {"torch"})
    assert "| **torch** | 2.14.1 | BSD/x |" in table and "| six | 1.0 | MIT |" in table
    base = envrecord.render_base_model(settings.ModelSettings("Org/M", "a" * 40, "Apache-2.0", "bfloat16"))
    assert "| Org/M | " + "a" * 40 + " | Apache-2.0 | https://huggingface.co/Org/M |" in base


def test_uv_pip_show_uses_runner():
    seen = []
    out = envrecord.uv_pip_show(["torch"], runner=lambda cmd: seen.append(cmd) or "Name: torch\nVersion: 2.14.1\n")
    assert "Version: 2.14.1" in out and seen[0][:3] == ["uv", "pip", "show"] and seen[0][-1] == "torch"


def test_torch_info_reports_versions():
    info = envrecord.torch_info()
    assert info["torch"].startswith("2.14.1") and isinstance(info["cuda_available"], bool)


def test_render_environment_contains_artifact():
    text = envrecord.render_environment(
        [("torch", "2.14.1", "BSD")], "Name: torch\nVersion: 2.14.1", {"torch": "2.14.1+cu130", "cuda_runtime": "13.0", "cudnn": 1, "cuda_available": True, "device": "GPU"},
        "uv 0.13.1", "3.12.3", STAMP,
    )
    assert "torch `2.14.1+cu130` built for CUDA `13.0`" in text and "Version: 2.14.1" in text and STAMP.commit in text


def test_main_writes_both_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(envrecord.repro, "start_run", lambda seed: STAMP)
    monkeypatch.setattr(envrecord, "uv_pip_show", lambda names: "Name: torch\nVersion: 2.14.1\n")

    class Done:
        stdout = "uv 0.13.1\n"

    monkeypatch.setattr(envrecord.subprocess, "run", lambda *a, **k: Done())
    env_md, tp = tmp_path / "ENV.md", tmp_path / "TP.md"
    assert envrecord.main(env_md=env_md, third_party_md=tp) == 0
    assert "## Python environment" in env_md.read_text(encoding="utf-8")
    tp_text = tp.read_text(encoding="utf-8")
    assert "## Python packages" in tp_text and "| **torch** |" in tp_text and "## Base model" in tp_text
