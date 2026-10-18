"""Unit tests for the quantizer records and Docker commands, with Docker faked (task 5.2; Req 6.1)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from fts import quantize, repro, settings

STAMP = repro.Stamp(command="python -m fts.quantize", seed=2, commit="a" * 40)
SERVE_CFG = settings.load_yaml(quantize.REPO_ROOT / "config" / "serve.yaml")
RUN = {"run_id": "B-seed2", "seed": 2, "adapter_path": "adapters/B-seed2"}
QUANT_LOG = """main: build = 11580
llama_model_loader: file type = F16 (16.00 BPW)
print_info: file size   = 2.88 GiB (16.00 BPW)
llama_model_quantize_impl: model size  =  2944.66 MB
llama_model_quantize_impl: quant size  =  1065.65 MB
main: quantize time = 12345.67 ms
"""


class FakeDocker:
    def __init__(self, artifacts: Path, fail_on: str | None = None):
        self.artifacts, self.calls, self.fail_on = artifacts, [], fail_on

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        if self.fail_on and self.fail_on in cmd:
            raise subprocess.CalledProcessError(1, list(cmd), output="boom")
        models = self.artifacts / "models"
        if "--convert" in cmd:
            (models / "ft-f16.gguf").write_bytes(b"x" * 3000)
            return "converted\n"
        if "--quantize" in cmd:
            (models / "ft-q4_k_m.gguf").write_bytes(b"x" * 1000)
            return QUANT_LOG
        if cmd[:3] == ["docker", "image", "inspect"]:
            return "ghcr.io/ggml-org/llama.cpp@sha256:" + "f" * 64 + "\n"
        raise AssertionError(cmd)


def fake_merge(model_settings, adapter_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    return 1500


def test_quantized_model_records_method_and_bit_width(tmp_path):
    docker = FakeDocker(tmp_path)
    models, details = quantize.quantize(settings.load_model_settings(), RUN, SERVE_CFG, tmp_path, docker, fake_merge, uid=1000, gid=1000, log=lambda m: None)
    assert models == [quantize.QuantizedModel("models/ft-f16.gguf", "gguf-f16", 16), quantize.QuantizedModel("models/ft-q4_k_m.gguf", "gguf-q4_k_m", 4)]
    assert details["files"]["gguf-q4_k_m"]["size_bytes"] == 1000 and details["files"]["gguf-f16"]["bits_per_weight_from_size"] == 3000 * 8 / 1500
    assert details["image_digest"].startswith("ghcr.io/ggml-org/llama.cpp@sha256:") and details["merged_parameters"] == 1500
    assert details["llama_quantize"]["quant_size_mb"] == 1065.65 and details["llama_quantize"]["input_bpw"] == 16.0


def test_docker_commands_follow_the_design(tmp_path):
    image = SERVE_CFG["full_image"]
    assert image == "ghcr.io/ggml-org/llama.cpp:full-b11580"
    conv = quantize.convert_command(image, tmp_path / "models", 1000, 1001)
    assert conv[:6] == ["docker", "run", "--rm", "--user", "1000:1001", "-e"]
    assert f"{tmp_path / 'models'}:/models" in conv and conv[conv.index(image) + 1] == "--convert"
    assert conv[-6:] == ["--convert", "/models/merged", "--outtype", "f16", "--outfile", "/models/ft-f16.gguf"]
    quant = quantize.quantize_command(image, tmp_path / "models", 1000, 1001)
    assert quant[-4:] == ["--quantize", "/models/ft-f16.gguf", "/models/ft-q4_k_m.gguf", "Q4_K_M"]


def test_parse_quantize_log_handles_missing_values():
    assert quantize.parse_quantize_log("nothing here") == {"model_size_mb": None, "quant_size_mb": None, "input_bpw": None, "reported_bpw": []}
    parsed = quantize.parse_quantize_log(QUANT_LOG)
    assert parsed["model_size_mb"] == 2944.66 and parsed["reported_bpw"] == [16.0, 16.0]


def test_run_command_returns_output_and_raises():
    assert "hi" in quantize.run_command(["bash", "-c", "echo hi"])
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        quantize.run_command(["bash", "-c", "echo bad >&2; exit 3"])
    assert excinfo.value.returncode == 3 and "bad" in excinfo.value.output


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    for f in ("serve.yaml", "model.yaml"):
        shutil.copy(quantize.REPO_ROOT / "config" / f, r / "config" / f)
    monkeypatch.setattr(quantize.repro, "start_run", lambda seed: STAMP)
    return r


def test_main_writes_quantization_json(root, tmp_path):
    assert quantize.main(root=root, artifacts=tmp_path / "art", runner=FakeDocker(tmp_path / "art"), merge=fake_merge) == 1  # no final_runs
    repro.write_json(root / "results" / "final_runs.json", {"runs": [RUN], "serving_seed": 2}, STAMP)
    assert quantize.main(root=root, artifacts=tmp_path / "art", runner=FakeDocker(tmp_path / "art"), merge=fake_merge) == 0
    q = json.loads((root / "results" / "quantization.json").read_text(encoding="utf-8"))
    assert [m["method"] for m in q["models"]] == ["gguf-f16", "gguf-q4_k_m"] and [m["bit_width"] for m in q["models"]] == [16, 4]
    assert q["stamp"]["commit"] == STAMP.commit and q["serving_run"] == "B-seed2"


def test_main_reports_docker_failure(root, tmp_path, capsys):
    repro.write_json(root / "results" / "final_runs.json", {"runs": [RUN], "serving_seed": 2}, STAMP)
    assert quantize.main(root=root, artifacts=tmp_path / "art", runner=FakeDocker(tmp_path / "art", fail_on="--quantize"), merge=fake_merge) == 1
    assert "exited 1" in capsys.readouterr().err


def test_merge_adapter_merges_and_saves(tmp_path, monkeypatch):
    import peft

    import fts.lora

    saved = {}

    class Merged:
        def parameters(self):
            import torch

            return [torch.zeros(3, 4), torch.zeros(5)]

        def save_pretrained(self, path, safe_serialization=True):
            saved["model"] = path

    class Wrapped:
        def merge_and_unload(self):
            return Merged()

    class FakePeft:
        @staticmethod
        def from_pretrained(base, path):
            saved["adapter"] = path
            return Wrapped()

    def save_pretrained_writes(path, safe_serialization=True):
        saved["model"] = path
        Path(path, "chat_template.jinja").write_text("t", encoding="utf-8")
        Path(path, "tokenizer_config.json").write_text("transformers-5 format", encoding="utf-8")

    Merged.save_pretrained = staticmethod(lambda path, safe_serialization=True: save_pretrained_writes(path))
    snap = tmp_path / "snapshot"
    snap.mkdir()
    for name in quantize.SNAPSHOT_FILES:
        (snap / name).write_text(f"original {name}", encoding="utf-8")
    monkeypatch.setattr(peft, "PeftModel", FakePeft)
    monkeypatch.setattr(fts.lora, "load_base_model", lambda name, rev, dtype, device: "base")
    n = quantize.merge_adapter(settings.load_model_settings(), tmp_path / "adapter", tmp_path / "merged", snapshot=lambda ms: snap)
    assert n == 17 and saved == {"adapter": str(tmp_path / "adapter"), "model": str(tmp_path / "merged")}
    for name in quantize.SNAPSHOT_FILES:
        assert (tmp_path / "merged" / name).read_text(encoding="utf-8") == f"original {name}"
    assert not (tmp_path / "merged" / "chat_template.jinja").exists()


def test_base_snapshot_uses_the_pinned_revision(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda name, revision, allow_patterns: calls.append((name, revision, allow_patterns)) or str(tmp_path))
    ms = settings.load_model_settings()
    assert quantize.base_snapshot(ms) == tmp_path
    assert calls == [(ms.name, ms.revision, list(quantize.SNAPSHOT_FILES))]
