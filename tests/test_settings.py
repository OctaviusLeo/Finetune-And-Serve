"""Unit tests for fts.settings (config loading and revision pinning)."""

from __future__ import annotations

import pytest

from fts import settings

REV = "0123456789abcdef0123456789abcdef01234567"


def _model_yaml(tmp_path, revision="null"):
    p = tmp_path / "model.yaml"
    p.write_text(
        "# comment kept\nname: Org/Model\nrevision: " + revision + "\nlicense: Apache-2.0\ndtype: bfloat16\n",
        encoding="utf-8",
    )
    return p


def test_repository_configs_load():
    model = settings.load_model_settings()
    train = settings.load_train_settings()
    assert model.name == "Qwen/Qwen2.5-1.5B-Instruct" and model.license == "Apache-2.0"
    assert train.max_seq_len == 256 and train.per_device_batch_size == 4 and train.grad_accum == 4
    assert train.dry_run.lora_r == 16 and train.dry_run.steps == 20
    assert "q_proj" in train.target_modules


def test_pin_revision_resolves_once_and_keeps_comments(tmp_path):
    p = _model_yaml(tmp_path)
    calls = []

    def resolver(name):
        calls.append(name)
        return REV

    pinned = settings.pin_revision(p, resolver)
    assert pinned.revision == REV and calls == ["Org/Model"]
    assert "# comment kept" in p.read_text(encoding="utf-8")
    again = settings.pin_revision(p, resolver)
    assert again.revision == REV and calls == ["Org/Model"]


def test_pin_revision_rejects_bad_hashes(tmp_path):
    with pytest.raises(ValueError):
        settings.pin_revision(_model_yaml(tmp_path), lambda name: "main")
    with pytest.raises(ValueError):
        settings.pin_revision(_model_yaml(tmp_path, revision="v1.0"), lambda name: REV)


def test_pin_revision_requires_one_revision_line(tmp_path):
    p = tmp_path / "model.yaml"
    p.write_text("name: Org/Model\nlicense: MIT\ndtype: bfloat16\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        settings.pin_revision(p, lambda name: REV)


def test_load_yaml_rejects_non_mapping(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        settings.load_yaml(p)


def test_hub_revision_uses_model_info(monkeypatch):
    import huggingface_hub

    class FakeApi:
        def model_info(self, name):
            return type("Info", (), {"sha": REV})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert settings.hub_revision("Org/Model") == REV
