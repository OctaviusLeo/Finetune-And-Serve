"""Unit tests for fts.lora on a tiny random Qwen2 model (CPU, no download)."""

from __future__ import annotations

import os

import pytest
import torch

from fts import lora

SPEC = lora.LoraSpec(r=4, alpha=8, dropout=0.0, target_modules=("q_proj", "v_proj"))


def test_set_determinism_seeds_and_sets_cublas(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    lora.set_determinism(5)
    a = torch.rand(3)
    lora.set_determinism(5)
    assert torch.equal(a, torch.rand(3))
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == lora.CUBLAS_WORKSPACE
    assert torch.are_deterministic_algorithms_enabled()


def test_torch_dtype_mapping():
    assert lora.torch_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError):
        lora.torch_dtype("int3")


def test_apply_lora_freezes_base_and_trains_adapter():
    model = lora.apply_lora(lora.tiny_qwen2(), SPEC, gradient_checkpointing=True)
    trainable, total = lora.count_parameters(model)
    assert 0 < trainable < total
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    ids = torch.randint(0, 128, (2, 8))
    model.train()
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_load_helpers_call_transformers(monkeypatch):
    import transformers

    calls = {}

    class FakeTok:
        pad_token = None
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, name, revision=None):
            calls["tok"] = (name, revision)
            return cls()

    class FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

    class FakeAuto:
        @classmethod
        def from_pretrained(cls, name, revision=None, dtype=None):
            calls["model"] = (name, revision, dtype)
            return FakeModel()

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTok)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAuto)
    tok = lora.load_tokenizer("Org/M", "rev")
    assert tok.pad_token == "<eos>" and calls["tok"] == ("Org/M", "rev")
    lora.load_base_model("Org/M", "rev", "bfloat16", "cpu")
    assert calls["model"] == ("Org/M", "rev", torch.bfloat16) and calls["device"] == "cpu"
