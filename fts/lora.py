"""Shared model plumbing for the Dry_Run, training and inference.

Loading the Base_Model in bf16, wrapping it with a LoRA Adapter, and the
determinism settings live here so the Dry_Run measures exactly the model setup
the trainer uses. torch, transformers and peft are imported inside functions so
importing this module stays cheap.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

CUBLAS_WORKSPACE = ":4096:8"


@dataclass(frozen=True)
class LoraSpec:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


def set_determinism(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels (warn-only).

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first cuBLAS call, so call
    this before any model touches the GPU.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def limit_cuda_memory(fraction: float | None) -> bool:
    """Cap this process's CUDA caching allocator at ``fraction`` of device memory; True when a cap was set.

    Under WSL2 an allocation beyond dedicated VRAM spills into shared system
    memory (slow) instead of failing, so without a cap the allocator's cache can
    grow past the card when batch shapes vary. With the cap it frees cached
    blocks first, and only a genuine need above the cap raises out-of-memory.
    """
    import torch

    if not fraction or not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(fraction)
    return True


def torch_dtype(name: str) -> Any:
    import torch

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in dtypes:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(dtypes)}")
    return dtypes[name]


def load_tokenizer(name: str, revision: str | None) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(name: str, revision: str | None, dtype: str = "bfloat16", device: str = "cuda") -> Any:
    """The Base_Model in ``dtype`` on ``device`` with the KV cache off (training default)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch_dtype(dtype))
    return model.to(device)


def apply_lora(model: Any, spec: LoraSpec, gradient_checkpointing: bool) -> Any:
    """Freeze the base weights and attach a trainable LoRA Adapter."""
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def count_parameters(model: Any) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def tiny_qwen2(vocab_size: int = 128, seed: int = 0) -> Any:
    """A randomly initialised two-layer Qwen2 model for CPU tests (no download)."""
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    return AutoModelForCausalLM.from_config(cfg)
