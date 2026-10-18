"""Load the project's YAML configuration and pin the Base_Model revision.

``config/model.yaml`` names the Base_Model; its ``revision`` starts as ``null``
and is resolved from the Hugging Face Hub on first use, then written back so
every later run loads exactly the same weights. ``config/train.yaml`` holds the
fixed LoRA settings shared by the Dry_Run and every training Run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
MODEL_YAML = CONFIG_DIR / "model.yaml"
TRAIN_YAML = CONFIG_DIR / "train.yaml"

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModelSettings:
    name: str
    revision: str | None
    license: str
    dtype: str


@dataclass(frozen=True)
class DryRunSettings:
    steps: int
    lora_r: int
    lora_alpha: int
    learning_rate: float
    seed: int


@dataclass(frozen=True)
class TrainSettings:
    peft_method: str
    max_seq_len: int
    per_device_batch_size: int
    grad_accum: int
    epochs: float
    warmup_ratio: float
    lr_scheduler: str
    lora_dropout: float
    target_modules: tuple[str, ...]
    gradient_checkpointing: bool
    logging_steps: int
    dry_run: DryRunSettings


def load_yaml(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def load_model_settings(path: Path | str = MODEL_YAML) -> ModelSettings:
    d = load_yaml(path)
    return ModelSettings(name=d["name"], revision=d.get("revision"), license=d["license"], dtype=d["dtype"])


def load_train_settings(path: Path | str = TRAIN_YAML) -> TrainSettings:
    d = load_yaml(path)
    dr = d["dry_run"]
    return TrainSettings(
        peft_method=d["peft_method"],
        max_seq_len=int(d["max_seq_len"]),
        per_device_batch_size=int(d["per_device_batch_size"]),
        grad_accum=int(d["grad_accum"]),
        epochs=float(d["epochs"]),
        warmup_ratio=float(d["warmup_ratio"]),
        lr_scheduler=d["lr_scheduler"],
        lora_dropout=float(d["lora_dropout"]),
        target_modules=tuple(d["target_modules"]),
        gradient_checkpointing=bool(d["gradient_checkpointing"]),
        logging_steps=int(d["logging_steps"]),
        dry_run=DryRunSettings(
            steps=int(dr["steps"]),
            lora_r=int(dr["lora_r"]),
            lora_alpha=int(dr["lora_alpha"]),
            learning_rate=float(dr["learning_rate"]),
            seed=int(dr["seed"]),
        ),
    )


def hub_revision(name: str) -> str:
    """The current commit hash of a Hugging Face model repository."""
    from huggingface_hub import HfApi

    return HfApi().model_info(name).sha


def pin_revision(
    path: Path | str = MODEL_YAML, resolver: Callable[[str], str] = hub_revision
) -> ModelSettings:
    """Return the settings with a pinned revision, resolving and writing it once if it is null.

    Only the ``revision:`` line is rewritten, so the file's comments survive.
    """
    settings = load_model_settings(path)
    if settings.revision:
        if not _REVISION_RE.match(settings.revision):
            raise ValueError(f"{path}: revision {settings.revision!r} is not a 40-character commit hash")
        return settings
    revision = resolver(settings.name)
    if not _REVISION_RE.match(revision or ""):
        raise ValueError(f"resolved revision {revision!r} for {settings.name} is not a 40-character hash")
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    new_text, n = re.subn(r"(?m)^revision:.*$", f"revision: {revision}", text)
    if n != 1:
        raise ValueError(f"{path} must contain exactly one 'revision:' line")
    p.write_text(new_text, encoding="utf-8")
    return load_model_settings(p)
