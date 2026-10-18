"""Prompt templates and chat-message builders (design.md "Prompts and few-shot exemplars").

Templates live in config/prompts/ so each system's exact prompt is recorded
(Req 3.6). Placeholders are ``{utterance}``, ``{intents}`` and ``{slot_types}``,
filled by plain string replacement so the JSON braces in the templates stay literal.

* ``finetune`` - compact user message used to train, evaluate and serve the Finetuned_Model.
* ``zero_shot`` - instruction, output format, every intent and slot type (Train_Split only), utterance.
* ``few_shot`` - the zero-shot preamble in the first user turn, then 10 exemplars as prior
  user/assistant turns (later user turns use ``few_shot_turn``), then the request.

Every prompt starts with the same explicit system message, so a model's default
system message never appears.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fts.data import Example

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"
TEMPLATE_FILES = {
    "system": "system.txt",
    "finetune": "finetune.txt",
    "zero_shot": "zero_shot.txt",
    "few_shot": "few_shot.txt",
    "few_shot_turn": "few_shot_turn.txt",
}


@dataclass(frozen=True)
class Templates:
    system: str
    finetune: str
    zero_shot: str
    few_shot: str
    few_shot_turn: str


def load_templates(directory: Path | str = PROMPTS_DIR) -> Templates:
    d = Path(directory)
    return Templates(**{k: (d / f).read_text(encoding="utf-8").strip("\n") for k, f in TEMPLATE_FILES.items()})


def template_path(name: str) -> str:
    """Repository-relative path of a template, as recorded in results."""
    return f"config/prompts/{TEMPLATE_FILES[name]}"


def fill(template: str, utterance: str, labels: dict[str, list[str]] | None = None) -> str:
    text = template.replace("{utterance}", utterance)
    if labels is not None:
        text = text.replace("{intents}", ", ".join(labels["intents"])).replace("{slot_types}", ", ".join(labels["slot_types"]))
    return text


def _system(t: Templates) -> dict:
    return {"role": "system", "content": t.system}


def finetune_messages(t: Templates, utterance: str) -> list[dict]:
    return [_system(t), {"role": "user", "content": fill(t.finetune, utterance)}]


def zero_shot_messages(t: Templates, utterance: str, labels: dict[str, list[str]]) -> list[dict]:
    return [_system(t), {"role": "user", "content": fill(t.zero_shot, utterance, labels)}]


def few_shot_messages(t: Templates, utterance: str, labels: dict[str, list[str]], exemplars: Sequence[Example]) -> list[dict]:
    """System, then the exemplars as prior turns (preamble in the first), then the request."""
    if not exemplars:
        return zero_shot_messages(t, utterance, labels)
    messages = [_system(t)]
    for i, ex in enumerate(exemplars):
        user = fill(t.few_shot, ex.inputs, labels) if i == 0 else fill(t.few_shot_turn, ex.inputs)
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": ex.target}]
    messages.append({"role": "user", "content": fill(t.few_shot_turn, utterance)})
    return messages


def training_messages(t: Templates, example: Example) -> list[dict]:
    """The compact prompt plus the gold JSON as the assistant turn."""
    return finetune_messages(t, example.inputs) + [{"role": "assistant", "content": example.target}]


def load_labels(path: Path | str) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"intents": list(data["intents"]), "slot_types": list(data["slot_types"])}
