"""Unit tests for fts.prompts (templates and chat messages)."""

from __future__ import annotations

import json

from fts import prompts
from fts.data import Example

LABELS = {"intents": ["alarm_set", "play_music"], "slot_types": ["date", "time"]}
T = prompts.load_templates()


def ex(i, utt, target):
    return Example(example_id=str(i), partition="train", inputs=utt, target=target)


def test_templates_load_and_have_placeholders():
    assert "{utterance}" in T.finetune and "{utterance}" in T.zero_shot and "{utterance}" in T.few_shot_turn
    assert "{intents}" in T.zero_shot and "{slot_types}" in T.few_shot
    assert prompts.template_path("few_shot") == "config/prompts/few_shot.txt"


def test_fill_keeps_json_braces_literal():
    text = prompts.fill(T.zero_shot, "wake me up", LABELS)
    assert '{"intent": "<intent>"' in text and "Intents: alarm_set, play_music" in text
    assert "Slot types: date, time" in text and text.endswith("Request: wake me up")


def test_finetune_and_training_messages():
    msgs = prompts.finetune_messages(T, "hi there")
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"] == T.system
    assert msgs[1]["content"].endswith("Request: hi there")
    train = prompts.training_messages(T, ex(1, "hi there", '{"intent": "x", "slots": []}'))
    assert train[-1] == {"role": "assistant", "content": '{"intent": "x", "slots": []}'}


def test_zero_shot_messages():
    msgs = prompts.zero_shot_messages(T, "u", LABELS)
    assert len(msgs) == 2 and "Intents:" in msgs[1]["content"]


def test_few_shot_messages_structure():
    exemplars = [ex(1, "first one", '{"intent": "a", "slots": []}'), ex(2, "second one", '{"intent": "b", "slots": []}')]
    msgs = prompts.few_shot_messages(T, "the request", LABELS, exemplars)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "Intents:" in msgs[1]["content"] and msgs[1]["content"].endswith("Request: first one")
    assert msgs[3]["content"] == "Request: second one" and msgs[4]["content"] == exemplars[1].target
    assert msgs[-1]["content"] == "Request: the request"
    assert sum("Intents:" in m["content"] for m in msgs) == 1


def test_few_shot_without_exemplars_is_zero_shot():
    assert prompts.few_shot_messages(T, "u", LABELS, []) == prompts.zero_shot_messages(T, "u", LABELS)


def test_load_labels(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(LABELS), encoding="utf-8")
    assert prompts.load_labels(p) == LABELS
