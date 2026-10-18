"""Unit tests for fts.infer on a tiny random Qwen2 model with a fake tokenizer (CPU, no download)."""

from __future__ import annotations

import json

import torch

from fts import infer
from fts.data import Example
from fts.lora import tiny_qwen2

GOLD = '{"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]}'


class CharTok:
    """Character-level stand-in for a Hugging Face tokenizer (ids 3..102; pad 0, eos 1)."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join(m["content"] for m in messages) + (">" if add_generation_prompt else "")

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        ids = [[3 + (ord(c) % 100) for c in t] for t in texts]
        width = max(len(x) for x in ids)
        pad = lambda x: [0] * (width - len(x)) + x if self.padding_side == "left" else x + [0] * (width - len(x))  # noqa: E731
        mask = lambda x: [0] * (width - len(x)) + [1] * len(x) if self.padding_side == "left" else [1] * len(x) + [0] * (width - len(x))  # noqa: E731
        return {"input_ids": torch.tensor([pad(x) for x in ids]), "attention_mask": torch.tensor([mask(x) for x in ids])}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)


def ex(i, utt, target=GOLD):
    return Example(example_id=str(i), partition="test", inputs=utt, target=target)


def model():
    m = tiny_qwen2(vocab_size=128, seed=1)
    m.eval()
    return m


def test_stop_token_ids():
    assert infer.stop_token_ids(1, [1, 2], None, 0) == {0, 1, 2}


def test_generate_batch_left_pads_and_counts_tokens():
    tok = CharTok()
    out = infer.generate_batch(model(), tok, ["ab", "abcdef"], max_new_tokens=5)
    assert tok.padding_side == "left"
    assert [o[1] for o in out] == [2, 6]
    assert all(1 <= o[2] <= 5 and isinstance(o[0], str) for o in out)


def test_generation_stops_at_eos():
    class EosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = type("G", (), {"eos_token_id": 7})()

        def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
            tail = torch.tensor([[5, 7, 9, 9]] * input_ids.shape[0])
            return torch.cat([input_ids, tail], dim=1)

    (text, prompt_tokens, completion_tokens), = infer.generate_batch(EosModel(), CharTok(), ["xyz"], max_new_tokens=4)
    assert completion_tokens == 2 and text == "f" and prompt_tokens == 3


def test_predict_is_deterministic_and_in_order():
    tok, m = CharTok(), model()
    exs = [ex(i, f"utt {i}") for i in range(5)]
    build = lambda e: [{"role": "user", "content": e.inputs}]  # noqa: E731
    seen = []
    first = infer.predict(m, tok, exs, build, batch_size=2, max_new_tokens=4, progress=lambda d, n: seen.append((d, n)))
    second = infer.predict(m, tok, exs, build, batch_size=2, max_new_tokens=4)
    assert [p.example_id for p in first] == ["0", "1", "2", "3", "4"]
    assert first == second and seen == [(2, 5), (4, 5), (5, 5)]


def test_score_reports_not_scored_and_continues():
    exs = [ex(1, "a"), ex(2, "b", target="broken gold"), ex(3, "c")]
    preds = [infer.Prediction("1", GOLD, 10, 5), infer.Prediction("2", GOLD, 10, 5)]
    report = infer.score(exs, preds)
    assert report.scored_ids == ["1"] and report.scores == [1.0]
    assert report.not_scored == [
        {"id": "2", "cause": "gold target is missing or not valid task JSON"},
        {"id": "3", "cause": "no prediction"},
    ]
    assert report.per_example == {"1": 1.0}


def test_save_and_load_predictions(tmp_path):
    exs = [ex(1, "a"), ex(2, "b", target="bad")]
    preds = [infer.Prediction("1", '{"intent": "alarm_set", "slots": []}', 10, 3), infer.Prediction("2", "x", 9, 1)]
    path = infer.save_predictions(tmp_path / "p" / "run.jsonl", exs, preds)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["f1"] == 2 / 3 and rows[0]["gold"] == GOLD and rows[1]["f1"] is None
    assert infer.load_predictions(path) == preds


def test_load_for_inference_applies_adapter(monkeypatch):
    import peft

    import fts.lora

    m = model()
    monkeypatch.setattr(fts.lora, "load_base_model", lambda name, rev, dtype, device: m)
    monkeypatch.setattr(fts.lora, "load_tokenizer", lambda name, rev: CharTok())
    calls = []

    class FakePeft:
        @staticmethod
        def from_pretrained(base, path):
            calls.append(path)
            return base

    monkeypatch.setattr(peft, "PeftModel", FakePeft)
    loaded, tok = infer.load_for_inference("Org/M", "rev", adapter_path="/adapters/run1", device="cpu")
    assert calls == ["/adapters/run1"] and loaded.config.use_cache is True and not loaded.training
    loaded2, _ = infer.load_for_inference("Org/M", "rev", device="cpu")
    assert calls == ["/adapters/run1"]
