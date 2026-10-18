"""Unit tests for Run recording and Test isolation (task 3.2; Req 4.2, 4.7), on the CPU with a tiny random model."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from fts import prompts, repro, settings, train
from fts.data import Example, Splits
from fts.lora import tiny_qwen2
from tests.fake_tokenizer import FakeChatTokenizer

STAMP = repro.Stamp(command="python -m fts.sweep", seed=1, commit="5" * 40)
T = prompts.load_templates()
TS = replace(settings.load_train_settings(), logging_steps=2, gradient_checkpointing=False)
GOLD = '{"intent": "alarm_set", "slots": []}'


def ex(i, partition):
    return Example(str(i), partition, f"wake me {i}", GOLD)


class GuardedSplits:
    """Splits whose Test_Split raises if anything reads it."""

    def __init__(self, train, validation):
        self.train, self.validation = train, validation

    @property
    def test(self):
        raise AssertionError("the trainer read the Test_Split")


def make_trainer(tmp_path, **kw):
    return train.PEFTTrainer(
        settings.ModelSettings("tiny", None, "Apache-2.0", "float32"), TS, T, device="cpu", artifacts=tmp_path / "artifacts",
        results_dir=tmp_path / "results" / "runs", predictions_dir=tmp_path / "pred", model_factory=lambda ms, d: tiny_qwen2(vocab_size=128, seed=3),
        tokenizer=FakeChatTokenizer(), val_batch=4, log=lambda m: None, **kw,
    )


def config(seed=1, epochs=2.0, batch=2, accum=2):
    return train.TrainConfig(name="A", lora_r=4, lora_alpha=8, learning_rate=1e-3, epochs=epochs, max_seq_len=256,
                             per_device_batch_size=batch, grad_accum=accum, seed=seed)


def test_train_records_run_and_never_reads_test(tmp_path):
    splits = GuardedSplits(tuple(ex(i, "train") for i in range(10)), tuple(ex(100 + i, "dev") for i in range(3)))
    record = make_trainer(tmp_path).train(splits, config(), STAMP)
    assert record.run_id == "A-seed1" and record.commit == STAMP.commit and len(record.commit) == 40
    assert record.config == config() and record.config.seed == 1
    # 10 examples / batch 2 = 5 micro-batches per epoch, 2 epochs = 10, accumulation 2 -> 5 steps, logged every 2 and at the end
    assert [s for s, _ in record.loss_curve] == [2, 4, 5] and all(v > 0 for _, v in record.loss_curve)
    assert 0.0 <= record.val_metric <= 1.0
    assert (tmp_path / "artifacts" / "adapters" / "A-seed1" / "adapter_config.json").exists()
    saved = json.loads((tmp_path / "results" / "runs" / "A-seed1.json").read_text(encoding="utf-8"))
    assert saved["stamp"]["commit"] == STAMP.commit and saved["config"]["lora_r"] == 4 and saved["splits_used"] == ["train", "validation"]
    assert saved["loss_curve"][0][0] == 2 and saved["validation_examples"] == 3 and saved["optimizer_steps"] == 5
    assert "test" not in json.dumps(saved["splits_used"])
    assert train.record_from_dict(saved) == record
    assert (tmp_path / "pred" / "val_A-seed1.jsonl").exists()


def test_training_splits_accessor_uses_only_train_and_validation():
    splits = GuardedSplits((ex(1, "train"),), (ex(2, "dev"),))
    tr, va = train.PEFTTrainer.training_splits(splits)
    assert tr[0].example_id == "1" and va[0].example_id == "2"


def test_same_seed_gives_the_same_loss_curve(tmp_path):
    splits = Splits(tuple(ex(i, "train") for i in range(8)), (ex(50, "dev"),), ())
    a = make_trainer(tmp_path / "a").train(splits, config(seed=2), STAMP)
    b = make_trainer(tmp_path / "b").train(splits, config(seed=2), STAMP)
    assert [s for s, _ in a.loss_curve] == [s for s, _ in b.loss_curve]
    assert all(abs(x - y) <= 1e-6 for (_, x), (_, y) in zip(a.loss_curve, b.loss_curve))


def test_encode_example_masks_the_prompt():
    tok = FakeChatTokenizer()
    ids, labels = train.encode_example(tok, T, ex(1, "train"), 1000)
    prompt = tok.apply_chat_template(prompts.finetune_messages(T, "wake me 1"), add_generation_prompt=True)
    n_prompt = len(tok(prompt)["input_ids"])
    assert labels[:n_prompt] == [train.IGNORE] * n_prompt and labels[n_prompt:] == ids[n_prompt:]
    assert len(ids) == len(labels) > n_prompt
    short_ids, short_labels = train.encode_example(tok, T, ex(1, "train"), 10)
    assert len(short_ids) == len(short_labels) == 10


def test_encode_example_rejects_inconsistent_template():
    class Odd(FakeChatTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return ("X" if add_generation_prompt else "Y") + super().apply_chat_template(messages, tokenize, add_generation_prompt)

    with pytest.raises(ValueError):
        train.encode_example(Odd(), T, ex(1, "train"), 100)


def test_counts_order_and_collate():
    assert train.micro_count(10, 3, 2.0) == 8 and train.micro_count(10, 2, 1.5) == 8
    assert train.optimizer_steps(10, config(batch=3, accum=2)) == 4
    assert train.epoch_order(5, 1, 0) == train.epoch_order(5, 1, 0) != train.epoch_order(5, 1, 1)
    batches = list(train.micro_batches([([i], [i]) for i in range(10)], 2, 0, 1.5))
    assert len(batches) == 8 and {e for e, _ in batches} == {0, 1}
    c = train.collate([([1, 2, 3], [-100, 2, 3]), ([4], [4])], pad_id=0)
    assert c["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]] and c["labels"].tolist()[1] == [4, -100, -100]
    assert c["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_max_steps_stops_early_and_partial_accumulation():
    model = tiny_qwen2(vocab_size=128, seed=0)
    from fts.lora import LoraSpec, apply_lora

    model = apply_lora(model, LoraSpec(4, 8, 0.0, ("q_proj",)), gradient_checkpointing=False)
    encoded = [train.encode_example(FakeChatTokenizer(), T, ex(i, "train"), 64) for i in range(6)]
    curve = train.train_loop(model, encoded, config(epochs=1.0, batch=2, accum=2), TS, "cpu", 0, log=lambda m: None, max_steps=1)
    assert curve == [] or curve[-1][0] == 1
    # 3 micro-batches with accumulation 2 -> steps of 2 and 1 micro-batch
    curve = train.train_loop(model, encoded, config(epochs=1.0, batch=2, accum=2), replace(TS, logging_steps=1), "cpu", 0, log=lambda m: None)
    assert [s for s, _ in curve] == [1, 2]


def test_config_helpers_and_run_id():
    c = train.config_from(settings.load_train_settings(), "B", 16, 32, 2e-4, seed=1)
    assert c.max_seq_len == 256 and c.per_device_batch_size == 4 and c.grad_accum == 4 and c.epochs == 2.0
    assert train.run_id(c) == "B-seed1" and train.with_seed(c, 3).seed == 3


def test_load_run_and_default_factory(tmp_path, monkeypatch):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert train.load_run(p) == {"a": 1}
    import fts.lora

    monkeypatch.setattr(fts.lora, "load_base_model", lambda name, rev, dtype, device: ("loaded", name, device))
    assert train._default_model_factory(settings.load_model_settings(), "cpu") == ("loaded", "Qwen/Qwen2.5-1.5B-Instruct", "cpu")


def test_tokenizer_loaded_lazily(monkeypatch, tmp_path):
    import fts.lora

    monkeypatch.setattr(fts.lora, "load_tokenizer", lambda name, rev: FakeChatTokenizer())
    trainer = make_trainer(tmp_path)
    trainer.tokenizer = None
    assert isinstance(trainer._tokenizer(), FakeChatTokenizer)
