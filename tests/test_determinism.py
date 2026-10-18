"""Unit tests for the training determinism check (fts.determinism)."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

from fts import data, determinism, prompts, repro, settings, train
from fts.lora import tiny_qwen2
from tests.fake_tokenizer import FakeChatTokenizer

STAMP = repro.Stamp(command="python scripts/check_train_determinism.py", seed=1, commit="7" * 40)
GOLD = '{"intent": "alarm_set", "slots": []}'
SPLITS = data.Splits(tuple(data.Example(str(i), "train", f"wake me {i}", GOLD) for i in range(8)), (), ())
CONFIG = train.TrainConfig("B", 4, 8, 1e-3, 2.0, 256, 2, 2, 1)


def make_trainer(tmp_path):
    ts = replace(settings.load_train_settings(), gradient_checkpointing=False)
    return train.PEFTTrainer(settings.ModelSettings("tiny", None, "Apache-2.0", "float32"), ts, prompts.load_templates(), device="cpu",
                             artifacts=tmp_path, results_dir=tmp_path, model_factory=lambda ms, d: tiny_qwen2(vocab_size=128, seed=3),
                             tokenizer=FakeChatTokenizer(), log=lambda m: None)


def test_compare_and_classify():
    ok, diff = determinism.compare([1.0, 2.0], [1.0, 2.0 + 5e-7])
    assert ok and abs(diff - 5e-7) < 1e-12
    ok, diff = determinism.compare([1.0, 2.0], [1.0, 2.1])
    assert not ok and abs(diff - 0.1) < 1e-12
    assert determinism.compare([1.0], [1.0, 2.0]) == (False, float("inf"))
    assert determinism.classify(True, []) == "identical"
    assert determinism.classify(False, ["Flash Attention defaults to a non-deterministic algorithm."]) == "documented"
    assert determinism.classify(False, []) == "unexplained"


def test_nondeterminism_messages_filters_and_dedupes():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("Flash Attention defaults to a non-deterministic algorithm.\nmore text", UserWarning)
        warnings.warn("Flash Attention defaults to a non-deterministic algorithm.\nmore text", UserWarning)
        warnings.warn("index_add_cuda_ does not have a deterministic implementation", UserWarning)
        warnings.warn("something unrelated", UserWarning)
        warnings.warn("", UserWarning)
    assert determinism.nondeterminism_messages(caught) == [
        "Flash Attention defaults to a non-deterministic algorithm.",
        "index_add_cuda_ does not have a deterministic implementation",
    ]


def test_check_on_cpu_is_identical(tmp_path):
    result = determinism.check(make_trainer(tmp_path), SPLITS, CONFIG, steps=3, log=lambda m: None)
    assert result["status"] == "identical" and result["equal"] and len(result["run1_losses"]) == 3
    assert result["run1_losses"] == result["run2_losses"] and result["max_abs_diff"] == 0.0


def test_render_section_mismatch_lists_kernels_and_values():
    result = {"config": {"name": "B", "seed": 1}, "steps": 2, "tolerance": 1e-6, "run1_losses": [1.0, 0.9], "run2_losses": [1.0, 0.9001],
              "max_abs_diff": 1e-4, "equal": False, "nondeterministic_kernels": ["Flash Attention defaults to a non-deterministic algorithm."],
              "status": "documented"}
    text = determinism.render_section(result, STAMP)
    assert "**documented**" in text and "Flash Attention" in text and "0.900000/0.900100" in text and STAMP.commit in text


class FakeTrainer:
    def __init__(self, runs):
        self.runs = list(runs)
        self.ts = settings.load_train_settings()

    def training_splits(self, splits):
        return splits.train, splits.validation


def test_main_paths(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    data.write_splits(SPLITS, root / "data" / "massive_en")
    env_md = tmp_path / "ENV.md"
    monkeypatch.setattr(determinism.repro, "start_run", lambda seed: STAMP)
    assert determinism.main(root=root, trainer=FakeTrainer([]), env_md=env_md) == 1  # no sweep.json
    repro.write_json(root / "results" / "sweep.json", {"selected": CONFIG.__dict__}, STAMP)

    outcomes = iter([([1.0, 0.5], []), ([1.0, 0.6], [])])
    monkeypatch.setattr(determinism, "run_once", lambda trainer, ex, cfg, steps: next(outcomes))
    assert determinism.main(root=root, trainer=FakeTrainer([]), env_md=env_md) == 1  # unexplained mismatch
    saved = json.loads((root / "results" / "train_determinism.json").read_text(encoding="utf-8"))
    assert saved["status"] == "unexplained" and saved["stamp"]["commit"] == STAMP.commit

    outcomes = iter([([1.0, 0.5], ["Flash Attention defaults to a non-deterministic algorithm."]), ([1.0, 0.6], [])])
    monkeypatch.setattr(determinism, "run_once", lambda trainer, ex, cfg, steps: next(outcomes))
    assert determinism.main(root=root, trainer=FakeTrainer([]), env_md=env_md) == 0
    assert "**documented**" in env_md.read_text(encoding="utf-8")
