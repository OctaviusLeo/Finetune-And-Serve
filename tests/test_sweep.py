"""Unit tests for sweep selection and the seed-count guard (task 3.4; Req 4.3-4.5)."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest

from fts import data, repro, settings, sweep
from fts.train import RunRecord

STAMP = repro.Stamp(command="python -m fts.sweep sweep", seed=1, commit="6" * 40)
BASE = settings.load_train_settings()
CONFIGS, SEEDS = sweep.load_sweep_configs(BASE)
SPLITS = data.Splits((), (), ())


def record(config, val):
    return RunRecord(run_id=f"{config.name}-seed{config.seed}", adapter_path=f"adapters/{config.name}-seed{config.seed}",
                     loss_curve=((10, 1.0),), config=config, val_metric=val, commit=STAMP.commit)


def fake_train(metrics):
    calls = []

    def train_fn(config):
        calls.append((config.name, config.seed))
        return record(config, metrics[(config.name, config.seed)] if (config.name, config.seed) in metrics else metrics[config.name])

    return train_fn, calls


def test_repository_sweep_config_matches_design():
    assert [(c.name, c.lora_r, c.lora_alpha, c.learning_rate) for c in CONFIGS] == [("A", 8, 16, 2e-4), ("B", 16, 32, 2e-4), ("C", 16, 32, 1e-4)]
    assert {c.seed for c in CONFIGS} == {1} and SEEDS == [1, 2, 3]


def test_sweep_selects_highest_validation_metric():
    train_fn, calls = fake_train({"A": 0.80, "B": 0.86, "C": 0.84})
    result, records = sweep.run_sweep(SPLITS, CONFIGS, train_fn)
    assert result.selected.name == "B" and result.val_metrics == {"A": 0.80, "B": 0.86, "C": 0.84}
    assert calls == [("A", 1), ("B", 1), ("C", 1)] and result.rationale_path == "docs/SWEEP_RATIONALE.md"
    payload = sweep.sweep_payload(result, records)
    assert payload["selected"]["name"] == "B" and payload["selected_val_metric"] == 0.86 and payload["runs"]["A"]["record"] == "results/runs/A-seed1.json"


def test_ties_go_to_the_earlier_config():
    assert sweep.select_final(CONFIGS, {"A": 0.85, "B": 0.85, "C": 0.85})[0].name == "A"
    assert sweep.select_final(CONFIGS, {"A": 0.80, "B": 0.85, "C": 0.85})[0].name == "B"


def test_sweep_requires_three_configs_and_one_seed():
    with pytest.raises(ValueError):
        sweep.select_final(CONFIGS[:2], {"A": 1.0, "B": 0.5})
    with pytest.raises(ValueError):
        sweep.run_sweep(SPLITS, CONFIGS[:2], lambda c: None)
    with pytest.raises(ValueError):
        sweep.run_sweep(SPLITS, [CONFIGS[0], CONFIGS[1], replace(CONFIGS[2], seed=2)], lambda c: None)


def test_final_seeds_reuse_seed_one_and_enforce_three_distinct_seeds():
    final = CONFIGS[1]
    train_fn, calls = fake_train({("B", 2): 0.87, ("B", 3): 0.85})
    seed1 = record(final, 0.86)
    records = sweep.run_final_seeds(SPLITS, final, [1, 2, 3], train_fn, existing={1: seed1})
    assert calls == [("B", 2), ("B", 3)] and records[0] is seed1
    assert sweep.serving_seed(records) == 2
    payload = sweep.final_payload(final, records)
    assert payload["seeds"] == [1, 2, 3] and payload["serving_seed"] == 2 and payload["runs"][1]["adapter_path"] == "adapters/B-seed2"
    for bad in ([1, 2], [1, 1, 2], [2, 2, 2, 3]):
        with pytest.raises(ValueError):
            sweep.run_final_seeds(SPLITS, final, bad, train_fn)


def test_serving_seed_ties_go_to_the_lowest_seed():
    recs = [record(replace(CONFIGS[1], seed=s), v) for s, v in ((3, 0.9), (1, 0.9), (2, 0.8))]
    assert sweep.serving_seed(recs) == 1


def test_reusable_record_requires_same_config_and_adapter(tmp_path):
    cfg = CONFIGS[0]
    results, artifacts = tmp_path / "runs", tmp_path / "art"
    assert sweep.reusable_record(cfg, results, artifacts) is None
    repro.write_json(results / "A-seed1.json", record(cfg, 0.8).to_dict(), STAMP)
    assert sweep.reusable_record(cfg, results, artifacts) is None  # adapter missing
    (artifacts / "adapters" / "A-seed1").mkdir(parents=True)
    (artifacts / "adapters" / "A-seed1" / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert sweep.reusable_record(cfg, results, artifacts).val_metric == 0.8
    assert sweep.reusable_record(replace(cfg, learning_rate=1e-3), results, artifacts) is None


class FakeTrainer:
    def __init__(self, metrics):
        self.metrics, self.calls = metrics, []

    def train(self, splits, config, stamp):
        self.calls.append((config.name, config.seed))
        return record(config, self.metrics.get((config.name, config.seed), 0.5))


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    (r / "docs").mkdir()
    for f in ("train.yaml", "sweep.yaml", "model.yaml"):
        shutil.copy(sweep.REPO_ROOT / "config" / f, r / "config" / f)
    shutil.copy(sweep.REPO_ROOT / "docs" / "SWEEP_RATIONALE.md", r / "docs" / "SWEEP_RATIONALE.md")
    data.write_splits(SPLITS, r / "data" / "massive_en")
    monkeypatch.setattr(sweep.repro, "start_run", lambda seed: STAMP)
    return r


def test_main_sweep_then_final(root, tmp_path):
    trainer = FakeTrainer({("A", 1): 0.7, ("B", 1): 0.9, ("C", 1): 0.8, ("B", 2): 0.95, ("B", 3): 0.9})
    assert sweep.main(["final"], root=root, trainer=trainer, artifacts=tmp_path / "art") == 1  # no sweep yet
    assert sweep.main(["sweep"], root=root, trainer=trainer, artifacts=tmp_path / "art") == 0
    saved = json.loads((root / "results" / "sweep.json").read_text(encoding="utf-8"))
    assert saved["selected"]["name"] == "B" and saved["stamp"]["commit"] == STAMP.commit
    # make the seed-1 sweep Run reusable: record plus Adapter
    repro.write_json(root / "results" / "runs" / "B-seed1.json", record(CONFIGS[1], 0.9).to_dict(), STAMP)
    (tmp_path / "art" / "adapters" / "B-seed1").mkdir(parents=True)
    (tmp_path / "art" / "adapters" / "B-seed1" / "adapter_config.json").write_text("{}", encoding="utf-8")
    trainer.calls.clear()
    assert sweep.main(["final"], root=root, trainer=trainer, artifacts=tmp_path / "art") == 0
    assert trainer.calls == [("B", 2), ("B", 3)]
    final = json.loads((root / "results" / "final_runs.json").read_text(encoding="utf-8"))
    assert final["serving_seed"] == 2 and final["seeds"] == [1, 2, 3]


def test_main_requires_rationale(root, tmp_path, capsys):
    (root / "docs" / "SWEEP_RATIONALE.md").unlink()
    assert sweep.main(["sweep"], root=root, trainer=FakeTrainer({}), artifacts=tmp_path / "art") == 1
    assert "SWEEP_RATIONALE.md must be written" in capsys.readouterr().err


def test_make_train_fn_reuses_finished_runs(tmp_path):
    cfg = CONFIGS[2]
    results, artifacts = tmp_path / "runs", tmp_path / "art"
    trainer = FakeTrainer({("C", 1): 0.6})
    logs = []
    fn = sweep.make_train_fn(trainer, SPLITS, STAMP, results, artifacts, log=logs.append)
    assert fn(cfg).val_metric == 0.6 and trainer.calls == [("C", 1)]
    repro.write_json(results / "C-seed1.json", record(cfg, 0.61).to_dict(), STAMP)
    (artifacts / "adapters" / "C-seed1").mkdir(parents=True)
    (artifacts / "adapters" / "C-seed1" / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert fn(cfg).val_metric == 0.61 and trainer.calls == [("C", 1)] and "reusing" in logs[0]
