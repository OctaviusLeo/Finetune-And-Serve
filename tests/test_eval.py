"""Unit tests for the evaluation entry point (fts.eval) with a fake predictor."""

from __future__ import annotations

import json

import pytest

from fts import data, eval as fts_eval, repro
from fts.infer import Prediction

STAMP = repro.Stamp(command="python -m fts.eval", seed=0, commit="8" * 40)
GOLD = [
    '{"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]}',
    '{"intent": "play_music", "slots": []}',
    '{"intent": "weather_query", "slots": [{"type": "place_name", "value": "paris"}]}',
    '{"intent": "alarm_query", "slots": []}',
]
TEST = tuple(data.Example(str(i), "test", f"u{i}", g) for i, g in enumerate(GOLD))
RUNS = [{"run_id": f"B-seed{s}", "seed": s, "adapter_path": f"adapters/B-seed{s}"} for s in (1, 2, 3)]


def predictor(wrong_by_seed):
    """Seed s gets the first wrong_by_seed[s] examples wrong (invalid output)."""
    calls = []

    def predict(run, test):
        calls.append(run["run_id"])
        n = wrong_by_seed[run["seed"]]
        return [Prediction(e.example_id, "oops" if i < n else e.target, 10, 3) for i, e in enumerate(test)]

    return predict, calls


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    data.write_splits(data.Splits((), (), TEST), r / "data" / "massive_en")
    repro.write_json(r / "results" / "final_runs.json", {"runs": RUNS, "serving_seed": 2}, STAMP)
    monkeypatch.setattr(fts_eval.repro, "start_run", lambda seed: STAMP)
    return r


def test_main_writes_finetuned_and_error_analysis(root):
    predict, calls = predictor({1: 1, 2: 0, 3: 2})
    assert fts_eval.main([], root=root, predict=predict) == 0
    assert calls == ["B-seed1", "B-seed2", "B-seed3"]
    ft = json.loads((root / "results" / "finetuned.json").read_text(encoding="utf-8"))
    points = [s["metric"]["point"] for s in ft["seeds"]]
    assert points == [0.75, 1.0, 0.5] and ft["mean"] == pytest.approx(0.75) and ft["spread"] == pytest.approx(0.25)
    assert ft["serving_seed"] == 2 and ft["stamp"]["commit"] == STAMP.commit and ft["test_ids_sha256"] == fts_eval.split_digest(TEST)
    ea = json.loads((root / "results" / "error_analysis.json").read_text(encoding="utf-8"))
    assert ea["seed"] == 2 and ea["analyzed"] == 0 and sum(g["count"] for g in ea["groups"]) == 0
    assert (root / "runs" / "predictions" / "test_B-seed3.jsonl").exists()


def test_error_analysis_uses_the_serving_seed(root):
    final = json.loads((root / "results" / "final_runs.json").read_text(encoding="utf-8"))
    final["serving_seed"] = 3
    (root / "results" / "final_runs.json").write_text(json.dumps(final), encoding="utf-8")
    predict, _ = predictor({1: 0, 2: 0, 3: 2})
    assert fts_eval.main([], root=root, predict=predict) == 0
    ea = json.loads((root / "results" / "error_analysis.json").read_text(encoding="utf-8"))
    assert ea["analyzed"] == 2 and {g["group"]: g["count"] for g in ea["groups"]}["invalid_output"] == 2


def test_missing_config_fails(root, capsys):
    (root / "results" / "final_runs.json").unlink()
    assert fts_eval.main([], root=root, predict=predictor({})[0]) == 1
    assert "final_runs.json is missing" in capsys.readouterr().err


def test_rerun_identical(root, tmp_path):
    predict, _ = predictor({1: 1, 2: 1, 3: 1})
    assert fts_eval.main(["--rerun-check"], root=root, predict=predict) == 1  # no saved predictions yet
    assert fts_eval.main([], root=root, predict=predict) == 0
    env_md = tmp_path / "ENV.md"
    assert fts_eval.main(["--rerun-check"], root=root, predict=predict, env_md=env_md) == 0
    result = json.loads((root / "results" / "eval_determinism.json").read_text(encoding="utf-8"))
    assert result["status"] == "identical" and result["seed"] == 2 and result["differing_ids"] == []
    assert "**identical**" in env_md.read_text(encoding="utf-8")


def test_rerun_with_differing_predictions_is_documented(root, tmp_path):
    predict, _ = predictor({1: 0, 2: 1, 3: 0})
    assert fts_eval.main([], root=root, predict=predict) == 0
    other, _ = predictor({1: 0, 2: 2, 3: 0})
    env_md = tmp_path / "ENV.md"
    assert fts_eval.main(["--rerun-check", "--seed", "2"], root=root, predict=other, env_md=env_md) == 0
    result = json.loads((root / "results" / "eval_determinism.json").read_text(encoding="utf-8"))
    assert result["status"] == "documented_gpu_nondeterminism" and result["differing_ids"] == ["1"]
    assert result["identical_subset"]["n"] == 3 and result["identical_subset"]["first"] == result["identical_subset"]["second"]
    assert "Differing ids" in env_md.read_text(encoding="utf-8")


def test_rerun_payload_metric_mismatch_without_identical_examples():
    run = RUNS[0]
    first = [Prediction(e.example_id, e.target, 1, 1) for e in TEST]
    second = [Prediction(e.example_id, "x", 1, 1) for e in TEST]
    assert fts_eval.rerun_payload(run, TEST, first, second)["status"] == "metric_mismatch"
    assert fts_eval.compare_predictions(first, second) == ["0", "1", "2", "3"]


def test_default_predictor_loads_adapter(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(fts_eval.infer, "load_for_inference", lambda name, rev, adapter, device: loads.append(str(adapter)) or ("m", "t"))
    monkeypatch.setattr(fts_eval.infer, "predict", lambda m, t, test, build, bs, progress: [progress(len(test), len(test))] and [Prediction("0", "x", 1, 1)])
    predict = fts_eval.default_predictor(artifacts=tmp_path, device="cpu")
    assert predict(RUNS[0], TEST)[0].text == "x" and loads == [str(tmp_path / "adapters/B-seed1")]
