"""Unit tests for fts.exemplars and scripts/select_exemplars.py."""

from __future__ import annotations

import json

from fts import data, exemplars
from fts.data import Example
from scripts import select_exemplars


def ex(i, intent, partition="train"):
    return Example(str(i), partition, f"utt {i}", json.dumps({"intent": intent, "slots": []}))


TRAIN = [ex(i, intent) for i, intent in enumerate(["alarm_set", "alarm_query", "iot_cleaning", "play_music", "qa_factoid", "iot_coffee", "weather_query"])]


def test_scenario_is_intent_prefix():
    assert exemplars.scenario_of(ex(1, "iot_hue_lightchange")) == "iot"
    assert exemplars.scenario_of(ex(1, "general_quirky")) == "general"


def test_selection_is_seeded_distinct_and_from_train():
    chosen = exemplars.select_exemplars(TRAIN, k=4, seed=0)
    assert len(chosen) == 4
    assert len({exemplars.scenario_of(e) for e in chosen}) == 4
    assert all(e in TRAIN for e in chosen)
    assert chosen == exemplars.select_exemplars(TRAIN, k=4, seed=0)
    assert exemplars.select_exemplars(TRAIN, k=10, seed=0) == exemplars.select_exemplars(TRAIN, k=5, seed=0)  # only 5 scenarios


def test_save_and_load_round_trip(tmp_path):
    chosen = exemplars.select_exemplars(TRAIN, k=3, seed=1)
    path = exemplars.save_exemplars(chosen, tmp_path / "ex.json", seed=1)
    assert exemplars.load_exemplars(path) == chosen
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 1


def test_repository_exemplars_are_ten_train_examples():
    chosen = exemplars.load_exemplars()
    train_ids = {e.example_id for e in data.read_split(exemplars.REPO_ROOT / "data" / "massive_en" / "train.jsonl")}
    assert len(chosen) == 10 and {e.example_id for e in chosen} <= train_ids
    assert len({exemplars.scenario_of(e) for e in chosen}) == 10


def test_script_writes_file(tmp_path, capsys):
    d = tmp_path / "massive_en"
    data.write_splits(data.Splits(tuple(TRAIN), (), ()), d)
    out = tmp_path / "ex.json"
    assert select_exemplars.main(["--k", "3"], data_dir=d, out=out) == 0
    assert len(exemplars.load_exemplars(out)) == 3 and "wrote 3 exemplars" in capsys.readouterr().out
    assert select_exemplars.main(["--k", "9"], data_dir=d, out=out) == 1  # fewer distinct scenarios than asked
