"""Unit tests for fts.data (MASSIVE transform, official splits, overlap report; Req 2.3-2.6)."""

from __future__ import annotations

import json

import pytest

from fts import data

RAW = {
    "id": "1",
    "locale": "en-US",
    "partition": "train",
    "scenario": "alarm",
    "intent": "alarm_set",
    "utt": "wake me up at five am this week",
    "annot_utt": "wake me up at [time : five am] [date : this week]",
}


def raw(**changes):
    r = dict(RAW)
    r.update(changes)
    return r


def ex(i, partition, utt="u", target='{"intent": "x", "slots": []}'):
    return data.Example(example_id=str(i), partition=partition, inputs=utt, target=target)


def test_transform_builds_canonical_target():
    e = data.transform_example(RAW, seed=0)
    assert e == data.Example(
        example_id="1",
        partition="train",
        inputs="wake me up at five am this week",
        target='{"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}, {"type": "date", "value": "this week"}]}',
    )
    assert json.loads(e.target)["slots"][1] == {"type": "date", "value": "this week"}


def test_transform_keeps_non_ascii_and_colons_in_values():
    e = data.transform_example(raw(utt="play café at 5:00 pm", annot_utt="play [place_name : café] at [time : 5:00 pm]"), 0)
    assert '"café"' in e.target and '"5:00 pm"' in e.target


def test_transform_without_slots():
    e = data.transform_example(raw(utt="olly quiet", annot_utt="olly quiet", intent="audio_volume_mute"), 0)
    assert e.target == '{"intent": "audio_volume_mute", "slots": []}'


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"annot_utt": "wake me up at [time : five am this week"}, "unbalanced brackets"),
        ({"annot_utt": "wake me up at time : five am] this week"}, "unbalanced brackets"),
        ({"annot_utt": "wake [me [time : up]] at five am this week"}, "unbalanced brackets"),
        ({"annot_utt": "wake me up at [time five am] this week"}, "slot span lacks ' : '"),
        ({"annot_utt": "wake me up at [time : ] five am this week"}, "empty slot label or value"),
        ({"intent": "   "}, "empty intent"),
        ({"utt": "wake me up at six am this week"}, "annot_utt without markup does not reproduce utt"),
        ({"partition": "holdout"}, "unknown partition 'holdout'"),
        ({"intent": None}, "missing field intent"),
    ],
)
def test_malformed_records_are_excluded_with_reason(changes, reason):
    example, why = data.transform_with_reason(raw(**changes))
    assert example is None and why == reason
    assert data.transform_example(raw(**changes), seed=0) is None


def test_whitespace_differences_are_collapsed():
    e = data.transform_example(raw(utt="wake  me up at five am this week ", annot_utt="wake me  up at [time : five am]  [date : this week]"), 0)
    assert e is not None and e.inputs == "wake me up at five am this week"


def test_transform_all_reports_ids_reasons_and_count():
    report = data.transform_all([RAW, raw(id="2", intent=""), "not a dict", raw(id="", annot_utt="[x]")], seed=0)
    assert [e.example_id for e in report.examples] == ["1"]
    assert report.exclusions == [
        data.Exclusion("2", "empty intent"),
        data.Exclusion("<line 3>", "record is not a JSON object"),
        data.Exclusion("<line 4>", "slot span lacks ' : '"),
    ]
    assert report.excluded_count == 3


def test_build_splits_uses_official_partitions_in_order():
    exs = [ex(1, "test"), ex(2, "train"), ex(3, "dev"), ex(4, "train")]
    s = data.build_splits(exs, seed=0)
    assert [e.example_id for e in s.train] == ["2", "4"]
    assert [e.example_id for e in s.validation] == ["3"] and [e.example_id for e in s.test] == ["1"]
    assert data.split_counts(s) == {"train": 2, "validation": 1, "test": 1}
    assert data.build_splits(exs, seed=99) == s
    with pytest.raises(ValueError):
        data.build_splits([ex(5, "other")], 0)


def test_overlap_report_lists_every_id_and_count():
    s = data.Splits(train=(ex("a", "train"), ex("b", "train")), validation=(ex("b", "dev"),), test=(ex("a", "test"), ex("b", "test")))
    overlaps = data.check_overlap(s)
    assert overlaps == {"train-validation": ["b"], "train-test": ["a", "b"], "validation-test": ["b"]}
    assert data.overlap_total(overlaps) == 2
    assert data.check_overlap(data.Splits((ex(1, "train"),), (ex(2, "dev"),), (ex(3, "test"),))) == {}


def test_shared_utterances_counts_texts():
    s = data.Splits((ex(1, "train", "hi"), ex(2, "train", "yo")), (ex(3, "dev", "hi"),), (ex(4, "test", "zz"),))
    assert data.shared_utterances(s) == {"train-validation": 1, "train-test": 0, "validation-test": 0}


def test_labels_from_train_only():
    train = (ex(1, "train", target=data.canonical_target("a_b", [("time", "x"), ("date", "y")])),
             ex(2, "train", target=data.canonical_target("c_d", [("time", "z")])))
    assert data.labels_from_train(train) == {"intents": ["a_b", "c_d"], "slot_types": ["date", "time"]}


def test_write_and_load_round_trip(tmp_path):
    s = data.build_splits([data.transform_example(RAW, 0), ex(2, "dev", "u2"), ex(3, "test", "u3")], 0)
    paths = data.write_splits(s, tmp_path / "massive_en")
    assert sorted(paths) == ["test", "train", "validation"]
    assert data.load_splits(tmp_path / "massive_en") == s
    assert "\r" not in paths["train"].read_text(encoding="utf-8")
