"""Unit tests for fts.massive with a synthetic MASSIVE-shaped tarball (no network)."""

from __future__ import annotations

import io
import json
import shutil
import tarfile

import pytest

from fts import massive, repro

STAMP = repro.Stamp(command="python scripts/build_data.py", seed=0, commit="2" * 40)


def rec(i, partition, utt, annot, intent="alarm_set"):
    return {"id": str(i), "locale": "en-US", "partition": partition, "scenario": "alarm", "intent": intent, "utt": utt, "annot_utt": annot}


RECORDS = [
    rec(1, "train", "wake me up at five am", "wake me up at [time : five am]"),
    rec(2, "train", "play jazz", "play [music_genre : jazz]", intent="play_music"),
    rec(3, "dev", "wake me up at six", "wake me up at [time : six]"),
    rec(4, "test", "play jazz", "play [music_genre : jazz]", intent="play_music"),
    rec(5, "test", "broken", "broken [time five]"),
]


def make_tar(tmp_path, records, extra_member=False, garbage_line=False):
    body = "\n".join(json.dumps(r) for r in records) + "\n"
    if garbage_line:
        body += "{not json\n"
    path = tmp_path / "massive.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, text in [("1.1/data/en-US.jsonl", body), ("1.1/data/de-DE.jsonl", "{}\n")] + ([("x/en-US.jsonl", body)] if extra_member else []):
            raw = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return path


@pytest.fixture()
def root(tmp_path):
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    shutil.copytree(massive.REPO_ROOT / "config" / "prompts", r / "config" / "prompts")
    (r / "THIRD_PARTY.md").write_text("# Third-party components\n\n## Dataset\n\nTBD\n", encoding="utf-8")
    return r


def count_chars(messages):
    return sum(len(m["content"]) for m in messages) // 4


def test_iter_member_records_reads_only_en_us(tmp_path):
    recs = list(massive.iter_member_records(make_tar(tmp_path, RECORDS, garbage_line=True)))
    assert len(recs) == 6 and recs[0]["id"] == "1" and recs[-1] == "{not json"


def test_iter_member_records_requires_exactly_one_member(tmp_path):
    with pytest.raises(ValueError, match="exactly one member"):
        list(massive.iter_member_records(make_tar(tmp_path, RECORDS, extra_member=True)))


def test_build_writes_splits_labels_counts_and_attribution(tmp_path, root):
    tar = make_tar(tmp_path, RECORDS)
    status, payload = massive.build(tar, "https://example.invalid/m.tgz", "2026-10-17", STAMP, count_chars, 256, root=root)
    assert status == 0
    assert payload["counts"] == {"train": 2, "validation": 1, "test": 1}
    assert payload["excluded"] == 1 and payload["exclusions"] == [{"id": "5", "reason": "slot span lacks ' : '"}]
    assert payload["shared_utterances"] == {"train-validation": 0, "train-test": 1, "validation-test": 0}
    assert payload["dataset"]["sha256"] == massive.sha256_of(tar)
    assert payload["longest_training_example"]["fits"] is True
    labels = json.loads((root / "config" / "labels.json").read_text(encoding="utf-8"))
    assert labels == {"intents": ["alarm_set", "play_music"], "slot_types": ["music_genre", "time"]}
    counts = json.loads((root / "results" / "data_counts.json").read_text(encoding="utf-8"))
    assert counts["stamp"]["commit"] == STAMP.commit
    assert (root / "data" / "massive_en" / "test.jsonl").read_text(encoding="utf-8").count("\n") == 1
    tp = (root / "THIRD_PARTY.md").read_text(encoding="utf-8")
    assert "CC-BY-4.0" in tp and "SLURP" in tp and massive.sha256_of(tar) in tp


def test_build_reports_overlong_examples(tmp_path, root, capsys):
    status, payload = massive.build(make_tar(tmp_path, RECORDS), "u", "d", STAMP, lambda m: 300, 256, root=root)
    assert status == 0 and payload["longest_training_example"] == {
        "tokens": 300, "example_id": "1", "max_seq_len": 256, "fits": False, "next_max_seq_len_if_not": 512,
    }
    assert "exceeds max_seq_len" in capsys.readouterr().err


def test_build_fails_on_overlap(tmp_path, root, capsys):
    dup = RECORDS + [rec(1, "test", "wake me up at five am", "wake me up at [time : five am]")]
    status, _ = massive.build(make_tar(tmp_path, dup), "u", "d", STAMP, count_chars, 256, root=root)
    assert status == 1
    err = capsys.readouterr().err
    assert "OVERLAP train-test: 1 ids: 1" in err and "1 examples appear in more than one split" in err
    assert not (root / "results" / "data_counts.json").exists()


def test_download_skips_existing_and_writes_atomically(tmp_path):
    dest = tmp_path / "cache" / "m.tgz"

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    calls = []
    assert massive.download("u", dest, opener=lambda url: calls.append(url) or Resp(b"abc")) == dest
    assert dest.read_bytes() == b"abc" and calls == ["u"]
    massive.download("u", dest, opener=lambda url: calls.append(url) or Resp(b"zzz"))
    assert dest.read_bytes() == b"abc" and calls == ["u"]


def test_next_power_of_two_above():
    assert massive.next_power_of_two_above(256) == 512
    assert massive.next_power_of_two_above(300) == 512
    assert massive.next_power_of_two_above(100) == 128


def test_chat_token_ids_handles_mapping_and_list():
    class DictTok:
        def apply_chat_template(self, messages, tokenize=True):
            return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}

    class ListTok:
        def apply_chat_template(self, messages, tokenize=True):
            return [5, 6]

    assert massive.chat_token_ids(DictTok(), []) == [1, 2, 3, 4]
    assert massive.chat_token_ids(ListTok(), []) == [5, 6]


def test_hf_token_counter_counts_input_ids(monkeypatch):
    import fts.lora

    class DictTok:
        def apply_chat_template(self, messages, tokenize=True):
            return {"input_ids": list(range(7)), "attention_mask": [1] * 7}

    monkeypatch.setattr(fts.lora, "load_tokenizer", lambda name, revision: DictTok())
    count = massive.hf_token_counter(massive.settings.load_model_settings())
    assert count([{"role": "user", "content": "x"}]) == 7
