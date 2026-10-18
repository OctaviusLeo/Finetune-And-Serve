"""Unit tests for the CPU Smoke_Evaluation (task 6.6; Req 9.3)."""

from __future__ import annotations

import shutil

from fts import smoke_eval


def copy_fixtures(tmp_path):
    d = tmp_path / "fixtures"
    shutil.copytree(smoke_eval.FIXTURES, d)
    return d


def test_runs_on_cpu_and_produces_the_expected_result(capsys):
    result = smoke_eval.run()
    assert result["counts"] == {"train": 2, "validation": 1, "test": 2} and result["excluded"] == 1 and result["overlap"] == 0
    assert abs(result["scores"]["s4"] - 0.8) < 1e-9 and result["scores"]["s5"] == 1.0 and abs(result["mean_f1"] - 0.9) < 1e-9
    assert smoke_eval.mismatches(result) == []
    assert smoke_eval.main([]) == 0 and "smoke evaluation OK" in capsys.readouterr().out


def test_does_not_import_torch():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", "import sys, fts.smoke_eval; fts.smoke_eval.run(); print('torch' in sys.modules)"],
                         capture_output=True, text=True, cwd=smoke_eval.FIXTURES.parent.parent)
    assert out.stdout.strip() == "False"


def test_fails_on_malformed_input(tmp_path, capsys):
    d = copy_fixtures(tmp_path)
    (d / "smoke_massive.jsonl").write_text((d / "smoke_massive.jsonl").read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    assert smoke_eval.main(["--fixtures", str(d)]) == 1
    assert "invalid JSON line" in capsys.readouterr().err


def test_fails_when_a_good_record_becomes_malformed(tmp_path, capsys):
    d = copy_fixtures(tmp_path)
    text = (d / "smoke_massive.jsonl").read_text(encoding="utf-8").replace("[time : five am] [date : this week]", "[time : five am [date : this week]")
    (d / "smoke_massive.jsonl").write_text(text, encoding="utf-8")
    assert smoke_eval.main(["--fixtures", str(d)]) == 1
    err = capsys.readouterr().err
    assert "counts" in err and "excluded" in err


def test_fails_on_missing_prediction_or_file(tmp_path, capsys):
    d = copy_fixtures(tmp_path)
    (d / "smoke_predictions.jsonl").write_text('{"id": "s5", "prediction": "{}"}\n', encoding="utf-8")
    assert smoke_eval.main(["--fixtures", str(d)]) == 1
    assert "no prediction for Test example s4" in capsys.readouterr().err
    assert smoke_eval.main(["--fixtures", str(tmp_path / "nowhere")]) == 1


def test_fails_on_wrong_scores(tmp_path):
    d = copy_fixtures(tmp_path)
    (d / "smoke_predictions.jsonl").write_text('{"id": "s4", "prediction": "x"}\n{"id": "s5", "prediction": "x"}\n', encoding="utf-8")
    assert smoke_eval.main(["--fixtures", str(d)]) == 1
    problems = smoke_eval.mismatches(smoke_eval.run(d))
    assert any(p.startswith("scores") for p in problems) and any(p.startswith("mean_f1") for p in problems)


def test_no_test_examples_is_an_error(tmp_path):
    d = copy_fixtures(tmp_path)
    (d / "smoke_massive.jsonl").write_text('{"id": "a", "partition": "train", "intent": "x_y", "utt": "hi", "annot_utt": "hi"}\n', encoding="utf-8")
    assert smoke_eval.main(["--fixtures", str(d)]) == 1
