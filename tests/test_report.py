"""Unit tests for Results_Table generation and the plots (task 6.2; Req 10.1, 10.2)."""

from __future__ import annotations

import re

from fts import report, repro
from tests.results_fixture import STAMP, write_results


def sections(text: str) -> dict[str, str]:
    parts = re.split(r"(?m)^## ", text)[1:]
    return {p.splitlines()[0]: p for p in parts}


def test_regenerating_produces_the_file_with_every_section(tmp_path):
    results = write_results(tmp_path / "results")
    out = report.write_results_table(results, results / "summary.md", repro.to_dict(STAMP))
    text = out.read_text(encoding="utf-8")
    assert list(sections(text)) == list(report.SECTIONS)
    assert "_Missing" not in text and f"commit `{STAMP.commit}`" in text.splitlines()[3]
    again = report.render_results_table(results, repro.to_dict(STAMP))
    assert again == text  # regeneration is deterministic


def test_every_quality_metric_has_a_ci(tmp_path):
    text = report.render_results_table(write_results(tmp_path / "results"))
    s = sections(text)
    rows = [line for line in s["Test F1: baselines and fine-tuned seeds"].splitlines() if line.startswith("| ") and "System" not in line and "---" not in line]
    ci = re.compile(r"\d\.\d{4} \[\d\.\d{4}, \d\.\d{4}\]")
    assert len(rows) == 3 + 3 + 1
    assert all(ci.search(r) for r in rows if "mean" not in r) and "0.8812 ± 0.0022" in rows[-1]
    quant_rows = [line for line in s["Quantization quality"].splitlines() if line.startswith("| GGUF")]
    assert len(quant_rows) == 2 and all(ci.search(r) for r in quant_rows)
    assert ci.search(s["Quantization quality"].split("Reference:")[1])


def test_every_section_carries_a_stamp(tmp_path):
    text = report.render_results_table(write_results(tmp_path / "results"))
    for title, body in sections(text).items():
        assert re.search(r"command `[^`]+`, seed `\d+`, commit `[0-9a-f]{40}`", body), title


def test_section_contents(tmp_path):
    s = sections(report.render_results_table(write_results(tmp_path / "results")))
    assert "| B (selected) | 16 | 32 | 0.0002 | 0.8902 |" in s["Sweep"] and "serving seed 2" in s["Sweep"]
    assert "| wrong_intent | 151 |" in s["Error analysis"] and "| **Total analyzed** | 355 |" in s["Error analysis"]
    assert "| 32 | 520 | 1540 | 1280.0 | 0 | no |" in s["Load test"] and "Maximum sustained concurrency: **16**" in s["Load test"]
    assert "$0.4318" in s["Hosted spend"] and "Spend_Cap of $2.00" in s["Hosted spend"]
    assert "5.11 bits per weight" in s["Quantization quality"] and "| Test | 2974 |" in s["Data"]


def test_missing_sources_are_flagged(tmp_path):
    text = report.render_results_table(tmp_path)
    for name in ("data_counts.json", "baselines.json", "finetuned.json", "sweep.json", "quant_quality.json", "error_analysis.json", "loadtest.json"):
        assert f"_Missing: results/{name}_" in text


def test_plots_are_written(tmp_path):
    results = write_results(tmp_path / "results")
    loss = report.plot_loss_curves(results, results / "plots" / "loss_curves.png")
    qual = report.plot_sweep_quality(results, results / "plots" / "sweep_quality.png")
    assert loss.stat().st_size > 1000 and qual.stat().st_size > 1000
    assert loss.read_bytes()[:4] == b"\x89PNG"
    assert report.plot_loss_curves(tmp_path / "none", tmp_path / "x.png") is None
    assert report.plot_sweep_quality(tmp_path / "none", tmp_path / "y.png") is None


def test_main_writes_summary_and_plots(tmp_path, monkeypatch):
    write_results(tmp_path / "results")
    monkeypatch.setattr(repro, "start_run", lambda seed: STAMP)
    assert report.main(root=tmp_path) == 0
    assert (tmp_path / "results" / "summary.md").exists()
    assert sorted(p.name for p in (tmp_path / "results" / "plots").iterdir()) == sorted(report.PLOTS)


def test_formatters():
    assert report.f1(0.88123) == "0.8812" and report.ms(1539.6) == "1540" and report.tps(12.34) == "12.3" and report.usd(0.43184) == "0.4318"
    assert report.ci({"point": 0.5, "lower": 0.4, "upper": 0.6}) == "0.5000 [0.4000, 0.6000]"


def test_quantization_line_shows_llama_quantize_bpw(tmp_path):
    import json

    results = write_results(tmp_path / "results")
    q = json.loads((results / "quantization.json").read_text(encoding="utf-8"))
    q["llama_quantize"] = {"reported_bpw": [16.0, 5.08]}
    (results / "quantization.json").write_text(json.dumps(q), encoding="utf-8")
    assert "llama-quantize reports 5.08 BPW" in report.render_results_table(results)
    q["llama_quantize"] = {"quant_bpw": 5.07, "reported_bpw": [16.0, 5.08]}
    (results / "quantization.json").write_text(json.dumps(q), encoding="utf-8")
    assert "llama-quantize reports 5.07 BPW" in report.render_results_table(results)
