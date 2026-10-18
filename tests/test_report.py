"""Unit tests for the Results_Table writer + plots (task 6.2)."""

from __future__ import annotations

import re

from fts.eval_core import BootstrapCI
from fts.report import (
    LoadTestRow,
    MetricRow,
    QuantQualityRow,
    render_results_table,
    write_loss_curve_plot,
    write_results_table,
    write_sweep_quality_plot,
)


def _quality_rows():
    return [
        MetricRow("base-zero-shot", BootstrapCI(point=0.40, lower=0.35, upper=0.45)),
        MetricRow("base-few-shot", BootstrapCI(point=0.55, lower=0.50, upper=0.60)),
        MetricRow("hosted-few-shot", BootstrapCI(point=0.62, lower=0.57, upper=0.67)),
        MetricRow("Finetuned_Model", BootstrapCI(point=0.78, lower=0.74, upper=0.82)),
    ]


def _quant_row():
    return QuantQualityRow(
        finetuned=BootstrapCI(point=0.78, lower=0.74, upper=0.82),
        quantized=BootstrapCI(point=0.76, lower=0.72, upper=0.80),
    )


def _load_rows():
    return [
        LoadTestRow(1, 100.0, 150.0, 42.0, 1, 0),
        LoadTestRow(4, 180.0, 260.0, 90.0, 4, 1),
    ]


def test_regenerating_writes_summary_file(tmp_path):
    """Regenerating produces results/summary.md (idempotent overwrite)."""
    path = write_results_table(
        _quality_rows(), _quant_row(), _load_rows(), seed=0, results_dir=tmp_path
    )
    assert path.exists()
    assert path.name == "summary.md"
    first = path.read_text(encoding="utf-8")

    # Regeneration overwrites and is stable (same seed/commit/inputs).
    path2 = write_results_table(
        _quality_rows(), _quant_row(), _load_rows(), seed=0, results_dir=tmp_path
    )
    assert path2 == path
    assert path2.read_text(encoding="utf-8") == first


def test_ci_appears_for_every_reported_quality_metric():
    """A 95% CI column value appears for every reported quality metric."""
    md = render_results_table(_quality_rows(), _quant_row(), _load_rows(), seed=0)

    # Each quality row renders a "[lower, upper]" CI cell.
    ci_cells = re.findall(r"\[\d+\.\d{4}, \d+\.\d{4}\]", md)
    # 4 baseline/finetuned rows + 2 pre/post-quant rows = 6 CI cells.
    assert len(ci_cells) == 6

    for row in _quality_rows():
        assert f"{row.metric.point:.4f}" in md
        assert f"[{row.metric.lower:.4f}, {row.metric.upper:.4f}]" in md


def test_provenance_block_records_command_and_seed():
    """The summary records its generating command and seed (traceability)."""
    md = render_results_table(_quality_rows(), None, [], seed=7, command="python -m fts.report")
    assert "python -m fts.report" in md
    assert "seed: `7`" in md
    assert "No quantization quality recorded" in md
    assert "No Load_Test recorded" in md


def test_quant_delta_rendered():
    """The pre/post-quant row renders the point-estimate delta."""
    md = render_results_table(_quality_rows(), _quant_row(), _load_rows(), seed=0)
    # 0.76 - 0.78 = -0.02 -> rendered as -0.0200.
    assert "-0.0200" in md


def test_plots_written_to_results_dir(tmp_path):
    """The loss-curve and sweep/quality plots render into results/."""
    loss_path = write_loss_curve_plot({0: 2.0, 10: 1.2, 20: 0.8}, results_dir=tmp_path)
    sweep_path = write_sweep_quality_plot(
        {"cfg-a": 0.70, "cfg-b": 0.78, "cfg-c": 0.74}, results_dir=tmp_path
    )
    assert loss_path.exists() and loss_path.stat().st_size > 0
    assert sweep_path.exists() and sweep_path.stat().st_size > 0
