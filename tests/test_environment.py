"""Unit tests for fts.environment (tasks 1.1, 1.2; Req 1.1-1.5). CPU only."""

from __future__ import annotations

from fts.environment import (
    parse_nvidia_smi_total_mib,
    python_version_ok,
    vram_within_budget,
)


def test_python_version_guard_accepts_3_11_and_newer():
    assert python_version_ok("3.11.5")
    assert python_version_ok("3.11")
    assert python_version_ok("3.12.0")
    assert python_version_ok("4.0.0")


def test_python_version_guard_rejects_older_and_malformed():
    assert not python_version_ok("3.10.9")
    assert not python_version_ok("2.7.18")
    assert not python_version_ok("not-a-version")
    assert not python_version_ok("")


def test_vram_within_budget_comparison():
    assert vram_within_budget(5000.0, 10240)
    assert vram_within_budget(10240.0, 10240)  # exactly at budget fits
    assert not vram_within_budget(10241.0, 10240)
    assert not vram_within_budget(100.0, 0)  # no budget available


def test_parse_nvidia_smi_csv_form():
    assert parse_nvidia_smi_total_mib("10240\n") == 10240
    assert parse_nvidia_smi_total_mib("10240 MiB\n") == 10240


def test_parse_nvidia_smi_table_form():
    text = "|  0%   45C    P8    30W / 320W |   1234MiB / 10240MiB |"
    assert parse_nvidia_smi_total_mib(text) == 10240


def test_parse_nvidia_smi_none_when_absent():
    assert parse_nvidia_smi_total_mib("no numbers here") is None
