"""Unit tests for the load-test runner + entry point (task 5.6).

The runner is invocable non-interactively (via an injected sender so no live
server is needed) and stamps each report with command/seed/concurrency/commit.
Also checks the pure summarizer's success-only exclusion and the P50<=P95
invariant on the frozen report.
"""

from __future__ import annotations

import json

import pytest

from fts.loadtest import (
    COMMAND,
    LoadTestReport,
    RequestOutcome,
    build_parser,
    main,
    run_load_test,
    summarize_outcomes,
)


def _fixed_sender(index: int) -> RequestOutcome:
    """Deterministic sender: every request succeeds with a fixed latency/tokens."""
    return RequestOutcome(ok=True, latency_ms=100.0 + index, completion_tokens=10)


def test_runner_stamps_command_seed_concurrency_commit():
    reports = run_load_test(
        base_url="http://127.0.0.1:8000",
        model="m",
        concurrency_levels=[1, 2],
        seed=42,
        send=_fixed_sender,
        requests_per_level=5,
    )
    assert len(reports) == 2
    for report, concurrency in zip(reports, [1, 2]):
        assert isinstance(report, LoadTestReport)
        assert report.command == COMMAND
        assert report.seed == 42
        assert report.concurrency_level == concurrency
        assert len(report.commit) == 40
        assert report.p50_latency_ms <= report.p95_latency_ms
        assert report.failed_requests == 0


def test_runner_excludes_failures_from_stats_and_counts_them():
    """Failed requests are excluded from latency/throughput but counted."""

    def half_failing(index: int) -> RequestOutcome:
        if index % 2 == 0:
            return RequestOutcome(ok=True, latency_ms=200.0, completion_tokens=20)
        return RequestOutcome(ok=False, latency_ms=0.0, completion_tokens=0)

    reports = run_load_test(
        base_url="http://x",
        model="m",
        concurrency_levels=[4],
        seed=7,
        send=half_failing,
        requests_per_level=10,
    )
    report = reports[0]
    assert report.failed_requests == 5  # the odd indices
    # Latency is computed over successes (all 200.0), so both percentiles are 200.
    assert abs(report.p50_latency_ms - 200.0) <= 1e-9
    assert abs(report.p95_latency_ms - 200.0) <= 1e-9


def test_max_concurrent_requests_tracks_sustained_levels():
    """A level failing above the rate threshold is not counted as sustained."""

    def fail_above_two(index: int) -> RequestOutcome:
        # Determinism via the request index is not available per-level, so use a
        # sender that always fails to make the whole run non-sustained.
        return RequestOutcome(ok=False, latency_ms=0.0, completion_tokens=0)

    reports = run_load_test(
        base_url="http://x",
        model="m",
        concurrency_levels=[1, 2, 4],
        seed=0,
        send=fail_above_two,
        requests_per_level=10,
    )
    # Every level fully failed -> never sustained -> max_concurrent_requests 0.
    assert all(r.max_concurrent_requests == 0 for r in reports)
    assert all(r.failed_requests == 10 for r in reports)


def test_raised_sender_counts_as_failure():
    def boom(index: int) -> RequestOutcome:
        raise RuntimeError("send failed")

    reports = run_load_test(
        base_url="http://x",
        model="m",
        concurrency_levels=[2],
        seed=1,
        send=boom,
        requests_per_level=4,
    )
    assert reports[0].failed_requests == 4
    assert reports[0].max_concurrent_requests == 0


def test_load_test_report_rejects_p50_above_p95():
    with pytest.raises(ValueError):
        LoadTestReport(
            p50_latency_ms=200.0,
            p95_latency_ms=100.0,
            tokens_per_second=1.0,
            max_concurrent_requests=1,
            failed_requests=0,
            seed=0,
            concurrency_level=1,
            command=COMMAND,
            commit="0" * 40,
        )


def test_load_test_report_rejects_bad_commit_length():
    with pytest.raises(ValueError):
        LoadTestReport(
            p50_latency_ms=1.0,
            p95_latency_ms=2.0,
            tokens_per_second=1.0,
            max_concurrent_requests=1,
            failed_requests=0,
            seed=0,
            concurrency_level=1,
            command=COMMAND,
            commit="abc",
        )


def test_summarize_empty_is_degenerate():
    stats = summarize_outcomes([])
    assert stats.failed_requests == 0
    assert stats.p50_latency_ms == 0.0
    assert stats.p95_latency_ms == 0.0
    assert stats.tokens_per_second == 0.0


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.seed == 0
    assert args.concurrency == [1, 2, 4]
    assert args.requests_per_level == 50


def test_main_runs_non_interactively(monkeypatch, capsys):
    """`python -m fts.loadtest` runs non-interactively and prints stamped JSON reports."""

    # Inject a deterministic sender so main needs no live server.
    def _fake_run_load_test(base_url, model, concurrency_levels, seed, requests_per_level):
        return run_load_test(
            base_url=base_url,
            model=model,
            concurrency_levels=concurrency_levels,
            seed=seed,
            send=_fixed_sender,
            requests_per_level=requests_per_level,
        )

    import fts.loadtest as loadtest

    monkeypatch.setattr(loadtest, "run_load_test", _fake_run_load_test)

    rc = main(["--seed", "3", "--concurrency", "1", "2", "--requests-per-level", "4"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 2
    for row in payload:
        assert row["command"] == COMMAND
        assert row["seed"] == 3
        assert len(row["commit"]) == 40
