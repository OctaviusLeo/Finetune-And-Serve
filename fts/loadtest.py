"""Load-test harness for the Serving_Endpoint (Req 6.6; task 5.4).

Drives concurrent requests at the OpenAI-compatible Serving_Endpoint and reports
P50/P95 latency, tokens/s, and the max sustained concurrency, counting
failed/timed-out requests and EXCLUDING them from the latency/throughput stats
while still reporting the failed count. Each report is stamped with the
generating command, seed, concurrency level, and the full 40-character commit.

Pure vs. driven
---------------
:func:`summarize_outcomes` is a pure function (percentiles + failed exclusion)
with no I/O, so Property 9 can exercise it directly. :func:`run_load_test`
drives the concurrent requests through an injectable ``send`` callable (production
passes a :class:`fts.model_client.ModelClient`-backed sender), so the harness is
testable without a live server and never imports the heavy stack.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from fts import repro

COMMAND = "python -m fts.loadtest"


@dataclass(frozen=True)
class RequestOutcome:
    """The result of a single load-test request.

    ``ok`` is True for a successful request; failed/timed-out requests have
    ``ok=False`` and are excluded from latency/throughput stats. ``latency_ms``
    and ``completion_tokens`` are only meaningful when ``ok`` is True.
    """

    ok: bool
    latency_ms: float
    completion_tokens: int


@dataclass(frozen=True)
class LoadTestReport:
    """Immutable report for one load-test run at a fixed concurrency level.

    Latency percentiles and tokens/s are computed over SUCCESSFUL requests only;
    ``failed_requests`` is the count of failed/timed-out requests (excluded from
    the stats). ``max_concurrent_requests`` is the max sustained concurrency
    (see the sustaining criterion in :func:`run_load_test`). Stamped with the
    generating ``command``, ``seed``, ``concurrency_level``, and the full
    40-character ``commit``.
    """

    p50_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float
    max_concurrent_requests: int
    failed_requests: int
    seed: int
    concurrency_level: int
    command: str
    commit: str

    def __post_init__(self) -> None:
        if len(self.commit) != 40:
            raise ValueError(
                f"commit must be a 40-character hash, got {len(self.commit)} chars"
            )
        if self.p50_latency_ms > self.p95_latency_ms:
            raise ValueError("p50 latency must not exceed p95 latency")


@dataclass(frozen=True)
class _Stats:
    """Pure latency/throughput summary over successful outcomes only."""

    p50_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float
    failed_requests: int


def summarize_outcomes(outcomes: Sequence[RequestOutcome]) -> _Stats:
    """Compute P50/P95 latency and tokens/s over SUCCESSES only, with failed count.

    Pure function (Property 9). Failed/timed-out outcomes are excluded from the
    latency and throughput statistics but counted in ``failed_requests``. P50 is
    the 50th and P95 the 95th percentile of successful latencies, so
    ``p50 <= p95`` always holds. Tokens/s is total successful completion tokens
    divided by total successful latency (seconds); it is ``0.0`` when there are
    no successes or zero total latency. With no successful outcomes the
    percentiles are ``0.0``.
    """
    successes = [o for o in outcomes if o.ok]
    failed = sum(1 for o in outcomes if not o.ok)

    if not successes:
        return _Stats(
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            tokens_per_second=0.0,
            failed_requests=failed,
        )

    latencies = np.asarray([o.latency_ms for o in successes], dtype=float)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    # Numerical safety: percentile monotonicity guarantees p50 <= p95, but clamp
    # to defend the LoadTestReport invariant against any float edge case.
    if p50 > p95:
        p50 = p95

    total_tokens = sum(o.completion_tokens for o in successes)
    total_seconds = float(latencies.sum()) / 1000.0
    tokens_per_second = total_tokens / total_seconds if total_seconds > 0 else 0.0

    return _Stats(
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        tokens_per_second=tokens_per_second,
        failed_requests=failed,
    )


#: A sender issues one request and returns its outcome. Injected so the harness
#: is testable without a live server; production wraps a ModelClient.
Sender = Callable[[int], RequestOutcome]


def _run_concurrency_level(
    send: Sender, concurrency: int, requests: int
) -> list[RequestOutcome]:
    """Issue ``requests`` requests at ``concurrency`` in parallel; collect outcomes."""
    outcomes: list[RequestOutcome] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send, i) for i in range(requests)]
        for future in futures:
            try:
                outcomes.append(future.result())
            except Exception:  # noqa: BLE001 - a raised sender counts as a failure
                outcomes.append(
                    RequestOutcome(ok=False, latency_ms=0.0, completion_tokens=0)
                )
    return outcomes


def run_load_test(
    base_url: str,
    model: str,
    concurrency_levels: Sequence[int],
    seed: int,
    send: Sender | None = None,
    requests_per_level: int = 50,
    max_failure_rate: float = 0.05,
) -> list[LoadTestReport]:
    """Drive concurrent requests at the Serving_Endpoint and report per level.

    For each level in ``concurrency_levels`` (ascending), issues
    ``requests_per_level`` requests at that concurrency through ``send``,
    summarizes successes-only latency/throughput via :func:`summarize_outcomes`,
    and stamps the report with command/seed/concurrency/40-char commit.

    Sustaining criterion for ``max_concurrent_requests``: a level is "sustained"
    when its observed failure rate is at or below ``max_failure_rate`` (default
    5%). ``max_concurrent_requests`` on each report is the highest sustained
    level seen up to and including that level (``0`` if none sustained yet).

    ``send`` is injected so this is testable without a live server; production
    passes a sender backed by :class:`fts.model_client.ModelClient` against
    ``base_url``/``model``. A missing ``send`` builds the live sender lazily.
    """
    if send is None:
        send = _build_live_sender(base_url, model, seed)

    commit = repro.commit_hash()
    reports: list[LoadTestReport] = []
    max_sustained = 0

    for concurrency in sorted(concurrency_levels):
        outcomes = _run_concurrency_level(send, concurrency, requests_per_level)
        stats = summarize_outcomes(outcomes)

        total = len(outcomes)
        failure_rate = stats.failed_requests / total if total else 1.0
        if failure_rate <= max_failure_rate:
            max_sustained = max(max_sustained, concurrency)

        reports.append(
            LoadTestReport(
                p50_latency_ms=stats.p50_latency_ms,
                p95_latency_ms=stats.p95_latency_ms,
                tokens_per_second=stats.tokens_per_second,
                max_concurrent_requests=max_sustained,
                failed_requests=stats.failed_requests,
                seed=seed,
                concurrency_level=concurrency,
                command=COMMAND,
                commit=commit,
            )
        )

    return reports


def _build_live_sender(base_url: str, model: str, seed: int) -> Sender:
    """Build a sender backed by :class:`fts.model_client.ModelClient`.

    GPU-free (httpx only) but a live-network seam; the owner wires the concrete
    prompt at M2. Returns a callable that issues one request and times it,
    reporting a failed outcome on any client error.
    """
    import time

    from fts.model_client import ModelClient, ModelClientError

    client = ModelClient(
        base_url=base_url,
        model=model,
        api_key_env="",  # a local server needs no key; override at the call site
        timeout_s=30.0,
        seed=seed,
    )

    def send(_index: int) -> RequestOutcome:
        start = time.perf_counter()
        try:
            completion = client.complete("You are a benchmark.", "ping")
        except ModelClientError:
            return RequestOutcome(ok=False, latency_ms=0.0, completion_tokens=0)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestOutcome(
            ok=True,
            latency_ms=latency_ms,
            completion_tokens=completion.completion_tokens,
        )

    return send


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=COMMAND,
        description=(
            "Load-test the OpenAI-compatible Serving_Endpoint: reports P50/P95 "
            "latency, tokens/s, and max sustained concurrency over SUCCESSFUL "
            "requests only (failed/timed-out excluded but counted), each stamped "
            "with command/seed/concurrency/commit."
        ),
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the Serving_Endpoint.",
    )
    parser.add_argument(
        "--model", type=str, default="quantized-model-tbd", help="Served model name."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="One or more concurrency levels to drive (ascending).",
    )
    parser.add_argument(
        "--requests-per-level",
        dest="requests_per_level",
        type=int,
        default=50,
        help="Number of requests issued at each concurrency level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Integer seed stamped onto every reported number.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Runs non-interactively and exits 0 on success.

    Issues the live load test against the Serving_Endpoint (lazy httpx sender)
    and prints the stamped per-level reports as JSON.
    """
    args = build_parser().parse_args(argv)
    reports = run_load_test(
        base_url=args.base_url,
        model=args.model,
        concurrency_levels=args.concurrency,
        seed=args.seed,
        requests_per_level=args.requests_per_level,
    )
    print(json.dumps([asdict(r) for r in reports], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
