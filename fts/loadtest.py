"""Load test of the Serving_Endpoint (Req 7.1-7.6).

Schedule (config/loadtest.yaml): concurrency levels 1, 2, 4, 8, 16, 32 against
the Q4_K_M model on llama-server. At each level: 5 untimed warm-up requests,
then 128 timed requests sent by ``concurrency`` worker threads. Prompts are Test
utterances in the compact format drawn with ``random.Random(seed)``;
``max_tokens`` 128, temperature 0, 60-second timeout, no client retries.

Per level:
* P50 and P95 latency in ms over successful requests (numpy linear percentiles);
* tokens/s = completion tokens of successful requests / wall-clock time of the timed part;
* failed requests (errors and timeouts), counted and excluded from both;
* sustained = no failed request and P95 <= 2,000 ms.
The maximum concurrency is the highest sustained level (0 if none). Every level
is stamped with the command, seed, concurrency and 40-character commit.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RequestOutcome:
    ok: bool
    latency_ms: float
    completion_tokens: int


@dataclass(frozen=True)
class LevelStats:
    p50_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float
    failed_requests: int
    successful_requests: int


@dataclass(frozen=True)
class LevelReport:
    concurrency_level: int
    p50_latency_ms: float
    p95_latency_ms: float  # p50 <= p95
    tokens_per_second: float
    failed_requests: int  # excluded from latency and throughput
    sustained: bool
    successful_requests: int = 0
    wall_seconds: float = 0.0
    stamp: dict | None = None


@dataclass(frozen=True)
class LoadTestReport:
    levels: tuple[LevelReport, ...]
    max_concurrent_requests: int
    seed: int
    command: str
    commit: str

    def to_dict(self) -> dict:
        return {"levels": [asdict(lv) for lv in self.levels], "max_concurrent_requests": self.max_concurrent_requests,
                "seed": self.seed, "command": self.command, "commit": self.commit}


def summarize(outcomes: Sequence[RequestOutcome], wall_seconds: float) -> LevelStats:
    """Percentiles and throughput over successful requests only; failures counted separately."""
    ok = [o for o in outcomes if o.ok]
    failed = len(outcomes) - len(ok)
    if not ok:
        return LevelStats(0.0, 0.0, 0.0, failed, 0)
    lat = np.asarray([o.latency_ms for o in ok], dtype=float)
    p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
    tps = sum(o.completion_tokens for o in ok) / wall_seconds if wall_seconds > 0 else 0.0
    return LevelStats(min(p50, p95), p95, tps, failed, len(ok))


def is_sustained(stats: LevelStats, p95_limit_ms: float) -> bool:
    return stats.failed_requests == 0 and stats.successful_requests > 0 and stats.p95_latency_ms <= p95_limit_ms


def max_sustained(levels: Sequence[LevelReport]) -> int:
    return max((lv.concurrency_level for lv in levels if lv.sustained), default=0)


def draw_prompts(utterances: Sequence[str], n: int, rng: random.Random) -> list[str]:
    return [rng.choice(list(utterances)) for _ in range(n)]


Sender = Callable[[str], RequestOutcome]


def run_level(send: Sender, prompts: Sequence[str], concurrency: int, warmup: int) -> tuple[list[RequestOutcome], float]:
    """Warm up sequentially, then send the timed prompts with ``concurrency`` workers; returns outcomes and wall time."""
    for p in prompts[:warmup]:
        send(p)
    timed = prompts[warmup:]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(send, timed))
    return outcomes, time.perf_counter() - start


def run_load_test(send: Sender, utterances: Sequence[str], concurrency_levels: Sequence[int], seed: int, *, warmup: int, timed: int,
                  p95_limit_ms: float, command: str, commit: str, log: Callable[[str], None] = lambda m: print(m, flush=True)) -> LoadTestReport:
    rng = random.Random(seed)
    levels = []
    for c in concurrency_levels:
        prompts = draw_prompts(utterances, warmup + timed, rng)
        outcomes, wall = run_level(send, prompts, c, warmup)
        stats = summarize(outcomes, wall)
        report = LevelReport(c, stats.p50_latency_ms, stats.p95_latency_ms, stats.tokens_per_second, stats.failed_requests,
                             is_sustained(stats, p95_limit_ms), stats.successful_requests, round(wall, 3),
                             {"command": command, "seed": seed, "concurrency": c, "commit": commit})
        levels.append(report)
        log(f"concurrency {c}: p50 {report.p50_latency_ms:.0f} ms, p95 {report.p95_latency_ms:.0f} ms, {report.tokens_per_second:.1f} tok/s, "
            f"failed {report.failed_requests}, sustained {report.sustained}")
    return LoadTestReport(tuple(levels), max_sustained(levels), seed, command, commit)


def make_sender(client, build_messages: Callable[[str], list[dict]], max_tokens: int) -> Sender:
    """One timed request through a shared ModelClient (httpx.Client is thread-safe); any client error is a failure."""
    from fts.model_client import ModelClientError

    def send(utterance: str) -> RequestOutcome:
        start = time.perf_counter()
        try:
            c = client.complete(build_messages(utterance), max_tokens)
        except ModelClientError:
            return RequestOutcome(False, (time.perf_counter() - start) * 1000, 0)
        return RequestOutcome(True, (time.perf_counter() - start) * 1000, c.completion_tokens)

    return send


def main(argv: list[str] | None = None, root=None, serve_fn=None, stop_fn=None, client=None, vram_query=None) -> int:
    """Single command: start the configured model, run the schedule while sampling VRAM, stop it, write results/loadtest.json."""
    import argparse
    import sys
    from pathlib import Path

    from fts import data, gpu, prompts, quant_quality, repro, settings
    from fts.model_client import ClientConfig, ModelClient
    from fts.vram_monitor import VramMonitor

    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(prog="python scripts/loadtest.py", description="Load-test the Serving_Endpoint.")
    parser.parse_args(argv)
    cfg = settings.load_yaml(root / "config" / "loadtest.yaml")
    stamp = repro.start_run(seed=int(cfg["seed"]))
    serve_cfg = settings.load_yaml(root / "config" / "serve.yaml")
    templates = prompts.load_templates(root / "config" / "prompts")
    utterances = [e.inputs for e in data.load_splits(root / "data" / "massive_en").test]
    serve_fn = serve_fn or (lambda v: quant_quality.serve(v, root))
    stop_fn = stop_fn or (lambda: quant_quality.stop(root))
    if client is None:
        client = ModelClient(ClientConfig(base_url=serve_cfg["base_url"], model=serve_cfg["model"],
                                          api_key_env=serve_cfg["api_key_env"] if serve_cfg.get("require_api_key") else None,
                                          timeout_s=float(cfg["timeout_s"]), seed=int(serve_cfg["seed"]), temperature=float(serve_cfg["temperature"]),
                                          max_tokens_field=serve_cfg["max_tokens_field"], max_attempts=1))
    try:
        serve_fn(cfg["variant"])
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    try:
        with VramMonitor(root / "runs" / "vram" / "loadtest.csv", query=vram_query or gpu.query_memory) as mon, client:
            send = make_sender(client, lambda u: prompts.finetune_messages(templates, u), int(cfg["max_tokens"]))
            report = run_load_test(send, utterances, [int(c) for c in cfg["concurrency_levels"]], int(cfg["seed"]), warmup=int(cfg["warmup_requests"]),
                                   timed=int(cfg["timed_requests"]), p95_limit_ms=float(cfg["p95_limit_ms"]), command=stamp.command, commit=stamp.commit)
    finally:
        stop_fn()
    vram = mon.summary()
    payload = report.to_dict() | {
        "schedule": cfg,
        "model_file": serve_cfg["models"][cfg["variant"]],
        "server": f"llama-server {serve_cfg['llama_cpp_build']} (-np 32 -c 16384)",
        "sustaining_criterion": f"no failed request and P95 <= {cfg['p95_limit_ms']} ms",
        "vram": vram,
    }
    repro.write_json(root / "results" / "loadtest.json", payload, stamp)
    print(f"maximum sustained concurrency {report.max_concurrent_requests}; peak VRAM {vram['peak_mib']} of {vram['total_mib']} MiB")
    return 0 if vram["within_budget"] else 1
