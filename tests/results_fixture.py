"""A complete, internally consistent results/ directory for report, docs and gate tests (values are fixtures, not results)."""

from __future__ import annotations

from pathlib import Path

from fts import repro

STAMP = repro.Stamp(command="python -m fts.fixture", seed=0, commit="d" * 40)
DIGEST = "e" * 64


def metric(p: float, w: float = 0.01) -> dict:
    return {"point": p, "lower": round(p - w, 6), "upper": round(p + w, 6)}


def cfg(name: str, r: int, a: int, lr: float, seed: int = 1) -> dict:
    return {"name": name, "lora_r": r, "lora_alpha": a, "learning_rate": lr, "epochs": 2.0, "max_seq_len": 256, "per_device_batch_size": 8,
            "grad_accum": 2, "seed": seed}


def write_results(results: Path) -> Path:
    w = lambda name, payload: repro.write_json(results / name, payload, STAMP)  # noqa: E731
    w("data_counts.json", {"counts": {"train": 11514, "validation": 2033, "test": 2974}, "excluded": 0, "overlap": {"pairs": {}, "total": 0},
                           "shared_utterances": {"train-validation": 11, "train-test": 21, "validation-test": 5}})
    ledger = {"spend_usd": 0.4321, "spend_cap_usd": 2.0, "prompt_price_per_mtok": 0.125, "completion_price_per_mtok": 0.5, "requests": 2975}
    baselines = [
        {"name": "base_zero_shot", "metric": metric(0.3112), "not_scored": 0, "test_ids_sha256": DIGEST, "prompt_template": "config/prompts/zero_shot.txt"},
        {"name": "base_few_shot", "metric": metric(0.4523), "not_scored": 0, "test_ids_sha256": DIGEST, "prompt_template": "config/prompts/few_shot.txt"},
        {"name": "hosted_few_shot", "metric": metric(0.7012), "not_scored": 0, "test_ids_sha256": DIGEST, "prompt_template": "config/prompts/few_shot.txt",
         "details": {"ledger": ledger, "response_models": ["gpt-6-luna-2026-09-30"]}},
    ]
    w("baselines.json", {"baselines": baselines, "test_ids_sha256": DIGEST, "n_test": 2974,
                         "hosted_spend": {"baseline_spend_usd": 0.4318, "ledger_total_usd": 0.4321, "spend_cap_usd": 2.0, "within_cap": True,
                                          "prompt_price_per_mtok": 0.125, "completion_price_per_mtok": 0.5}})
    w("sweep.json", {"configs": [cfg("A", 8, 16, 2e-4), cfg("B", 16, 32, 2e-4), cfg("C", 16, 32, 1e-4)], "val_metrics": {"A": 0.8811, "B": 0.8902, "C": 0.8855},
                     "selected": cfg("B", 16, 32, 2e-4), "selection_rule": "highest Validation F1; ties go to the config listed first",
                     "rationale_path": "docs/SWEEP_RATIONALE.md"})
    runs = [{"run_id": f"B-seed{s}", "seed": s, "val_metric": v, "adapter_path": f"adapters/B-seed{s}", "commit": "d" * 40,
             "loss_curve": [[10, 1.9 - s * 0.01], [20, 0.8], [30, 0.4]], "config": cfg("B", 16, 32, 2e-4, s)} for s, v in ((1, 0.8902), (2, 0.8931), (3, 0.8877))]
    w("final_runs.json", {"final_config": cfg("B", 16, 32, 2e-4), "runs": runs, "seeds": [1, 2, 3], "serving_seed": 2})
    seeds = [{"run_id": f"B-seed{s}", "seed": s, "metric": metric(p), "not_scored": 0, "test_ids_sha256": DIGEST, "scored": 2974, "n_test": 2974}
             for s, p in ((1, 0.8790), (2, 0.8834), (3, 0.8812))]
    w("finetuned.json", {"seeds": seeds, "mean": 0.8812, "spread": 0.0022, "serving_seed": 2, "test_ids_sha256": DIGEST, "n_test": 2974})
    groups = [{"group": g, "count": n} for g, n in (("invalid_output", 3), ("wrong_intent", 151), ("missing_slot", 64), ("extra_slot", 40), ("wrong_slot_value", 97))]
    w("error_analysis.json", {"analyzed": 355, "available_failures": 355, "groups": groups, "seed": 2, "run_id": "B-seed2",
                              "examples": {"wrong_intent": [{"id": "9", "utterance": "what's on today", "gold": "{}", "prediction": "{}", "f1": 0.5}]}})
    w("quantization.json", {"models": [{"method": "gguf-f16", "bit_width": 16}, {"method": "gguf-q4_k_m", "bit_width": 4}],
                            "files": {"gguf-f16": {"size_bytes": 3093669888, "bits_per_weight_from_size": 16.02},
                                      "gguf-q4_k_m": {"size_bytes": 986048768, "bits_per_weight_from_size": 5.11}}, "image_digest": "x@sha256:" + "f" * 64})
    vram = {"peak_mib": 6123, "total_mib": 10240, "within_budget": True}
    w("quant_quality.json", {"variants": [{"variant": "f16", "method": "gguf-f16", "metric": metric(0.8829), "failed_requests": 0, "vram": vram},
                                          {"variant": "q4", "method": "gguf-q4_k_m", "metric": metric(0.8791), "failed_requests": 0, "vram": vram}],
                             "reference_hf": {"seed": 2, "run_id": "B-seed2", "metric": metric(0.8834)}, "peak_vram_within_total": True})
    levels = [{"concurrency_level": c, "p50_latency_ms": 200.0 + c * 10, "p95_latency_ms": 260.0 + c * 40, "tokens_per_second": 40.0 * c, "failed_requests": 0,
               "sustained": c <= 16, "stamp": {"command": "python scripts/loadtest.py", "seed": 0, "concurrency": c, "commit": "d" * 40}} for c in (1, 2, 4, 8, 16, 32)]
    w("loadtest.json", {"levels": levels, "max_concurrent_requests": 16, "sustaining_criterion": "no failed request and P95 <= 2000 ms", "vram": vram})
    return results
