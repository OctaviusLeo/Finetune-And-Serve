"""Pre/post-quantization quality through the Serving_Endpoint (Req 6.3, 6.4).

For each GGUF file (F16, then Q4_K_M): start it with ``scripts/serve.sh``, send
every Test example one request at a time (compact prompt, temperature 0, seed)
through ModelClient while sampling VRAM once per second, score with the
Task_Metric and a 95% CI, then stop this project's serve container. The same
seed's Hugging Face-evaluated F1 (results/finetuned.json) is shown beside them
for reference; numbers from the two engines are compared only within an engine.

    python -m fts.quant_quality  -> results/quant_quality.json
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from fts import data, gpu, infer, prompts, repro, settings
from fts.eval_core import evaluate_predictions
from fts.infer import Prediction
from fts.model_client import ModelClient, ModelClientError, client_from_config
from fts.vram_monitor import VramMonitor

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_SEED = 0
MAX_TOKENS = 128
VARIANTS = (("f16", "gguf-f16"), ("q4", "gguf-q4_k_m"))


def serve(variant: str, root: Path = REPO_ROOT) -> None:
    out = subprocess.run(["bash", "scripts/serve.sh", variant], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"scripts/serve.sh {variant} failed ({out.returncode}): {out.stderr.strip()[-1000:]}")


def stop(root: Path = REPO_ROOT) -> None:
    """Stop only this project's serve container."""
    subprocess.run(["docker", "compose", "-f", "docker-compose.yml", "stop", "serve"], cwd=root, capture_output=True, text=True)


def predict_endpoint(client: ModelClient, test: Sequence[data.Example], templates: prompts.Templates,
                     log: Callable[[str], None] = lambda m: print(m, flush=True)) -> list[Prediction]:
    """One request at a time, in dataset order; a failed request is recorded as an empty prediction (scores 0)."""
    preds = []
    for i, ex in enumerate(test):
        try:
            c = client.complete(prompts.finetune_messages(templates, ex.inputs), MAX_TOKENS)
            preds.append(Prediction(ex.example_id, c.text, c.prompt_tokens, c.completion_tokens))
        except ModelClientError as exc:
            log(f"request for {ex.example_id} failed: {exc}")
            preds.append(Prediction(ex.example_id, "", 0, 0))
        if (i + 1) % 500 == 0:
            log(f"  {i + 1}/{len(test)}")
    return preds


def evaluate_variant(variant: str, method: str, test: Sequence[data.Example], serve_cfg: dict, templates: prompts.Templates, root: Path,
                     serve_fn: Callable[[str], None], stop_fn: Callable[[], None], client_factory: Callable[[dict], ModelClient],
                     vram_dir: Path, vram_query: Callable[[], gpu.GpuMemory] = gpu.query_memory) -> dict:
    serve_fn(variant)
    start = time.monotonic()
    try:
        with VramMonitor(vram_dir / f"serve_{variant}.csv", query=vram_query) as mon, client_factory(serve_cfg) as client:
            preds = predict_endpoint(client, test, templates)
    finally:
        stop_fn()
    infer.save_predictions(root / "runs" / "predictions" / f"serve_{variant}.jsonl", test, preds)
    result = evaluate_predictions(f"serve-{variant}", 0, test, preds, BOOTSTRAP_SEED)
    return {
        "variant": variant,
        "method": method,
        "model_file": serve_cfg["models"][variant],
        "metric": result.metric.as_dict(),
        "scored": result.scored,
        "not_scored": len(result.not_scored),
        "failed_requests": sum(1 for p in preds if p.completion_tokens == 0 and p.text == ""),
        "test_ids_sha256": result.test_ids_sha256,
        "seconds": round(time.monotonic() - start, 1),
        "vram": mon.summary(),
    }


def default_client(serve_cfg: dict) -> ModelClient:
    cfg = dict(serve_cfg)
    cfg["api_key_env"] = serve_cfg["api_key_env"] if serve_cfg.get("require_api_key") else None
    return client_from_config(cfg)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, serve_fn: Callable[[str], None] | None = None, stop_fn: Callable[[], None] | None = None,
         client_factory: Callable[[dict], ModelClient] = default_client, vram_query: Callable[[], gpu.GpuMemory] = gpu.query_memory) -> int:
    import json

    finetuned_path = root / "results" / "finetuned.json"
    if not finetuned_path.exists():
        print("FAIL: results/finetuned.json is missing (the reference Hugging Face F1)", file=sys.stderr)
        return 1
    finetuned = json.loads(finetuned_path.read_text(encoding="utf-8"))
    stamp = repro.start_run(seed=BOOTSTRAP_SEED)
    serve_cfg = settings.load_yaml(root / "config" / "serve.yaml")
    templates = prompts.load_templates(root / "config" / "prompts")
    test = data.load_splits(root / "data" / "massive_en").test
    serve_fn = serve_fn or (lambda v: serve(v, root))
    stop_fn = stop_fn or (lambda: stop(root))
    rows = []
    for variant, method in VARIANTS:
        print(f"evaluating {variant} through the endpoint", flush=True)
        try:
            rows.append(evaluate_variant(variant, method, test, serve_cfg, templates, root, serve_fn, stop_fn, client_factory, root / "runs" / "vram", vram_query))
        except RuntimeError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        m = rows[-1]["metric"]
        print(f"{variant}: F1 {m['point']:.4f} [{m['lower']:.4f}, {m['upper']:.4f}], peak VRAM {rows[-1]['vram']['peak_mib']} of {rows[-1]['vram']['total_mib']} MiB", flush=True)
    serving_seed = finetuned["serving_seed"]
    reference = next(s for s in finetuned["seeds"] if s["seed"] == serving_seed)
    payload = {
        "variants": rows,
        "reference_hf": {"seed": serving_seed, "run_id": reference["run_id"], "metric": reference["metric"], "engine": "transformers bf16 (fts.infer)"},
        "engine": "llama-server " + serve_cfg["llama_cpp_build"] + ", one request at a time, temperature 0, seed " + str(serve_cfg["seed"]),
        "prompt_template": f"{prompts.template_path('system')}, {prompts.template_path('finetune')}",
        "peak_vram_within_total": all(r["vram"]["within_budget"] for r in rows),
    }
    repro.write_json(root / "results" / "quant_quality.json", payload, stamp)
    if not payload["peak_vram_within_total"]:
        print("FAIL: peak serving VRAM exceeded the nvidia-smi total", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
