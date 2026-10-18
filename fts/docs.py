"""Generate README.md and MODEL_CARD.md from templates and generated results (Req 8, 9.6, 10.1-10.4).

Templates live in docs/README.template.md and docs/MODEL_CARD.template.md and
use ``{{name}}`` placeholders. Every number enters through a placeholder whose
value comes from a generated file: the sections of results/summary.md (rendered
by fts.report with the same formatters), results/*.json, or
results/doc_facts.json, which this module writes from the configs (split sizes,
seeds, schedule, prices, cap, fixed training settings). An unknown placeholder
is an error, so no number can be typed into a template by accident.

    python scripts/build_docs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fts import report, repro, settings

REPO_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
DATASET_LICENSE = "CC BY 4.0"  # MASSIVE NOTICE.md (and SLURP); recorded in THIRD_PARTY.md by scripts/build_data.py


def load_json(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def doc_facts(root: Path = REPO_ROOT) -> dict:
    """Configuration facts quoted in the docs (each traceable to this generated file)."""
    train = settings.load_train_settings(root / "config" / "train.yaml")
    sweep = settings.load_yaml(root / "config" / "sweep.yaml")
    hosted = settings.load_yaml(root / "config" / "hosted.yaml")
    serve = settings.load_yaml(root / "config" / "serve.yaml")
    lt = settings.load_yaml(root / "config" / "loadtest.yaml")
    model = settings.load_model_settings(root / "config" / "model.yaml")
    exemplars = load_json(root, "config/few_shot_exemplars.json")
    labels = load_json(root, "config/labels.json")
    prices = hosted["prices_usd_per_1m_tokens"]
    return {
        "base_model": model.name,
        "base_model_revision": model.revision,
        "base_model_license": model.license,
        "dataset_license": DATASET_LICENSE,
        "max_seq_len": train.max_seq_len,
        "per_device_batch_size": train.per_device_batch_size,
        "grad_accum": train.grad_accum,
        "effective_batch": train.per_device_batch_size * train.grad_accum,
        "epochs": int(train.epochs) if train.epochs == int(train.epochs) else train.epochs,
        "warmup_percent": round(train.warmup_ratio * 100),
        "lora_dropout": train.lora_dropout,
        "max_grad_norm": train.max_grad_norm,
        "logging_steps": train.logging_steps,
        "sweep_seed": sweep["seed"],
        "final_seeds": ", ".join(str(s) for s in sweep["final_seeds"]),
        "n_final_seeds": len(sweep["final_seeds"]),
        "n_sweep_configs": len(sweep["configs"]),
        "n_exemplars": len(exemplars["exemplars"]),
        "exemplar_seed": exemplars["seed"],
        "n_intents": len(labels["intents"]),
        "n_slot_types": len(labels["slot_types"]),
        "bootstrap_resamples": 1000,
        "ci_percent": 95,
        "max_new_tokens": 128,
        "hosted_model": hosted["model"],
        "price_input": prices["input"],
        "price_cached_input": prices["cached_input"],
        "price_cache_write": prices["cache_write"],
        "price_output": prices["output"],
        "prices_accessed": hosted["prices_accessed"],
        "spend_cap_usd": f"{float(hosted['spend_cap_usd']):.2f}",
        "llama_cpp_build": serve["llama_cpp_build"],
        "serve_port": int(serve["base_url"].rsplit(":", 1)[1].split("/")[0]),
        "serve_slots": 32,
        "serve_ctx": 16384,
        "loadtest_levels": ", ".join(str(c) for c in lt["concurrency_levels"]),
        "loadtest_warmup": lt["warmup_requests"],
        "loadtest_timed": lt["timed_requests"],
        "loadtest_timeout_s": lt["timeout_s"],
        "loadtest_max_tokens": lt["max_tokens"],
        "loadtest_p95_limit_ms": lt["p95_limit_ms"],
        "min_python": "3.11",
        "coverage_percent": 80,
    }


def facts(root: Path = REPO_ROOT) -> dict:
    """All placeholder values: config facts, summary sections, and headline numbers from results/*.json."""
    results = root / "results"
    f = {k: str(v) for k, v in doc_facts(root).items()}
    for key, render in zip(("summary_data", "summary_test", "summary_sweep", "summary_quant", "summary_errors", "summary_load", "summary_spend"), report.RENDERERS):
        f[key] = "\n".join(render(results))
    base = {b["name"]: b for b in load_json(root, "results/baselines.json")["baselines"]}
    ft = load_json(root, "results/finetuned.json")
    qq = load_json(root, "results/quant_quality.json")
    lt = load_json(root, "results/loadtest.json")
    ea = load_json(root, "results/error_analysis.json")
    counts = load_json(root, "results/data_counts.json")["counts"]
    variants = {v["method"]: v for v in qq["variants"]}
    top = max(ea["groups"], key=lambda g: g["count"])
    example = (ea["examples"].get(top["group"]) or [None])[0]
    f.update({
        "n_train": str(counts["train"]), "n_validation": str(counts["validation"]), "n_test": str(counts["test"]),
        "f1_zero_shot": report.ci(base["base_zero_shot"]["metric"]),
        "f1_few_shot": report.ci(base["base_few_shot"]["metric"]),
        "f1_hosted": report.ci(base["hosted_few_shot"]["metric"]),
        "f1_ft_mean": report.f1(ft["mean"]), "f1_ft_spread": report.f1(ft["spread"]),
        "serving_seed": str(ft["serving_seed"]),
        "f1_ft_serving": report.ci(next(s for s in ft["seeds"] if s["seed"] == ft["serving_seed"])["metric"]),
        "f1_f16": report.ci(variants["gguf-f16"]["metric"]), "f1_q4": report.ci(variants["gguf-q4_k_m"]["metric"]),
        "max_concurrency": str(lt["max_concurrent_requests"]),
        "ea_analyzed": str(ea["analyzed"]), "ea_top_group": top["group"], "ea_top_count": str(top["count"]),
        "ea_example": (f"for example, \"{example['utterance']}\" was predicted as `{example['prediction']}` against the gold `{example['gold']}`"
                       if example else "no example was saved"),
    })
    return f


def render(template: str, values: dict) -> str:
    missing = sorted({m for m in PLACEHOLDER.findall(template) if m not in values})
    if missing:
        raise KeyError(f"template placeholders without a generated value: {missing}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    stamp = repro.start_run(seed=0)
    repro.write_json(root / "results" / "doc_facts.json", doc_facts(root), stamp)
    values = facts(root)
    for template, out in (("docs/README.template.md", "README.md"), ("docs/MODEL_CARD.template.md", "MODEL_CARD.md")):
        text = render((root / template).read_text(encoding="utf-8"), values)
        (root / out).write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    return 0
