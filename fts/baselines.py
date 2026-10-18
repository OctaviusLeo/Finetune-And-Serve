"""Baselines on the Test_Split (Req 3.1-3.6): base zero-shot, base few-shot, hosted few-shot.

Each baseline predicts every Test example, is scored with the Task_Metric, and
gets a 95% bootstrap CI. Each records its prompt template(s) and the number of
Test examples not scored. The local baselines run the Base_Model through
fts.infer. The hosted baseline asks TypeSafe's Jev, through JevClient and the
spend ledger, to select the intent and a slot type per candidate span, with the
same 10 exemplars in its state (fts.jev_extract), and assembles the task JSON.
Hosted predictions are appended to a cache as they arrive, so a restarted run
resumes without re-sending finished examples. If the ledger refuses a request,
the run stops, reports the spend reached and the cap, and exits non-zero.

``python -m fts.baselines --baseline local|hosted_few_shot|all`` writes one
stamped record per baseline to results/baselines/<name>.json and, once all
three exist, the combined results/baselines.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fts import data, exemplars, infer, jev_extract, prompts, repro, settings
from fts.eval_core import BootstrapCI, bootstrap_ci, split_digest
from fts.infer import Prediction
from fts.spend import SpendCapExceeded, SpendLedger

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "baselines"
COMBINED_JSON = REPO_ROOT / "results" / "baselines.json"
NAMES = ("base_zero_shot", "base_few_shot", "hosted_few_shot")
SEED = 0

LocalEngine = Callable[[Sequence[data.Example], Callable[[data.Example], list[dict]], int], list[Prediction]]


@dataclass(frozen=True)
class BaselineResult:
    name: str
    metric: BootstrapCI
    spend_usd: float  # 0.0 for local baselines; hosted from the ledger
    prompt_template: str  # recorded template path(s), comma-separated
    not_scored: int
    scored: int = 0
    n_test: int = 0
    test_ids_sha256: str = ""
    predictions_path: str = ""
    not_scored_examples: list[dict] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metric"] = self.metric.as_dict()
        return d


def templates_for(name: str) -> str:
    if name == "base_zero_shot":
        return prompts.template_path("system") + ", " + prompts.template_path("zero_shot")
    if name == "hosted_few_shot":
        return "config/prompts/jev.yaml, config/labels.json, config/few_shot_exemplars.json"
    return ", ".join([prompts.template_path("system"), prompts.template_path("few_shot"), prompts.template_path("few_shot_turn"), "config/few_shot_exemplars.json"])


def evaluate(name: str, test: Sequence[data.Example], preds: Sequence[Prediction], seed: int, spend_usd: float = 0.0,
             predictions_path: str = "", details: dict | None = None) -> BaselineResult:
    report = infer.score(test, preds)
    if not report.scores:
        raise ValueError(f"{name}: no Test example could be scored")
    return BaselineResult(
        name=name,
        metric=bootstrap_ci(report.scores, seed=seed),
        spend_usd=spend_usd,
        prompt_template=templates_for(name),
        not_scored=len(report.not_scored),
        scored=len(report.scores),
        n_test=len(test),
        test_ids_sha256=split_digest(test),
        predictions_path=predictions_path,
        not_scored_examples=report.not_scored,
        details=details or {},
    )


def message_builders(templates: prompts.Templates, labels: dict, shots: Sequence[data.Example]) -> dict[str, Callable[[data.Example], list[dict]]]:
    return {
        "base_zero_shot": lambda e: prompts.zero_shot_messages(templates, e.inputs, labels),
        "base_few_shot": lambda e: prompts.few_shot_messages(templates, e.inputs, labels, shots),
    }


def run_local(name: str, test: Sequence[data.Example], build: Callable[[data.Example], list[dict]], engine: LocalEngine, seed: int,
              predictions_dir: Path = infer.PREDICTIONS_DIR) -> BaselineResult:
    batch = infer.LONG_BATCH
    start = time.monotonic()
    preds = engine(test, build, batch)
    path = infer.save_predictions(predictions_dir / f"{name}.jsonl", test, preds)
    details = {"engine": "transformers bf16 greedy, max_new_tokens 128, left padding", "batch_size": batch, "seconds": round(time.monotonic() - start, 1),
               "prompt_tokens": sum(p.prompt_tokens for p in preds), "completion_tokens": sum(p.completion_tokens for p in preds)}
    return evaluate(name, test, preds, seed, predictions_path=_rel(path), details=details)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_cache(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["example_id"]] = row
    return rows


@dataclass(frozen=True)
class JevPlan:
    """What the hosted baseline asks Jev (design.md "Hosted model and Spend_Cap")."""

    labels: dict
    exemplars: Sequence[data.Example]
    templates: jev_extract.JevTemplates
    max_span_tokens: int
    per_request: int


def jev_predict(client, example: data.Example, plan: JevPlan) -> dict:
    """Ask Jev about one request (one or more requests over the same state) and assemble the task JSON."""
    spans, batches = jev_extract.question_batches(example.inputs, plan.labels, plan.templates, plan.max_span_tokens, plan.per_request)
    state = jev_extract.build_state(example.inputs, plan.exemplars)
    answers: dict = {}
    input_tokens = output_tokens = 0
    models: set[str] = set()
    for questions in batches:
        r = client.ask(state, questions)
        answers.update(r.answers)
        input_tokens += r.input_tokens
        output_tokens += r.output_tokens
        models.add(r.model)
    compact = {qid: [a["choice"], round(float(a["probabilities"].get(a["choice"], 0.0)), 4)] for qid, a in answers.items()}
    return {"example_id": example.example_id, "text": jev_extract.decode(answers, spans), "input_tokens": input_tokens,
            "output_tokens": output_tokens, "requests": len(batches), "questions": len(spans) + 1, "models": sorted(models), "answers": compact}


def run_hosted(test: Sequence[data.Example], client, ledger: SpendLedger, plan: JevPlan, seed: int, cache_path: Path,
               log: Callable[[str], None] = lambda m: print(m, flush=True)) -> BaselineResult:
    """Ask Jev about each Test example once (resuming from ``cache_path``); SpendCapExceeded propagates after the cache is saved."""
    cache = _load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with cache_path.open("a", encoding="utf-8") as fh:
        for i, ex in enumerate(test):
            if ex.example_id in cache:
                continue
            row = jev_predict(client, ex, plan)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            cache[ex.example_id] = row
            if (i + 1) % 100 == 0:
                log(f"hosted: {i + 1}/{len(test)} done, spend ${ledger.total_usd:.4f} of ${ledger.cap_usd:.2f}")
    rows = [cache[e.example_id] for e in test]
    preds = [Prediction(r["example_id"], r["text"], r["input_tokens"], r["output_tokens"]) for r in rows]
    path = infer.save_predictions(cache_path.with_name("hosted_few_shot.jsonl"), test, preds)
    details = {
        "engine": "TypeSafe Jev over HTTP: intent choice plus one slot-type choice per candidate span, assembled in code",
        "configured_model": client.model,
        "response_models": sorted({m for r in rows for m in r["models"]}),
        "max_span_tokens": plan.max_span_tokens,
        "requests": sum(r["requests"] for r in rows),
        "questions": sum(r["questions"] for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "seconds_this_session": round(time.monotonic() - start, 1),
        "ledger": ledger.summary(),
    }
    return evaluate("hosted_few_shot", test, preds, seed, spend_usd=_baseline_spend(ledger), predictions_path=_rel(path), details=details)


def _baseline_spend(ledger: SpendLedger) -> float:
    """Spend of the hosted baseline itself (ledger entries tagged hosted_few_shot)."""
    return sum(ledger.prices.cost(e["prompt_tokens"], e["completion_tokens"]) for e in ledger.entries if e.get("tag") == "hosted_few_shot")


def run_baselines(splits: data.Splits, few_shot_exemplars: Sequence[data.Example], spend_cap_usd: float, seed: int, *,
                  engine: LocalEngine, client, ledger: SpendLedger, labels: dict, templates: prompts.Templates, plan: JevPlan,
                  cache_path: Path, predictions_dir: Path = infer.PREDICTIONS_DIR) -> list[BaselineResult]:
    """All three baselines on the Test_Split (design.md interface); the hosted client must use ``ledger``."""
    if abs(ledger.cap_usd - spend_cap_usd) > 1e-12:
        raise ValueError(f"ledger cap {ledger.cap_usd} differs from spend_cap_usd {spend_cap_usd}")
    builders = message_builders(templates, labels, few_shot_exemplars)
    results = [run_local(n, splits.test, builders[n], engine, seed, predictions_dir) for n in ("base_zero_shot", "base_few_shot")]
    results.append(run_hosted(splits.test, client, ledger, plan, seed, cache_path))
    return results


def plan_from_config(cfg: dict, labels: dict, shots: Sequence[data.Example], templates_path: Path) -> JevPlan:
    return JevPlan(labels=labels, exemplars=tuple(shots), templates=jev_extract.load_templates(templates_path),
                   max_span_tokens=int(cfg["max_span_tokens"]), per_request=int(cfg["max_span_questions_per_request"]))


def write_result(result: BaselineResult, stamp: repro.Stamp, results_dir: Path = RESULTS_DIR) -> Path:
    return repro.write_json(results_dir / f"{result.name}.json", result.to_dict(), stamp)


def combine(results_dir: Path = RESULTS_DIR) -> dict | None:
    """The combined payload when all three per-baseline records exist and share one Test_Split, else None."""
    records = {}
    for name in NAMES:
        p = results_dir / f"{name}.json"
        if not p.exists():
            return None
        records[name] = json.loads(p.read_text(encoding="utf-8"))
    digests = {r["test_ids_sha256"] for r in records.values()}
    if len(digests) != 1:
        raise ValueError(f"baselines were scored on different Test_Splits: {digests}")
    hosted = records["hosted_few_shot"]
    ledger = hosted["details"]["ledger"]
    return {
        "baselines": [records[n] for n in NAMES],
        "test_ids_sha256": digests.pop(),
        "n_test": hosted["n_test"],
        "hosted_spend": {
            "baseline_spend_usd": hosted["spend_usd"],
            "ledger_total_usd": ledger["spend_usd"],
            "spend_cap_usd": ledger["spend_cap_usd"],
            "within_cap": ledger["spend_usd"] <= ledger["spend_cap_usd"],
            "prompt_price_per_mtok": ledger["prompt_price_per_mtok"],
            "completion_price_per_mtok": ledger["completion_price_per_mtok"],
        },
    }


def default_engine(model_settings: settings.ModelSettings, device: str = "cuda") -> LocalEngine:
    """Loads the Base_Model once, on first use, and predicts with fts.infer."""
    state: dict = {}

    def engine(examples, build, batch_size):
        if "model" not in state:
            state["model"], state["tok"] = infer.load_for_inference(model_settings.name, model_settings.revision, device=device)
        return infer.predict(state["model"], state["tok"], examples, build, batch_size,
                             progress=lambda d, n: print(f"  {d}/{n}", flush=True) if d % 400 < batch_size or d == n else None)

    return engine


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, engine: LocalEngine | None = None, transport=None) -> int:
    from fts import hosted as hosted_mod
    from fts.model_client import MissingSecretError, ModelClientError

    parser = argparse.ArgumentParser(prog="python -m fts.baselines", description="Evaluate the baselines on the Test_Split.")
    parser.add_argument("--baseline", choices=[*NAMES, "local", "all"], default="all")
    args = parser.parse_args(argv)
    stamp = repro.start_run(seed=SEED)

    splits = data.load_splits(root / "data" / "massive_en")
    labels = prompts.load_labels(root / "config" / "labels.json")
    templates = prompts.load_templates(root / "config" / "prompts")
    shots = exemplars.load_exemplars(root / "config" / "few_shot_exemplars.json")
    builders = message_builders(templates, labels, shots)
    names = {"local": ["base_zero_shot", "base_few_shot"], "all": list(NAMES)}.get(args.baseline, [args.baseline])
    results_dir = root / "results" / "baselines"
    predictions_dir = root / "runs" / "predictions"

    local = [n for n in names if n != "hosted_few_shot"]
    if local:
        engine = engine or default_engine(settings.load_model_settings(root / "config" / "model.yaml"))
        for name in local:
            print(f"running {name}", flush=True)
            result = run_local(name, splits.test, builders[name], engine, SEED, predictions_dir)
            write_result(result, stamp, results_dir)
            print(f"{name}: F1 {result.metric.point:.4f} [{result.metric.lower:.4f}, {result.metric.upper:.4f}], not scored {result.not_scored}", flush=True)

    hosted_cfg = hosted_mod.load_config(root / "config" / "hosted.yaml")
    if "hosted_few_shot" in names:
        try:
            client, ledger = hosted_mod.open_hosted(hosted_cfg, root, tag="hosted_few_shot", transport=transport)
        except MissingSecretError as exc:
            print(f"FAIL: {exc}; no hosted request was sent", file=sys.stderr)
            return 2
        plan = plan_from_config(hosted_cfg, labels, shots, root / "config" / "prompts" / "jev.yaml")
        with client:
            try:
                result = run_hosted(splits.test, client, ledger, plan, SEED, predictions_dir / "hosted_few_shot.cache.jsonl")
            except SpendCapExceeded as exc:
                print(f"FAIL: {exc}. Spend reached ${exc.spent_usd:.6f}; cap ${exc.cap_usd:.2f}.", file=sys.stderr)
                return 1
            except ModelClientError as exc:
                print(f"FAIL: hosted request failed: {exc} (completed requests are cached and will not be re-sent)", file=sys.stderr)
                return 1
        write_result(result, stamp, results_dir)
        print(f"hosted_few_shot: F1 {result.metric.point:.4f} [{result.metric.lower:.4f}, {result.metric.upper:.4f}], "
              f"spend ${result.spend_usd:.4f} (ledger ${ledger.total_usd:.4f} of ${ledger.cap_usd:.2f})", flush=True)

    combined = combine(results_dir)
    if combined is not None:
        repro.write_json(root / "results" / "baselines.json", combined, stamp)
        print("wrote results/baselines.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
