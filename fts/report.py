"""Results_Table and plots generated from results/*.json (Req 10.1, 10.2).

``write_results_table`` renders results/summary.md, the only place reported
numbers are tabulated. Sections: data counts; baselines and the three final
seeds plus their mean +/- standard deviation (Test F1 with 95% CI); the sweep
(Validation F1); pre- versus post-quantization quality; error-analysis group
counts; the load test per concurrency level with the maximum sustained level;
hosted spend versus the Spend_Cap. Each section names its source file and that
file's command, seed and 40-character commit. A missing source is shown as
``_Missing: <path>_`` so the gate can flag it. Every number goes through the
formatters below, which scripts/build_docs.py reuses, so the README shows the
same strings the summary does.

The plots (matplotlib, Agg backend) go to results/plots/: the training loss of
the three final seeds, and Validation F1 per sweep config beside Test F1 with
95% CIs per system.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ("Data", "Test F1: baselines and fine-tuned seeds", "Sweep", "Quantization quality", "Error analysis", "Load test", "Hosted spend")
PLOTS = ("loss_curves.png", "sweep_quality.png")
SYSTEM_LABELS = {"base_zero_shot": "Base zero-shot", "base_few_shot": "Base few-shot", "hosted_few_shot": "Hosted few-shot (TypeSafe Jev)"}


def f1(x: float) -> str:
    return f"{x:.4f}"


def ci(metric: dict) -> str:
    return f"{f1(metric['point'])} [{f1(metric['lower'])}, {f1(metric['upper'])}]"


def ms(x: float) -> str:
    return f"{x:.0f}"


def tps(x: float) -> str:
    return f"{x:.1f}"


def usd(x: float) -> str:
    return f"{x:.4f}"


def load(results_dir: Path, name: str) -> dict | None:
    p = results_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def stamp_line(source: str, payload: dict) -> str:
    s = payload.get("stamp", {})
    return f"Source `results/{source}`; command `{s.get('command')}`, seed `{s.get('seed')}`, commit `{s.get('commit')}`."


def missing(source: str) -> list[str]:
    return [f"_Missing: results/{source}_"]


def section_data(r: Path) -> list[str]:
    d = load(r, "data_counts.json")
    if d is None:
        return missing("data_counts.json")
    c = d["counts"]
    return [
        "| Split | Examples |", "| --- | --- |",
        f"| Train | {c['train']} |", f"| Validation | {c['validation']} |", f"| Test | {c['test']} |",
        "",
        f"Excluded as malformed: {d['excluded']}. Overlapping example ids across splits: {d['overlap']['total']}. "
        f"Utterance texts shared train/validation {d['shared_utterances']['train-validation']}, train/test {d['shared_utterances']['train-test']}, "
        f"validation/test {d['shared_utterances']['validation-test']}.",
        "", stamp_line("data_counts.json", d),
    ]


def section_test(r: Path) -> list[str]:
    lines = ["| System | Prompt | Test F1 [95% CI] | Not scored |", "| --- | --- | --- | --- |"]
    base, ft = load(r, "baselines.json"), load(r, "finetuned.json")
    if base is None:
        lines = missing("baselines.json") + [""] + lines
    else:
        for b in base["baselines"]:
            prompt = "zero-shot" if b["name"] == "base_zero_shot" else "few-shot (10 exemplars)"
            lines.append(f"| {SYSTEM_LABELS[b['name']]} | {prompt} | {ci(b['metric'])} | {b['not_scored']} |")
    if ft is None:
        lines += ["", *missing("finetuned.json")]
    else:
        for s in ft["seeds"]:
            lines.append(f"| Fine-tuned, seed {s['seed']} | compact | {ci(s['metric'])} | {s['not_scored']} |")
        lines.append(f"| Fine-tuned, mean ± std over {len(ft['seeds'])} seeds | compact | {f1(ft['mean'])} ± {f1(ft['spread'])} | |")
    lines.append("")
    lines.append("Every system is scored on the same Test_Split with the same field-level F1; each uses its own recorded prompt template.")
    if base is not None:
        lines += ["", stamp_line("baselines.json", base)]
    if ft is not None:
        lines += ["", stamp_line("finetuned.json", ft)]
    return lines


def section_sweep(r: Path) -> list[str]:
    s = load(r, "sweep.json")
    if s is None:
        return missing("sweep.json")
    lines = ["| Config | lora_r | lora_alpha | Learning rate | Validation F1 |", "| --- | --- | --- | --- | --- |"]
    for c in s["configs"]:
        mark = " (selected)" if c["name"] == s["selected"]["name"] else ""
        lines.append(f"| {c['name']}{mark} | {c['lora_r']} | {c['lora_alpha']} | {c['learning_rate']:g} | {f1(s['val_metrics'][c['name']])} |")
    lines += ["", f"Selection: {s['selection_rule']}. Rationale: `{s['rationale_path']}`."]
    final = load(r, "final_runs.json")
    if final is not None:
        vals = ", ".join(f"seed {x['seed']} {f1(x['val_metric'])}" for x in final["runs"])
        lines += ["", f"Final_Configuration Validation F1: {vals}; serving seed {final['serving_seed']}.", "", stamp_line("final_runs.json", final)]
    lines += ["", stamp_line("sweep.json", s)]
    return lines


def section_quant(r: Path) -> list[str]:
    q = load(r, "quant_quality.json")
    if q is None:
        return missing("quant_quality.json")
    names = {"gguf-f16": "GGUF F16 (pre-quantization)", "gguf-q4_k_m": "GGUF Q4_K_M (4-bit nominal)"}
    lines = ["| Model (llama-server) | Test F1 [95% CI] | Failed requests | Peak VRAM / total (MiB) |", "| --- | --- | --- | --- |"]
    for v in q["variants"]:
        lines.append(f"| {names[v['method']]} | {ci(v['metric'])} | {v['failed_requests']} | {v['vram']['peak_mib']} / {v['vram']['total_mib']} |")
    ref = q["reference_hf"]
    lines += ["", f"Reference: the same seed ({ref['seed']}) evaluated with Hugging Face transformers: {ci(ref['metric'])}. "
                  "Numbers are compared only within one engine.", ""]
    qz = load(r, "quantization.json")
    if qz is not None:
        files = qz["files"]
        lines += [f"Q4_K_M file: {files['gguf-q4_k_m']['size_bytes']} bytes, {files['gguf-q4_k_m']['bits_per_weight_from_size']:.2f} bits per weight by size "
                  f"(F16: {files['gguf-f16']['bits_per_weight_from_size']:.2f}).", "", stamp_line("quantization.json", qz), ""]
    lines.append(stamp_line("quant_quality.json", q))
    return lines


def section_errors(r: Path) -> list[str]:
    e = load(r, "error_analysis.json")
    if e is None:
        return missing("error_analysis.json")
    lines = ["| Failure group | Count |", "| --- | --- |"]
    lines += [f"| {g['group']} | {g['count']} |" for g in e["groups"]]
    lines += [f"| **Total analyzed** | {e['analyzed']} |", "", f"Every Test failure (F1 < 1) of the serving seed ({e['seed']}) is assigned to the first matching group.",
              "", stamp_line("error_analysis.json", e)]
    return lines


def section_load(r: Path) -> list[str]:
    lt = load(r, "loadtest.json")
    if lt is None:
        return missing("loadtest.json")
    lines = ["| Concurrency | P50 (ms) | P95 (ms) | Tokens/s | Failed | Sustained |", "| --- | --- | --- | --- | --- | --- |"]
    for lv in lt["levels"]:
        lines.append(f"| {lv['concurrency_level']} | {ms(lv['p50_latency_ms'])} | {ms(lv['p95_latency_ms'])} | {tps(lv['tokens_per_second'])} | "
                     f"{lv['failed_requests']} | {'yes' if lv['sustained'] else 'no'} |")
    lines += ["", f"Maximum sustained concurrency: **{lt['max_concurrent_requests']}** ({lt['sustaining_criterion']}). "
                  f"Peak VRAM during the test: {lt['vram']['peak_mib']} of {lt['vram']['total_mib']} MiB.", "", stamp_line("loadtest.json", lt)]
    return lines


def section_spend(r: Path) -> list[str]:
    base = load(r, "baselines.json")
    if base is None:
        return missing("baselines.json")
    h = base["hosted_spend"]
    return [f"Hosted few-shot baseline: ${usd(h['baseline_spend_usd'])}; hosted ledger total ${usd(h['ledger_total_usd'])} against the "
            f"Spend_Cap of ${h['spend_cap_usd']:.2f} (prompt tokens charged at ${h['prompt_price_per_mtok']} and completion tokens at "
            f"${h['completion_price_per_mtok']} per 1M tokens, an upper bound on the bill).", "", stamp_line("baselines.json", base)]


RENDERERS = (section_data, section_test, section_sweep, section_quant, section_errors, section_load, section_spend)


def render_results_table(results_dir: Path, stamp: dict | None = None) -> str:
    out = ["# Results summary", "", "Generated by `python scripts/build_summary.py` from `results/*.json`; do not edit by hand."]
    if stamp is not None:
        out.append(f"This file: command `{stamp['command']}`, seed `{stamp['seed']}`, commit `{stamp['commit']}`.")
    out.append("")
    for title, render in zip(SECTIONS, RENDERERS):
        out += [f"## {title}", "", *render(results_dir), ""]
    return "\n".join(out).rstrip("\n") + "\n"


def write_results_table(results_dir: Path | str, path: Path | str, stamp: dict | None = None) -> Path:
    p = Path(path)
    p.write_text(render_results_table(Path(results_dir), stamp), encoding="utf-8")
    return p


def plot_loss_curves(results_dir: Path, out: Path) -> Path | None:
    final = load(results_dir, "final_runs.json")
    if final is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for run in final["runs"]:
        steps, losses = zip(*run["loss_curve"])
        ax.plot(steps, losses, label=f"seed {run['seed']}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss (JSON tokens)")
    ax.set_title(f"LoRA training loss, config {final['final_config']['name']}")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_sweep_quality(results_dir: Path, out: Path) -> Path | None:
    sweep, base, ft = load(results_dir, "sweep.json"), load(results_dir, "baselines.json"), load(results_dir, "finetuned.json")
    if sweep is None or base is None or ft is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2]})
    names = [c["name"] for c in sweep["configs"]]
    left.bar(names, [sweep["val_metrics"][n] for n in names], color="#4C72B0")
    left.set_title("Sweep: Validation F1")
    left.set_ylim(0, 1)
    systems = [(SYSTEM_LABELS[b["name"]].replace(" (TypeSafe Jev)", ""), b["metric"]) for b in base["baselines"]]
    systems += [(f"fine-tuned s{s['seed']}", s["metric"]) for s in ft["seeds"]]
    qq = load(results_dir, "quant_quality.json")
    if qq is not None:
        systems += [("served " + v["variant"].upper(), v["metric"]) for v in qq["variants"]]
    points = [m["point"] for _, m in systems]
    err = [[m["point"] - m["lower"] for _, m in systems], [m["upper"] - m["point"] for _, m in systems]]
    right.bar(range(len(systems)), points, yerr=err, capsize=3, color="#55A868")
    right.set_xticks(range(len(systems)), [n for n, _ in systems], rotation=30, ha="right")
    right.set_title("Test F1 with 95% CI")
    right.set_ylim(0, 1)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    from fts import repro

    stamp = repro.start_run(seed=0)
    results = root / "results"
    write_results_table(results, results / "summary.md", repro.to_dict(stamp))
    made = [p for p in (plot_loss_curves(results, results / "plots" / PLOTS[0]), plot_sweep_quality(results, results / "plots" / PLOTS[1])) if p]
    print(f"wrote results/summary.md and {len(made)} plot(s)")
    return 0
