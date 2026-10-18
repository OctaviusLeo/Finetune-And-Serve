# Fine-Tune and Serve

Code is released under the [MIT License](LICENSE). Third-party packages, data and models are listed with their licenses in [THIRD_PARTY.md](THIRD_PARTY.md). There is no published package or release, and the fine-tuned weights and LoRA Adapters are not published.

This project fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` with LoRA on one narrow task: turning an English voice-assistant request into intent-and-slot JSON, using the MASSIVE dataset. It measures whether that beats prompting the base model and a hosted model on the same Test_Split and metric. It then quantizes the fine-tuned model with llama.cpp, serves it behind an OpenAI-compatible endpoint, and load-tests it on one RTX 3080.

Every number in this file is generated. It comes from `results/summary.md` or `results/*.json`, and each of those records the command, seed and commit that produced it. `python scripts/build_docs.py` writes this README from `docs/README.template.md`, and `python scripts/check_traceability.py` fails if any number here cannot be traced to a generated file.

## Headline

- **Fine-tuned** (mean ± standard deviation over 3 seeds): field-level F1 **0.8503 ± 0.0016** on the 2974 Test examples.
- **Baselines** on the same Test_Split and metric: base model zero-shot 0.3232 [0.3118, 0.3347], base model few-shot 0.3578 [0.3448, 0.3699], TypeSafe Jev (`jev-1.13.0`) few-shot 0.6872 [0.6745, 0.6996]. Each figure is the point estimate with its 95% bootstrap CI.
- **Served** with llama-server: GGUF F16 0.8501 [0.8394, 0.8600], GGUF Q4_K_M 0.8462 [0.8360, 0.8561]. Maximum sustained concurrency in the load test: **16**.

## Architecture

```mermaid
flowchart LR
    D[MASSIVE en-US] --> P[Transform + official splits + overlap check]
    P --> B[Baselines: base zero-shot, base few-shot, hosted few-shot]
    P --> T[LoRA training on bf16 weights]
    T --> S[Sweep on Validation] --> F[Final seeds]
    F --> E[Test evaluation + bootstrap CI + error analysis]
    F --> Q[Merge Adapter -> GGUF F16 -> Q4_K_M]
    Q --> V[llama-server, OpenAI-compatible] --> L[Load test]
    B & E & V & L --> R[results/summary.md] --> M[README + Model card]
```

- **Task and metric.** Input: one utterance. Output: `{"intent": ..., "slots": [{"type": ..., "value": ...}]}`. The score is field-level F1 over the multiset of (intent) and (slot type, value) pairs, with strings lowercased and whitespace collapsed. An invalid output scores 0. Range: 0 to 1.
- **Data.** MASSIVE 1.1, English, in its official partitions: 11514 Train, 2033 Validation and 2974 Test examples, checked to be disjoint. The prompts list 60 intents and 55 slot types, built from the Train_Split only.
- **Training.** LoRA on all attention and MLP projections of the bf16 base model, with `max_seq_len` 256, per-device batch 4 and gradient accumulation 4 (effective batch 16). 2 epochs, linear decay with 3% warmup, LoRA dropout 0.05, gradient clipping 1.0. The loss covers the JSON tokens only, and the seeds are fixed. A 3-config sweep selects the configuration on Validation F1, and that configuration is trained with seeds 1, 2, 3.
- **Evaluation.** Hugging Face `transformers` in bf16 with greedy decoding (`max_new_tokens` 128). Each metric gets a 95% percentile bootstrap CI from 1000 resamples. Every Test failure of the serving seed is grouped by type.
- **Serving.** The serving seed's Adapter is merged into the base model, converted to GGUF F16 and quantized to GGUF Q4_K_M with llama.cpp `b11580`. It is served by `llama-server` in Docker on port 8080 with 32 slots.

## Results

The tables below are the sections of `results/summary.md`.

### Test F1: baselines and fine-tuned seeds

| System | Prompt | Test F1 [95% CI] | Not scored |
| --- | --- | --- | --- |
| Base zero-shot | zero-shot | 0.3232 [0.3118, 0.3347] | 0 |
| Base few-shot | few-shot (10 exemplars) | 0.3578 [0.3448, 0.3699] | 0 |
| Hosted few-shot (TypeSafe Jev) | few-shot (10 exemplars) | 0.6872 [0.6745, 0.6996] | 0 |
| Fine-tuned, seed 1 | compact | 0.8516 [0.8411, 0.8607] | 0 |
| Fine-tuned, seed 2 | compact | 0.8508 [0.8402, 0.8604] | 0 |
| Fine-tuned, seed 3 | compact | 0.8485 [0.8381, 0.8579] | 0 |
| Fine-tuned, mean ± std over 3 seeds | compact | 0.8503 ± 0.0016 | |

Every system is scored on the same Test_Split with the same field-level F1; each uses its own recorded prompt template.

Source `results/baselines.json`; command `python -m fts.baselines --baseline hosted_few_shot`, seed `0`, commit `93cd217aa7743b458b9c3da497791e4633f56386`.

Source `results/finetuned.json`; command `python -m fts.eval`, seed `0`, commit `4e37f1be3ac2763570c4003ee343d59076f9be41`.

![Training loss of the final seeds](results/plots/loss_curves.png)

![Sweep Validation F1 and Test F1 with CIs](results/plots/sweep_quality.png)

### Sweep

| Config | lora_r | lora_alpha | Learning rate | Validation F1 |
| --- | --- | --- | --- | --- |
| A | 8 | 16 | 0.0002 | 0.8390 |
| B (selected) | 16 | 32 | 0.0002 | 0.8407 |
| C | 16 | 32 | 0.0001 | 0.8340 |

Selection: highest Validation F1; ties go to the config listed first. Rationale: `docs/SWEEP_RATIONALE.md`.

Final_Configuration Validation F1: seed 1 0.8407, seed 2 0.8450, seed 3 0.8429; serving seed 2.

Source `results/final_runs.json`; command `python -m fts.sweep final`, seed `1`, commit `2538faa3850db6c1902b6fcb4d9f44119879afd2`.

Source `results/sweep.json`; command `python -m fts.sweep sweep`, seed `1`, commit `2538faa3850db6c1902b6fcb4d9f44119879afd2`.

### Quantization quality

| Model (llama-server) | Test F1 [95% CI] | Failed requests | Peak VRAM / total (MiB) |
| --- | --- | --- | --- |
| GGUF F16 (pre-quantization) | 0.8501 [0.8394, 0.8600] | 0 | 4859 / 10240 |
| GGUF Q4_K_M (4-bit nominal) | 0.8462 [0.8360, 0.8561] | 0 | 2617 / 10240 |

Reference: the same seed (2) evaluated with Hugging Face transformers: 0.8508 [0.8402, 0.8604]. Numbers are compared only within one engine.

Q4_K_M file: 986048064 bytes, 5.11 bits per weight by size (F16: 16.03); llama-quantize reports 5.08 BPW.

Source `results/quantization.json`; command `python -m fts.quantize`, seed `2`, commit `e47011f37ab925ba66d7ec4d0929b0cb53712ae4`.

Source `results/quant_quality.json`; command `python -m fts.quant_quality`, seed `0`, commit `e47011f37ab925ba66d7ec4d0929b0cb53712ae4`.

### Load test

Schedule: concurrency levels 1, 2, 4, 8, 16, 32. At each level, 5 warm-up requests and 128 timed requests are sent, with a 60 s timeout and `max_tokens` 128. Latency percentiles and tokens/s count successful requests only. A level is sustained when no request fails and P95 is at or below 2000 ms.

| Concurrency | P50 (ms) | P95 (ms) | Tokens/s | Failed | Sustained |
| --- | --- | --- | --- | --- | --- |
| 1 | 111 | 207 | 239.7 | 0 | yes |
| 2 | 167 | 266 | 312.7 | 0 | yes |
| 4 | 281 | 484 | 358.7 | 0 | yes |
| 8 | 466 | 858 | 435.0 | 0 | yes |
| 16 | 738 | 1353 | 497.3 | 0 | yes |
| 32 | 1356 | 2843 | 515.1 | 0 | no |

Maximum sustained concurrency: **16** (no failed request and P95 <= 2000 ms). Peak VRAM during the test: 2642 of 10240 MiB.

Source `results/loadtest.json`; command `python scripts/loadtest.py`, seed `0`, commit `45a605d82be8518b86c5eba168578d099236d601`.

### Error analysis

| Failure group | Count |
| --- | --- |
| invalid_output | 0 |
| wrong_intent | 293 |
| missing_slot | 267 |
| extra_slot | 188 |
| wrong_slot_value | 112 |
| **Total analyzed** | 860 |

Every Test failure (F1 < 1) of the serving seed (2) is assigned to the first matching group.

Source `results/error_analysis.json`; command `python -m fts.eval`, seed `0`, commit `4e37f1be3ac2763570c4003ee343d59076f9be41`.

### Data

| Split | Examples |
| --- | --- |
| Train | 11514 |
| Validation | 2033 |
| Test | 2974 |

Excluded as malformed: 0. Overlapping example ids across splits: 0. Utterance texts shared train/validation 11, train/test 21, validation/test 5.

Source `results/data_counts.json`; command `python scripts/build_data.py`, seed `0`, commit `fd1f18e7141320f836000a4a81f0b274ffb84186`.

### Hosted spend

Hosted few-shot baseline: $1.8624; hosted ledger total $1.8711 against the Spend_Cap of $2.00 (prompt tokens charged at $0.042 and completion tokens at $0.0 per 1M tokens, an upper bound on the bill).

Source `results/baselines.json`; command `python -m fts.baselines --baseline hosted_few_shot`, seed `0`, commit `93cd217aa7743b458b9c3da497791e4633f56386`.

## Commands

Everything runs inside WSL `Ubuntu-24.04` through the wrapper. Each command below is written as `<command>` in:

```
wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Finetune-And-Serve/scripts/wsl_run.sh <command>
```

| What | Single command |
| --- | --- |
| Unit and property tests (CPU, with the per-module coverage rule) | `bash scripts/test_unit.sh` |
| CPU Smoke_Evaluation | `python -m fts.smoke_eval` |
| Start the Serving_Endpoint (Q4_K_M by default, or `f16`) | `bash scripts/serve.sh` |
| Data counts and splits (`results/data_counts.json`) | `python scripts/build_data.py` |
| VRAM Dry_Run (`docs/ENVIRONMENT.md`) | `python scripts/dry_run.py` |
| Baselines (`results/baselines.json`) | `python -m fts.baselines --baseline all` |
| Sweep (`results/sweep.json`) | `python -m fts.sweep sweep` |
| Final seeds (`results/final_runs.json`) | `python -m fts.sweep final` |
| Training determinism (`results/train_determinism.json`) | `python scripts/check_train_determinism.py` |
| Test evaluation and error analysis (`results/finetuned.json`, `results/error_analysis.json`) | `python -m fts.eval` |
| Quantization (`results/quantization.json`) | `python -m fts.quantize` |
| Pre/post-quantization quality (`results/quant_quality.json`) | `python -m fts.quant_quality` |
| Load test (`results/loadtest.json`) | `python scripts/loadtest.py` |
| Results table and plots (`results/summary.md`) | `python scripts/build_summary.py` |
| This README and the Model card | `python scripts/build_docs.py` |
| Number traceability check | `python scripts/check_traceability.py` |

CI (GitHub Actions on `ubuntu-latest`) runs the unit tests and the Smoke_Evaluation on every push and pull request. Modules with more than one public function or long bodies must reach 80% statement coverage. Environment versions and evidence are in `docs/ENVIRONMENT.md`, and licenses are in `THIRD_PARTY.md`.

## Known limitations

- **Each system has its own prompt.** The base and hosted baselines see the full intent and slot-type lists plus 10 exemplars from the Train_Split. The fine-tuned model sees a compact prompt. The comparison is between documented systems on the same Test_Split and metric.
- **The hosted baseline selects; it does not generate.** TypeSafe's Jev returns typed probabilities, so it picks the intent from the label list and, for each candidate span of up to 6 words, a slot type or none; code assembles the JSON and resolves overlapping spans. Slot values longer than that cannot be produced. The version is pinned to `jev-1.13.0`, and each response's `model` field is recorded.
- **Duplicate utterances.** MASSIVE's official partitions share some utterance texts. The counts are in the Data section.
- **Two inference engines.** Training-stage numbers come from Hugging Face `transformers` in bf16. The pre/post-quantization comparison and the load test use `llama-server`. Numbers are compared only within one engine.
- **"4-bit" is nominal.** Q4_K_M mixes 4-bit and higher-precision k-quant blocks. The measured bits per weight are in the Quantization section.
- **One GPU under WSL2.** Latency and throughput are specific to an RTX 3080 under WSL2 with the compact prompt and short outputs, and other programs shared the GPU.
- **Weights are not published.**

## How this was built

This project was built with Claude Code (Anthropic's agentic coding tool) under a spec-driven workflow: written requirements, then a design, then a task list organized into milestones (the spec files under `.kiro/specs/`), each milestone ending in an automated gate script that had to pass before the next one started. The commit history shows this openly; commits carry a `Co-Authored-By` trailer naming the model that helped write them. Every reported number comes from a script in this repository and records the command, seed and commit that produced it, so nothing was typed, estimated or borrowed. I set the goals and constraints in the specification, I am accountable for the verification and the results, and I can explain and change any part of this code.
