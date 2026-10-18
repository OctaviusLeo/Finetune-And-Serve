# Sweep rationale

Written before any sweep Run (task 3.3); the configurations are in `config/sweep.yaml`.

| Config | `lora_r` | `lora_alpha` | learning rate |
| --- | --- | --- | --- |
| A | 8 | 16 | 2e-4 |
| B | 16 | 32 | 2e-4 |
| C | 16 | 32 | 1e-4 |

**Why these three.** LoRA rank and learning rate are the settings quality is most sensitive to. A versus B isolates the rank at a fixed learning rate (with `lora_alpha` = 2 x rank, so the adapter's scale `alpha / r` stays at 2), and B versus C isolates the learning rate at a fixed rank. Everything else is fixed for every Run in `config/train.yaml`: LoRA on bf16 base weights over all attention and MLP projections, dropout 0.05, `max_seq_len` 256, per-device batch 4 with gradient accumulation 4 (effective batch 16; batch 8 x 2 before the VRAM fallback of 2026-10-18, see `config/train.yaml`), 2 epochs, linear decay with 3% warmup, gradient clipping 1.0, loss on the JSON tokens only.

**Selection.** All three sweep Runs use seed 1 and are trained on the Train_Split only. The Final_Configuration is the config with the highest Validation_Split field-level F1; a tie goes to the config listed earlier. The Test_Split is not used for any training or selection decision.

**Noise.** With 2,033 Validation examples and a per-example F1 standard deviation near 0.25 (a planning assumption, not a measurement), a 95% interval half-width is about 1.96 x 0.25 / sqrt(2033), roughly 0.011, so Validation differences smaller than that are within noise. Each Run's measured Validation CI is recorded in `results/runs/<run_id>.json`.

**Final Runs.** The Final_Configuration is trained with seeds 1, 2 and 3; its seed-1 Run is the sweep Run, so two new Runs are trained. The serving seed is the one with the highest Validation F1, ties going to the lowest seed.
