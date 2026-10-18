# Blockers

Recorded by the agent under the retry and GPU-busy rules in `.kiro/steering/environment.md`. Each blocker names the task, the command, its exit status, the log, and the suspected cause.

## 1. GPU shared with other projects (tasks 3.6 onward: every training and serving run)

- **Tasks:** 3.6 (sweep) and, behind it, 3.7, 3.8, 4.7, 5.5 and 5.8, with the gates 3.9, 4.8, 5.9 and 6.9. The Dry_Run (1.6) and the two local baselines (2.13) did complete.
- **Command:** `wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Finetune-And-Serve/scripts/bg.sh pipeline bash scripts/run_pipeline.sh baselines_local sweep final determinism eval eval_rerun` (restarted by `runs/retry_pipeline.sh`).
- **Exit status and log (`runs/logs/pipeline.log`):**
  - 2026-10-17 22:42 EDT: `scripts/gpu_check.py` exited 1 (`BLOCKED: GPU memory used 7087 MiB stayed above idle 2410 + 1024 MiB for 1200 s`) while the Grounded QA Ollama container held a model.
  - 23:24: the GPU freed and the local baselines finished. The sweep's first Run (config A, seed 1) passed the GPU-free check at 1341 MiB, trained 30 of 1440 optimizer steps (loss 1.42 → 0.59), then failed with `torch.AcceleratorError: CUDA error: out of memory`. The pipeline exited 1.
- **Suspected cause:** another project allocated GPU memory while the Run was training. The same Run fitted in the Dry_Run's worst case (torch peak 9326 MiB), and other sessions (Grounded QA's `gqa-ollama-1`, Learned Opponents training, Autonomy Test Harness simulations) used the GPU on and off all evening. This project's rules forbid stopping other projects' containers.
- **What unblocks it:** run the training stages while no other session uses the GPU (pause the other sessions' GPU work, or stop `gqa-ollama-1` yourself). Then restart from the sweep: `wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Finetune-And-Serve/scripts/bg.sh pipeline bash scripts/run_pipeline.sh sweep final determinism eval eval_rerun`. Finished Runs are reused, so nothing completed is retrained.

## Resolved

- **OpenAI key missing (1.6 hosted check, 2.13 hosted baseline):** resolved on 2026-10-18 when the owner chose TypeSafe's Jev as the hosted model instead of OpenAI. Its key, `TYPESAFE_API_KEY`, is set in the owner's environment and forwarded into WSL with `WSLENV=TYPESAFE_API_KEY/u`.
