# Blockers

Recorded by the agent under the retry and GPU-busy rules in `.kiro/steering/environment.md`. Each blocker names the task, the command, its exit status, the log, and the suspected cause.

## 1. GPU held by other projects (tasks 2.13 onward: every GPU run)

- **Task:** 2.13 (run the baselines). Every later GPU task is waiting behind it: 3.6, 3.7, 3.8, 4.7, 5.5 and 5.8, and with them the gates 2.14, 3.9, 4.8, 5.9 and 6.9.
- **Command:** `wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Finetune-And-Serve/scripts/bg.sh pipeline bash scripts/run_pipeline.sh dry_run baselines_local sweep final determinism eval eval_rerun`, which runs `bash scripts/gpu_job.sh python -m fts.baselines --baseline local` for this stage.
- **Exit status:** 1 from `scripts/gpu_check.py` (`BLOCKED: GPU memory used 7087 MiB stayed above idle 2410 + 1024 MiB for 1200 s`) at 2026-10-17 22:42 EDT. The Dry_Run stage before it passed.
- **Log:** `runs/logs/pipeline.log`
- **Suspected cause:** other projects' processes hold GPU memory for long periods. During the 22:21–22:42 wait the GPU sat at about 7,090 MiB with 0% utilization. At that time the Grounded QA Compose project was running (`gqa-ollama-1`, whose `llama-server` keeps a model loaded while `gqa-api-1` runs), and earlier in the evening Learned Opponents training and Autonomy Test Harness simulations also used the GPU. This project's rules forbid stopping other projects' containers, and running the 1.5B bf16 model next to them would spill into shared memory under WSL2 and distort timings.
- **What unblocks it:** free the GPU (stop or pause the other sessions' GPU work, or stop `gqa-ollama-1` yourself) so `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` reads at most 3,434 MiB. The pipeline retries on its own every 15 minutes for about two hours (`runs/logs/pipeline.log`). After that, restart it with the command above.

## 2. OpenAI key missing (tasks 1.6 hosted check, 2.13 hosted baseline)

- **Task:** 1.6 (hosted API proof) and the hosted few-shot baseline in 2.13. The M0 gate (1.7) and M1 gate (2.14) need both.
- **Command:** `python scripts/hosted_check.py` and `python -m fts.baselines --baseline hosted_few_shot` (both through `scripts/wsl_run.sh`).
- **Exit status:** not run. `~/.config/fts/secrets.env` does not exist in `Ubuntu-24.04`, so `OPENAI_API_KEY` is unset, and both commands would exit 2 before sending any request (tested in `tests/test_hosted.py` and `tests/test_baselines.py`).
- **Log:** none (the commands were not started).
- **Suspected cause:** guide steps 3–6 (OpenAI credit and API key) were not completed.
- **What unblocks it:** in the Ubuntu window, run `mkdir -p ~/.config/fts && chmod 700 ~/.config/fts && read -rsp "Paste your OpenAI API key, then press Enter: " k && echo && printf 'OPENAI_API_KEY=%q\n' "$k" > ~/.config/fts/secrets.env && chmod 600 ~/.config/fts/secrets.env; unset k`. Then run `python scripts/hosted_check.py` and `python -m fts.baselines --baseline hosted_few_shot`.
