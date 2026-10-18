# Environment

Exact versions of every tool, library, API and model this project relies on, each with a verifiable artifact obtained on the target machine (Windows 11 desktop, NVIDIA RTX 3080, WSL2 `Ubuntu-24.04`).

## Machine and GPU

- GPU total VRAM reported by `nvidia-smi`: **10240 MiB**
- Python: `3.12.3` (Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39)
- Command: `python scripts/dry_run.py`
- Seed: `0`
- Commit: `f855c93c8b34d23182093943a50ae5d8ef643543`

`nvidia-smi` output captured by the Dry_Run:

```text
Sat Oct 17 22:18:36 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 615.78.05              KMD Version: 617.42        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080        On  |   00000000:2B:00.0  On |                  N/A |
| 30%   57C    P0            141W /  368W |    3338MiB /  10240MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              27      G   /Xwayland                             N/A      |
|    0   N/A  N/A              58      G   /Xwayland                             N/A      |
+-----------------------------------------------------------------------------------------+
```

## Python environment

- Python: `3.12.3`; uv: `uv 0.13.1 (x86_64-unknown-linux-gnu)`; environment `~/.venvs/fts` created by `uv sync --frozen` from `uv.lock`
- torch `2.14.1+cu130` built for CUDA `13.0`, cuDNN `92400`; `torch.cuda.is_available()` = `True` on `NVIDIA GeForce RTX 3080`
- Command: `python scripts/record_env.py`
- Seed: `0`
- Commit: `b58de542d8899e6044569863291ef72f280008fd`

| Package | Installed version | License |
| --- | --- | --- |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| transformers | 5.20.1 | Apache 2.0 License |
| peft | 0.21.2 | Apache |
| accelerate | 1.15.0 | Apache |
| huggingface-hub | 2.2.0 | Apache-2.0 |
| safetensors | 0.8.0 | Apache Software License |
| tokenizers | 0.23.3 | Apache Software License |

Verifiable artifact: `uv pip show` on this machine for the fine-tuning stack:

```text
Name: accelerate
Version: 1.15.0
Location: /home/octaviusleo/.venvs/fts/lib/python3.12/site-packages
Requires: huggingface-hub, numpy, packaging, psutil, pyyaml, safetensors, torch
Required-by: finetune-and-serve, peft
---
Name: huggingface-hub
Version: 2.2.0
Location: /home/octaviusleo/.venvs/fts/lib/python3.12/site-packages
Requires: click, filelock, fsspec, hf-xet, httpx2, packaging, pyyaml, tqdm, typing-extensions
Required-by: accelerate, peft, tokenizers, transformers
---
Name: peft
Version: 0.21.2
Location: /home/octaviusleo/.venvs/fts/lib/python3.12/site-packages
Requires: accelerate, huggingface-hub, numpy, packaging, psutil, pyyaml, safetensors, torch, tqdm, transformers
Required-by: finetune-and-serve
---
Name: safetensors
Version: 0.8.0
Location: /home/octaviusleo/.venvs/fts/lib/python3.12/site-packages
Requires:
Required-by: accelerate, peft, transformers
---
Name: tokenizers
Version: 0.23.3
Location: /home/octaviusleo/.venvs/fts/lib/python3.12/site-packages
Requires: huggingface-hub
Required-by: transformers
---
Name: torch
Version: 2.14.1
Location: /home/octaviusleo/.venvs/fts/lib/python3.12/site-packages
Requires: cuda-bindings, cuda-toolkit, filelock, fsspec, jinja2, networkx, nvidia-cudnn-cu13, nvidia-cusparselt-cu13, nvidia-nccl-cu13, nvidia-nvshmem-cu13, setuptools, sympy, triton, typing-extensions
Required-by: accelerate, finetune-and-serve, peft
---
Name: transformers
Version: 5.20.1
Location: /home/octaviusleo/.venvs/fts/lib/python3.12/site-packages
Requires: huggingface-hub, numpy, packaging, pyyaml, regex, safetensors, tokenizers, tqdm, typer
Required-by: finetune-and-serve, peft
```

## Base model and VRAM Dry_Run

- Result: **PASSED**
- Base_Model: `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; license Apache-2.0
- PEFT method: LoRA on bf16 base weights (no quantization); quantization during the Dry_Run: none (bf16 weights)
- LoRA: r=16, alpha=32, dropout=0.05, targets q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj; gradient checkpointing on
- Sequence length: 256 tokens (every sequence padded to the full length)
- Per-device batch size: 8; gradient accumulation 2
- Optimizer steps: 20 (AdamW, learning rate 0.0002)
- Trainable parameters: 18464768 of 1562179072
- Peak GPU memory, `torch.cuda.max_memory_reserved()`: **9326 MiB**
- Peak GPU memory, `nvidia-smi` sampled once per second (whole GPU): **10020 MiB**
- `nvidia-smi` total: **10240 MiB**
- Other programs held 2410 MiB before the run (`runs/gpu_idle_mib.txt`); torch peak reserved + that = 11736 MiB vs total 10240 MiB: exceeds dedicated VRAM; under WSL2 the excess is served from shared system memory, which slows the job but does not fail it
- Wall time: 169.5 s
- Command: `python scripts/dry_run.py`
- Seed: `0`
- Commit: `f855c93c8b34d23182093943a50ae5d8ef643543`

## Hosted model and Spend_Cap

- Hosted_Model: TypeSafe Jev, model `jev-1.13.0` (pinned version) via `POST https://api.typesafe.ai/v1/systemone`; key read from `TYPESAFE_API_KEY`
- API check: the response's `model` field was `jev-1.13.0`; P(yes) for the check question 0.98
- Check tokens: 292 input, 21 output; cost $0.00001226
- Prices (USD per 1M tokens, https://docs.typesafe.ai/models, accessed 2026-10-18): input 0.042, output 0.0 (output is free); the ledger charges input tokens at 0.042
- **Spend_Cap: $2.00**; ledger `results/hosted_ledger.jsonl` total after the check: $0.00001226 over 1 request(s)
- Command: `python scripts/hosted_check.py`
- Seed: `0`
- Commit: `23470e7f41fde69d0c9f36ac64368c094753f445`

## Determinism notes

Filled in at tasks 3.8 and 4.7.

## Serving images

Filled in at task 5.5.

## Determinism notes: training

- Training determinism check (30 optimizer steps of config B, seed 1, run twice): **identical**; largest loss difference 0 (tolerance 1e-06)
- Nondeterministic kernels named by PyTorch's warn-only determinism mode:
  - `Memory Efficient attention defaults to a non-deterministic algorithm. To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False). (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/transformers/cuda/attention_backward.cu:910.)`
- Command: `python scripts/check_train_determinism.py`
- Seed: `1`
- Commit: `b7212565dd32fd119b0881d99d5d351d6facf4f9`

## Determinism notes: evaluation

- Evaluation re-run of B-seed2 on the Test_Split: **identical**; 0 prediction(s) differed; Test F1 0.850752 then 0.850752
- Command: `python -m fts.eval --rerun-check`
- Seed: `0`
- Commit: `960b5357a4ee8f6fe9679a407c0fb997836f5078`
