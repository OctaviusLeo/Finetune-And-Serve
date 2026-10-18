# Environment

Exact versions of every tool, library, API and model this project relies on, each with a verifiable artifact obtained on the target machine (Windows 11 desktop, NVIDIA RTX 3080, WSL2 `Ubuntu-24.04`).

## Machine and GPU

Filled in by the Dry_Run (task 1.6).

## Python environment

- Python: `3.12.3`; uv: `uv 0.13.1 (x86_64-unknown-linux-gnu)`; environment `~/.venvs/fts` created by `uv sync --frozen` from `uv.lock`
- torch `2.14.1+cu130` built for CUDA `13.0`, cuDNN `92400`; `torch.cuda.is_available()` = `True` on `NVIDIA GeForce RTX 3080`
- Command: `python scripts/record_env.py`
- Seed: `0`
- Commit: `51730ccdb8692958d912af9c72154214a8ba5df6`

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

Filled in by the Dry_Run (task 1.6).

## Hosted model and Spend_Cap

Filled in at task 1.6.

## Determinism notes

Filled in at tasks 3.8 and 4.7.

## Serving images

Filled in at task 5.5.
