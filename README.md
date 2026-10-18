# Fine-Tune and Serve

**This repository is private.** It has no public license, no published package, and no public release, and the fine-tuned weights and LoRA adapters are not published.

Fine-Tune and Serve fine-tunes Qwen2.5-1.5B-Instruct with LoRA on one narrow task (English intent-and-slot extraction to JSON on the MASSIVE dataset), compares it with prompting baselines on the same Test split and metric, then quantizes it with llama.cpp (GGUF Q4_K_M) and serves it behind an OpenAI-compatible endpoint with a measured load test.

This README is regenerated from `docs/README.template.md` by `scripts/build_docs.py` once the results exist (task 6.8). Every number it reports comes from a generated file under `results/`.

## Running things

All Python, Docker and shell commands run inside WSL `Ubuntu-24.04` through the wrapper, for example:

```
wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Finetune-And-Serve/scripts/wsl_run.sh bash scripts/test_unit.sh
```

`bash scripts/test_unit.sh` runs every unit and property test that needs no GPU or Docker, with statement coverage over `fts/` and `scripts/`, and enforces at least 80% coverage per non-trivial module.
