"""Fine-Tune and Serve (fts).

LoRA fine-tuning of Qwen2.5-1.5B-Instruct on MASSIVE intent/slot JSON, evaluation
with bootstrap confidence intervals, llama.cpp quantization and serving, and a load
test. Pure logic (splits, metric, bootstrap, spend ledger, percentiles,
traceability) imports no heavy ML library at module scope, so it runs on a CPU-only
CI runner; torch, transformers and peft are imported inside the GPU functions.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
