"""Environment check and VRAM Dry_Run (Req 1.1-1.5, 1.8, 0.4).

Before any full training Run, prove that the Base_Model, PEFT method and
sequence length fit in VRAM: run a few LoRA optimizer steps at the worst case
among the sweep configs (``lora_r`` 16, the configured per-device batch, every sequence a
full ``max_seq_len`` tokens) and measure peak GPU memory two ways:

* ``torch.cuda.max_memory_reserved()`` - what PyTorch's allocator reserved;
* ``nvidia-smi`` sampled once per second - the whole GPU, other programs included.

Both must be at most the ``nvidia-smi`` total. The evidence (nvidia-smi output,
peaks, method, sequence length, batch size, model revision, command, seed and
commit) goes into docs/ENVIRONMENT.md and results/dry_run.json. Any failure -
over budget, CUDA out of memory, or Python older than 3.11 - is reported with
the measured value and the limit, recorded, and exits non-zero.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from fts import envdoc, gpu, repro, settings
from fts.lora import LoraSpec

MIN_PYTHON = (3, 11)
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_JSON = REPO_ROOT / "results" / "dry_run.json"
MACHINE_SECTION = "Machine and GPU"
DRY_RUN_SECTION = "Base model and VRAM Dry_Run"


@dataclass(frozen=True)
class DryRunConfig:
    model_name: str
    revision: str | None
    license: str
    dtype: str
    peft_method: str
    lora: LoraSpec
    max_seq_len: int
    per_device_batch_size: int
    grad_accum: int
    steps: int
    learning_rate: float
    seed: int
    gradient_checkpointing: bool


@dataclass
class DryRunResult:
    ok: bool
    failure: str | None = None
    peak_reserved_mib: float = 0.0
    peak_smi_mib: int = 0
    total_mib: int = 0
    trainable_params: int = 0
    total_params: int = 0
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    idle_mib: int | None = None  # memory other programs held before the run (runs/gpu_idle_mib.txt)


def headroom_line(result: DryRunResult) -> str:
    """How the torch peak compares with the memory left after other programs (informational)."""
    if result.idle_mib is None:
        return "- Other programs' GPU memory before the run: not recorded (no runs/gpu_idle_mib.txt)"
    combined = int(round(result.peak_reserved_mib)) + result.idle_mib
    verdict = (
        "fits in dedicated VRAM"
        if combined <= result.total_mib
        else "exceeds dedicated VRAM; under WSL2 the excess is served from shared system memory, which slows the job but does not fail it"
    )
    return (
        f"- Other programs held {result.idle_mib} MiB before the run (`runs/gpu_idle_mib.txt`); torch peak reserved + that = "
        f"{combined} MiB vs total {result.total_mib} MiB: {verdict}"
    )


def config_from_settings(model: settings.ModelSettings, train: settings.TrainSettings) -> DryRunConfig:
    dr = train.dry_run
    return DryRunConfig(
        model_name=model.name,
        revision=model.revision,
        license=model.license,
        dtype=model.dtype,
        peft_method=train.peft_method,
        lora=LoraSpec(r=dr.lora_r, alpha=dr.lora_alpha, dropout=train.lora_dropout, target_modules=train.target_modules),
        max_seq_len=train.max_seq_len,
        per_device_batch_size=train.per_device_batch_size,
        grad_accum=train.grad_accum,
        steps=dr.steps,
        learning_rate=dr.learning_rate,
        seed=dr.seed,
        gradient_checkpointing=train.gradient_checkpointing,
    )


def check_python(version_info: Sequence[int] = tuple(sys.version_info[:3])) -> str | None:
    """None when Python is new enough, otherwise the failed prerequisite."""
    found = tuple(version_info[:2])
    if found >= MIN_PYTHON:
        return None
    have = ".".join(str(v) for v in version_info[:3])
    return f"Python {have} is older than the required {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"


def budget_failure(peak_reserved_mib: float, peak_smi_mib: float, total_mib: float) -> str | None:
    """None when both peaks fit in the total, otherwise which peak exceeded it and by how much."""
    if total_mib <= 0:
        return f"no VRAM total available from nvidia-smi (total {total_mib} MiB)"
    over = []
    if not gpu.within_budget(peak_reserved_mib, total_mib):
        over.append(f"torch peak reserved {peak_reserved_mib:.0f} MiB")
    if not gpu.within_budget(peak_smi_mib, total_mib):
        over.append(f"nvidia-smi peak {peak_smi_mib:.0f} MiB")
    if over:
        return f"VRAM over budget: {' and '.join(over)} > total {total_mib:.0f} MiB"
    return None


def is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def oom_failure(exc: BaseException, peak_reserved_mib: float, total_mib: float) -> str:
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return (
        f"CUDA out of memory (limit {total_mib:.0f} MiB total VRAM; "
        f"torch peak reserved before the failure {peak_reserved_mib:.0f} MiB): {first_line}"
    )


def lora_steps(model: Any, cfg: DryRunConfig, device: str) -> list[float]:
    """Run ``cfg.steps`` optimizer steps on random full-length sequences; return the losses."""
    import torch

    vocab = int(model.config.vocab_size)
    gen = torch.Generator().manual_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    model.train()
    losses: list[float] = []
    shape = (cfg.per_device_batch_size, cfg.max_seq_len)
    for _ in range(cfg.steps):
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            ids = torch.randint(0, vocab, shape, generator=gen).to(device)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
            (out.loss / cfg.grad_accum).backward()
            step_loss += float(out.loss.detach()) / cfg.grad_accum
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(step_loss)
    return losses


def _default_model_factory(cfg: DryRunConfig, device: str) -> Any:
    from fts.lora import load_base_model

    return load_base_model(cfg.model_name, cfg.revision, cfg.dtype, device)


def run(
    cfg: DryRunConfig,
    device: str = "cuda",
    model_factory: Callable[[DryRunConfig, str], Any] = _default_model_factory,
    query: Callable[[], gpu.GpuMemory] = gpu.query_memory,
    sampler_interval_s: float = 1.0,
) -> DryRunResult:
    """Load, wrap with LoRA, run the steps while sampling nvidia-smi, and judge the budget."""
    import torch

    from fts.lora import apply_lora, count_parameters, set_determinism

    set_determinism(cfg.seed)
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def peak_reserved() -> float:
        return torch.cuda.max_memory_reserved() / 2**20 if cuda else 0.0

    result = DryRunResult(ok=False)
    start = time.monotonic()
    sampler = gpu.MemorySampler(interval_s=sampler_interval_s, query=query)
    try:
        with sampler:
            model = apply_lora(model_factory(cfg, device), cfg.lora, cfg.gradient_checkpointing)
            result.trainable_params, result.total_params = count_parameters(model)
            result.losses = lora_steps(model, cfg, device)
            if cuda:
                torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - an OOM is a measured outcome, everything else re-raises
        if not is_oom(exc):
            raise
        result.peak_reserved_mib = peak_reserved()
        result.peak_smi_mib = sampler.peak_mib
        result.total_mib = sampler.total_mib
        result.failure = oom_failure(exc, result.peak_reserved_mib, result.total_mib)
        result.seconds = time.monotonic() - start
        return result
    result.seconds = time.monotonic() - start
    result.peak_reserved_mib = peak_reserved()
    result.peak_smi_mib = sampler.peak_mib
    result.total_mib = sampler.total_mib
    result.failure = budget_failure(result.peak_reserved_mib, result.peak_smi_mib, result.total_mib)
    result.ok = result.failure is None
    return result


def render_machine_section(smi_text: str, stamp: repro.Stamp) -> str:
    total = gpu.parse_total_mib(smi_text)
    lines = [
        f"- GPU total VRAM reported by `nvidia-smi`: **{total} MiB**",
        f"- Python: `{platform.python_version()}` ({platform.platform()})",
        *envdoc.stamp_lines(stamp),
        "",
        "`nvidia-smi` output captured by the Dry_Run:",
        "",
        *envdoc.code_block(smi_text),
    ]
    return "\n".join(lines)


def render_dry_run_section(cfg: DryRunConfig, result: DryRunResult, stamp: repro.Stamp) -> str:
    status = "PASSED" if result.ok else f"FAILED: {result.failure}"
    lines = [
        f"- Result: **{status}**",
        f"- Base_Model: `{cfg.model_name}` at revision `{cfg.revision}`; license {cfg.license}",
        f"- PEFT method: {cfg.peft_method}; quantization during the Dry_Run: none (bf16 weights)",
        f"- LoRA: r={cfg.lora.r}, alpha={cfg.lora.alpha}, dropout={cfg.lora.dropout}, "
        f"targets {', '.join(cfg.lora.target_modules)}; gradient checkpointing {'on' if cfg.gradient_checkpointing else 'off'}",
        f"- Sequence length: {cfg.max_seq_len} tokens (every sequence padded to the full length)",
        f"- Per-device batch size: {cfg.per_device_batch_size}; gradient accumulation {cfg.grad_accum}",
        f"- Optimizer steps: {cfg.steps} (AdamW, learning rate {cfg.learning_rate})",
        f"- Trainable parameters: {result.trainable_params} of {result.total_params}",
        f"- Peak GPU memory, `torch.cuda.max_memory_reserved()`: **{result.peak_reserved_mib:.0f} MiB**",
        f"- Peak GPU memory, `nvidia-smi` sampled once per second (whole GPU): **{result.peak_smi_mib} MiB**",
        f"- `nvidia-smi` total: **{result.total_mib} MiB**",
        headroom_line(result),
        f"- Wall time: {result.seconds:.1f} s",
        *envdoc.stamp_lines(stamp),
    ]
    return "\n".join(lines)


def result_payload(cfg: DryRunConfig, result: DryRunResult) -> dict:
    cfg_dict = asdict(cfg)
    cfg_dict["lora"]["target_modules"] = list(cfg.lora.target_modules)
    return {"config": cfg_dict, "result": asdict(result)}


def record_failure(message: str, stamp: repro.Stamp, env_md: Path = envdoc.ENVIRONMENT_MD) -> None:
    body = "\n".join([f"- Result: **FAILED: {message}**", *envdoc.stamp_lines(stamp)])
    envdoc.replace_section(DRY_RUN_SECTION, body, env_md)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Environment check and VRAM Dry_Run.")
    p.add_argument("--per-device-batch", type=int, default=None, help="override per_device_batch_size (VRAM fallback)")
    p.add_argument("--grad-accum", type=int, default=None, help="override grad_accum (VRAM fallback)")
    p.add_argument("--max-seq-len", type=int, default=None, help="override max_seq_len")
    return p


def main(argv: list[str] | None = None, env_md: Path = envdoc.ENVIRONMENT_MD, result_json: Path = RESULT_JSON) -> int:
    args = build_parser().parse_args(argv)
    train = settings.load_train_settings()
    stamp = repro.start_run(seed=train.dry_run.seed)

    py_failure = check_python()
    if py_failure:
        print(f"FAIL: {py_failure}", file=sys.stderr)
        record_failure(py_failure, stamp, env_md)
        return 1

    model = settings.pin_revision()
    cfg = config_from_settings(model, train)
    overrides = {
        k: v
        for k, v in (("per_device_batch_size", args.per_device_batch), ("grad_accum", args.grad_accum), ("max_seq_len", args.max_seq_len))
        if v is not None
    }
    cfg = replace(cfg, **overrides)

    smi_text = gpu.run_nvidia_smi()
    envdoc.replace_section(MACHINE_SECTION, render_machine_section(smi_text, stamp), env_md)
    print(f"Dry_Run: {cfg.model_name}@{cfg.revision}, batch {cfg.per_device_batch_size}, seq {cfg.max_seq_len}", flush=True)
    result = run(cfg)
    idle_file = REPO_ROOT / "runs" / "gpu_idle_mib.txt"
    if idle_file.exists():
        result.idle_mib = gpu.read_idle_mib(idle_file)
    envdoc.replace_section(DRY_RUN_SECTION, render_dry_run_section(cfg, result, stamp), env_md)
    repro.write_json(result_json, result_payload(cfg, result), stamp)
    print(
        f"peak reserved {result.peak_reserved_mib:.0f} MiB, nvidia-smi peak {result.peak_smi_mib} MiB, "
        f"total {result.total_mib} MiB, {result.seconds:.1f} s"
    )
    if not result.ok:
        print(f"FAIL: {result.failure}", file=sys.stderr)
        return 1
    print("Dry_Run PASSED")
    return 0
