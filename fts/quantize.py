"""Merge the serving seed's Adapter and quantize it with llama.cpp (Req 6.1).

1. Load the bf16 Base_Model on the CPU, apply the serving seed's Adapter with
   peft, ``merge_and_unload``, and save the merged model and tokenizer to
   ``~/fts-artifacts/models/merged/``.
2. Convert it to ``ft-f16.gguf`` with the pinned llama.cpp ``full-<build>`` image
   (``--convert`` runs ``convert_hf_to_gguf.py``).
3. Quantize that to ``ft-q4_k_m.gguf`` (``--quantize`` runs ``llama-quantize``).

Both containers mount ``~/fts-artifacts/models`` and run as the calling user.
results/quantization.json records each file's method, nominal bit width, size,
the sizes and bits per weight that llama.cpp reports, and the image digest.

    python -m fts.quantize
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from fts import repro, settings

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get("FTS_ARTIFACTS", Path.home() / "fts-artifacts"))

Runner = Callable[[Sequence[str]], str]


@dataclass(frozen=True)
class QuantizedModel:
    path: str  # relative to ~/fts-artifacts
    method: str  # "gguf-f16" (pre-quant reference) | "gguf-q4_k_m"
    bit_width: int  # nominal: 16 | 4


def run_command(cmd: Sequence[str]) -> str:
    """Run a command, returning stdout+stderr; raises CalledProcessError with the output on failure."""
    out = subprocess.run(list(cmd), capture_output=True, text=True)
    combined = (out.stdout or "") + (out.stderr or "")
    if out.returncode != 0:
        raise subprocess.CalledProcessError(out.returncode, list(cmd), output=combined)
    return combined


def docker_base(image: str, models_dir: Path, uid: int, gid: int) -> list[str]:
    return ["docker", "run", "--rm", "--user", f"{uid}:{gid}", "-e", "HOME=/tmp", "-v", f"{models_dir}:/models", image]


def convert_command(image: str, models_dir: Path, uid: int, gid: int) -> list[str]:
    return docker_base(image, models_dir, uid, gid) + ["--convert", "/models/merged", "--outtype", "f16", "--outfile", "/models/ft-f16.gguf"]


def quantize_command(image: str, models_dir: Path, uid: int, gid: int) -> list[str]:
    return docker_base(image, models_dir, uid, gid) + ["--quantize", "/models/ft-f16.gguf", "/models/ft-q4_k_m.gguf", "Q4_K_M"]


def image_digest(image: str, runner: Runner = run_command) -> str:
    return runner(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image]).strip()


def parse_quantize_log(text: str) -> dict:
    """Sizes and bits per weight that llama-quantize prints (None when a value is absent)."""
    def first(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    bpw = [float(x) for x in re.findall(r"\(\s*([\d.]+)\s*BPW\s*\)", text)]
    return {
        "model_size_mb": first(r"model size\s*=\s*([\d.]+)\s*MB"),
        "quant_size_mb": first(r"quant size\s*=\s*([\d.]+)\s*MB"),
        "input_bpw": bpw[0] if bpw else None,
        "reported_bpw": bpw,
    }


def bits_per_weight(size_bytes: int, n_params: int) -> float:
    return size_bytes * 8 / n_params


def merge_adapter(model_settings: settings.ModelSettings, adapter_dir: Path, out_dir: Path) -> int:
    """Merge the Adapter into the bf16 Base_Model on the CPU and save model + tokenizer; returns the parameter count."""
    from peft import PeftModel

    from fts.lora import load_base_model, load_tokenizer

    base = load_base_model(model_settings.name, model_settings.revision, model_settings.dtype, "cpu")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    load_tokenizer(model_settings.name, model_settings.revision).save_pretrained(str(out_dir))
    return sum(p.numel() for p in merged.parameters())


def quantize(model_settings: settings.ModelSettings, serving_run: dict, serve_cfg: dict, artifacts: Path = ARTIFACTS,
             runner: Runner = run_command, merge: Callable[[settings.ModelSettings, Path, Path], int] = merge_adapter,
             uid: int | None = None, gid: int | None = None, log: Callable[[str], None] = lambda m: print(m, flush=True)) -> tuple[list[QuantizedModel], dict]:
    """Merge, convert and quantize; returns the two QuantizedModels and the details recorded in results/quantization.json."""
    models_dir = artifacts / "models"
    uid = os.getuid() if uid is None else uid
    gid = os.getgid() if gid is None else gid
    image = serve_cfg["full_image"]
    log(f"merging {serving_run['adapter_path']} into {model_settings.name}")
    n_params = merge(model_settings, artifacts / serving_run["adapter_path"], models_dir / "merged")
    log("converting to GGUF F16")
    convert_log = runner(convert_command(image, models_dir, uid, gid))
    log("quantizing to Q4_K_M")
    quant_log = runner(quantize_command(image, models_dir, uid, gid))
    parsed = parse_quantize_log(quant_log)
    files = {}
    models = [QuantizedModel("models/ft-f16.gguf", "gguf-f16", 16), QuantizedModel("models/ft-q4_k_m.gguf", "gguf-q4_k_m", 4)]
    for qm in models:
        size = (artifacts / qm.path).stat().st_size
        files[qm.method] = asdict(qm) | {"size_bytes": size, "bits_per_weight_from_size": bits_per_weight(size, n_params)}
    details = {
        "serving_run": serving_run["run_id"],
        "adapter_path": serving_run["adapter_path"],
        "merged_parameters": n_params,
        "image": image,
        "image_digest": image_digest(image, runner),
        "llama_cpp_build": serve_cfg["llama_cpp_build"],
        "commands": {"convert": convert_command(image, models_dir, uid, gid)[-7:], "quantize": quantize_command(image, models_dir, uid, gid)[-5:]},
        "files": files,
        "llama_quantize": parsed,
        "convert_log_tail": convert_log.strip().splitlines()[-5:],
        "quantize_log_tail": quant_log.strip().splitlines()[-12:],
    }
    return models, details


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, artifacts: Path = ARTIFACTS, runner: Runner = run_command,
         merge: Callable[[settings.ModelSettings, Path, Path], int] = merge_adapter) -> int:
    final_path = root / "results" / "final_runs.json"
    if not final_path.exists():
        print("FAIL: results/final_runs.json is missing", file=sys.stderr)
        return 1
    final = json.loads(final_path.read_text(encoding="utf-8"))
    stamp = repro.start_run(seed=final["serving_seed"])
    run = next(r for r in final["runs"] if r["seed"] == final["serving_seed"])
    serve_cfg = settings.load_yaml(root / "config" / "serve.yaml")
    try:
        models, details = quantize(settings.load_model_settings(root / "config" / "model.yaml"), run, serve_cfg, artifacts, runner, merge)
    except subprocess.CalledProcessError as exc:
        print(f"FAIL: {' '.join(exc.cmd[:6])} ... exited {exc.returncode}\n{(exc.output or '')[-2000:]}", file=sys.stderr)
        return 1
    repro.write_json(root / "results" / "quantization.json", {"models": [asdict(m) for m in models], **details}, stamp)
    for m in models:
        print(f"{m.method}: {details['files'][m.method]['size_bytes']} bytes, {details['files'][m.method]['bits_per_weight_from_size']:.2f} bits/weight by size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
