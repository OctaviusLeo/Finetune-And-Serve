#!/usr/bin/env bash
# Run reported GPU stages back to back, each after the GPU-free check, stopping at the first failure.
# Usage (detached): bash scripts/bg.sh pipeline bash scripts/run_pipeline.sh <stage> [<stage> ...]
# Stages write only results/, runs/ and docs/, so later stages start from the same committed code.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
declare -A STAGES=(
  [dry_run]="python scripts/dry_run.py"
  [baselines_local]="python -m fts.baselines --baseline local"
  [baselines_hosted]="python -m fts.baselines --baseline hosted_few_shot"
  [sweep]="python -m fts.sweep sweep"
  [final]="python -m fts.sweep final"
  [determinism]="python scripts/check_train_determinism.py"
  [eval]="python -m fts.eval"
  [eval_rerun]="python -m fts.eval --rerun-check"
  [quantize]="python -m fts.quantize"
  [quant_quality]="python -m fts.quant_quality"
  [loadtest]="python scripts/loadtest.py"
)
for stage in "$@"; do
  if [ -z "${STAGES[$stage]:-}" ]; then echo "unknown stage: $stage" >&2; exit 2; fi
done
for stage in "$@"; do
  echo "=== $(date -Is) stage $stage: ${STAGES[$stage]}"
  # shellcheck disable=SC2086
  bash scripts/gpu_job.sh ${STAGES[$stage]}
  echo "=== $(date -Is) stage $stage done"
done
