#!/usr/bin/env bash
# Run one GPU job after the GPU-free check (design.md "One GPU job at a time").
# Usage (detached): bash scripts/bg.sh <name> bash scripts/gpu_job.sh <command> [args...]
# Exits 1 without running the command when the GPU stays busy (scripts/gpu_check.py).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
python scripts/gpu_check.py
exec "$@"
