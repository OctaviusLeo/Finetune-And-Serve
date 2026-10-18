#!/usr/bin/env bash
# Start a long job detached so it survives the calling terminal.
# Usage: bash scripts/bg.sh <name> <command> [args...]
# Writes runs/logs/<name>.log, runs/logs/<name>.pid, and, when the job ends, runs/logs/<name>.exit (its exit status).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
name="$1"; shift
mkdir -p runs/logs
rm -f "runs/logs/$name.exit" "runs/logs/$name.pid"
setsid nohup bash -c 'n="$1"; shift; echo "$$" > "runs/logs/$n.pid"; bash scripts/wsl_run.sh "$@"; echo "$?" > "runs/logs/$n.exit"' \
  _ "$name" "$@" > "runs/logs/$name.log" 2>&1 < /dev/null &
for _ in $(seq 50); do [ -s "runs/logs/$name.pid" ] && break; sleep 0.1; done
echo "started $name (pid $(cat "runs/logs/$name.pid" 2>/dev/null || echo unknown)); log: runs/logs/$name.log"
