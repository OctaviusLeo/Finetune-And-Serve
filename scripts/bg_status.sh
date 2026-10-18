#!/usr/bin/env bash
# Report a detached job's state: RUNNING, DONE <exit status>, or LOST (gone without an exit file).
# Usage: bash scripts/bg_status.sh <name> [lines]   (also prints the last [lines] log lines, default 20)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
name="$1"; lines="${2:-20}"; d=runs/logs
alive() { local s; s="$(ps -o stat= -p "$(cat "$d/$name.pid" 2>/dev/null)" 2>/dev/null | tr -d ' ')"; [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }
if [ -f "$d/$name.exit" ]; then echo "DONE $(cat "$d/$name.exit")"
elif alive; then echo "RUNNING"
elif [ -f "$d/$name.exit" ]; then echo "DONE $(cat "$d/$name.exit")"
else echo "LOST"
fi
tail -n "$lines" "$d/$name.log" 2>/dev/null || true
