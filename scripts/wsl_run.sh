#!/usr/bin/env bash
# Run one command in WSL from the repository root with the project's environment.
# From PowerShell: wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Finetune-And-Serve/scripts/wsl_run.sh <command> [args...]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/fts"
export FTS_ARTIFACTS="$HOME/fts-artifacts"
export PATH="$HOME/.local/bin:$UV_PROJECT_ENVIRONMENT/bin:$PATH"
secrets="$HOME/.config/fts/secrets.env"
if [ -f "$secrets" ]; then set -a; . "$secrets"; set +a; fi
exec "$@"
