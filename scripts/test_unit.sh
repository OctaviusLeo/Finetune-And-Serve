#!/usr/bin/env bash
# The hosted-CI test command: every test except the gpu- and docker-marked ones,
# with statement coverage over fts/ and scripts/, then the per-module coverage rule.
# Usage: bash scripts/test_unit.sh [extra pytest args...]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p runs
uv run --frozen pytest -m "not gpu and not docker" \
  --cov=fts --cov=scripts --cov-report=term --cov-report=json:runs/coverage.json "$@"
uv run --frozen python scripts/check_coverage.py runs/coverage.json
