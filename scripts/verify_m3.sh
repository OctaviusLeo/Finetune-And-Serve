#!/usr/bin/env bash
# M3 gate: unit tests, the full suite, then the recorded M3 results (python scripts/verify.py m3).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh -q
uv run --frozen pytest -q -p no:cacheprovider
python scripts/verify.py m3
