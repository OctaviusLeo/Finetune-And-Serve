#!/usr/bin/env bash
# M4 gate: unit tests, the full suite (including the docker serving test), then the recorded M4 results.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh -q
uv run --frozen pytest -q -p no:cacheprovider
python scripts/verify.py m4
