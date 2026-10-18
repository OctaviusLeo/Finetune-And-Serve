#!/usr/bin/env bash
# M2 gate: unit tests, the full suite, then the recorded M2 results (python scripts/verify.py m2).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh -q
uv run --frozen pytest -q -p no:cacheprovider -m "not docker"  # the docker serving test first runs at the M4 gate (task 5.4)
python scripts/verify.py m2
