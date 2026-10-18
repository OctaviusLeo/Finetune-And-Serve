#!/usr/bin/env bash
# M1 gate: unit tests, the full suite, a rebuild of the splits that must match the committed ones, then the data and baseline checks.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh -q
uv run --frozen pytest -q -p no:cacheprovider
python scripts/check_splits.py
python scripts/verify.py m1
