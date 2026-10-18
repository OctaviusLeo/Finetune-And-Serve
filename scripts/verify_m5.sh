#!/usr/bin/env bash
# End-to-end gate: unit tests, the full suite, the Smoke_Evaluation, the traceability check, then the M5 checks.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh -q
uv run --frozen pytest -q -p no:cacheprovider
python -m fts.smoke_eval
python scripts/check_traceability.py
python scripts/verify.py m5
