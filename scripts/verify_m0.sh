#!/usr/bin/env bash
# M0 gate: unit tests, then the recorded environment, Dry_Run, hosted check, ledger and git attribute checks.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh -q
python scripts/verify.py m0
