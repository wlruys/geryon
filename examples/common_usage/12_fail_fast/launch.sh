#!/usr/bin/env bash
set -euo pipefail
# Demonstrates fail-fast: stop after 3 failures.

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${GERYON_EXAMPLE_OUT_DIR:-$EXAMPLE_DIR/outputs}"
RUN_ID="${GERYON_EXAMPLE_RUN_ID:-$(basename "$EXAMPLE_DIR")_$(date +%Y%m%d_%H%M%S)_$$}"
EXPERIMENT="$EXAMPLE_DIR/experiment.yaml"
PROFILES_FILE="$EXAMPLE_DIR/../configs/profiles.yaml"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$EXAMPLE_DIR"

echo "=== Plan ==="
PLAN_OUT=$("$PYTHON_BIN" -m geryon plan \
  --experiment "$EXPERIMENT" \
  --out "$OUT_DIR" \
  --batch-size 8 \
  --run-id "$RUN_ID" \
  --format json)
echo "$PLAN_OUT"

RUN_ROOT=$(echo "$PLAN_OUT" | "$PYTHON_BIN" -c "import sys,json; print(json.load(sys.stdin)['run_root'])")

echo ""
echo "=== Run with local_fail_fast profile (stops after 3 failures) ==="
"$PYTHON_BIN" -m geryon run-local \
  --run "$RUN_ROOT" \
  --profiles-file "$PROFILES_FILE" \
  --profile local_fail_fast \
  --format json

echo ""
echo "=== Status (shows skipped configs) ==="
"$PYTHON_BIN" -m geryon status \
  --run "$RUN_ROOT" \
  --format table
