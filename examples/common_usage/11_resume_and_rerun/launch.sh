#!/usr/bin/env bash
set -euo pipefail
# Demonstrates the full launch/recover lifecycle.

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${GERYON_EXAMPLE_OUT_DIR:-$EXAMPLE_DIR/outputs}"
RUN_ID="${GERYON_EXAMPLE_RUN_ID:-$(basename "$EXAMPLE_DIR")_$(date +%Y%m%d_%H%M%S)_$$}"
EXPERIMENT="$EXAMPLE_DIR/experiment.yaml"
PROFILES_FILE="$EXAMPLE_DIR/../configs/profiles.yaml"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$EXAMPLE_DIR"

echo "=== Step 1: Launch (validate + plan + run) ==="
LAUNCH_OUT=$("$PYTHON_BIN" -m geryon launch \
  --experiment "$EXPERIMENT" \
  --out "$OUT_DIR" \
  --batch-size 9 \
  --run-id "$RUN_ID" \
  --backend local \
  --profiles-file "$PROFILES_FILE" \
  --profile local_throughput \
  --format json)
echo "$LAUNCH_OUT"

RUN_ROOT=$(echo "$LAUNCH_OUT" | "$PYTHON_BIN" -c "import sys,json; print(json.load(sys.stdin)['plan']['run_root'])")

echo ""
echo "=== Step 2: Check status ==="
"$PYTHON_BIN" -m geryon status \
  --run "$RUN_ROOT" \
  --format table

echo ""
echo "=== Step 3: Recover failed configs ==="
RECOVER_OUT=$("$PYTHON_BIN" -m geryon recover \
  --run "$RUN_ROOT" \
  --status failed \
  --backend local \
  --profiles-file "$PROFILES_FILE" \
  --profile local_throughput \
  --format json 2>&1) || {
    echo "No failed configs to recover - all succeeded on first try!"
    exit 0
}
echo "$RECOVER_OUT"

echo ""
echo "=== Step 4: Final status ==="
"$PYTHON_BIN" -m geryon status \
  --run "$RUN_ROOT" \
  --by-pack experiment \
  --format table

echo ""
echo "=== Step 5: Collect results ==="
"$PYTHON_BIN" -m geryon collect \
  --run "$RUN_ROOT" \
  --format table
