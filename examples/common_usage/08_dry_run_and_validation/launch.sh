#!/usr/bin/env bash
set -euo pipefail
# Demonstrates validate-config, inspect-config, and plan --dry-run
# before running the actual experiment.

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${GERYON_EXAMPLE_OUT_DIR:-$EXAMPLE_DIR/outputs}"
RUN_ID="${GERYON_EXAMPLE_RUN_ID:-$(basename "$EXAMPLE_DIR")_$(date +%Y%m%d_%H%M%S)_$$}"
EXPERIMENT="$EXAMPLE_DIR/experiment.yaml"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$EXAMPLE_DIR"

echo "=== Step 1: Validate config ==="
"$PYTHON_BIN" -m geryon validate-config \
  --experiment "$EXPERIMENT" \
  --show-diagnostics \
  --format table

echo ""
echo "=== Step 2: Inspect composed config ==="
"$PYTHON_BIN" -m geryon inspect-config \
  --experiment "$EXPERIMENT"

echo ""
echo "=== Step 3: Dry-run plan (no artifacts written) ==="
"$PYTHON_BIN" -m geryon plan \
  --experiment "$EXPERIMENT" \
  --out "$OUT_DIR" \
  --batch-size 8 \
  --run-id "$RUN_ID" \
  --dry-run

echo ""
echo "=== Step 4: Actual launch ==="
"$PYTHON_BIN" -m geryon launch \
  --experiment "$EXPERIMENT" \
  --out "$OUT_DIR" \
  --batch-size 8 \
  --run-id "$RUN_ID" \
  --backend local \
  --format table
