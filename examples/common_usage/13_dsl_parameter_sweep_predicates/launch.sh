#!/usr/bin/env bash
set -euo pipefail
# Builds experiment.yaml via geryon.dsl, then runs validate -> plan -> run.

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${GERYON_EXAMPLE_OUT_DIR:-$EXAMPLE_DIR/outputs}"
RUN_ID="${GERYON_EXAMPLE_RUN_ID:-$(basename "$EXAMPLE_DIR")_$(date +%Y%m%d_%H%M%S)_$$}"
EXPERIMENT="$EXAMPLE_DIR/experiment.yaml"
PROFILES_FILE="$EXAMPLE_DIR/../configs/profiles.yaml"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$EXAMPLE_DIR"

CLI=()
if command -v geryon >/dev/null 2>&1; then
  CLI=(geryon)
elif command -v uv >/dev/null 2>&1; then
  CLI=(uv run geryon)
else
  CLI=("$PYTHON_BIN" -m geryon)
fi

BUILD_CMD=("$PYTHON_BIN")
if command -v uv >/dev/null 2>&1; then
  BUILD_CMD=(uv run python)
fi

echo "=== Step 1: Build experiment via DSL ==="
"${BUILD_CMD[@]}" "$EXAMPLE_DIR/build_experiment.py" --out "$EXPERIMENT"

echo ""
echo "=== Step 2: Validate ==="
"${CLI[@]}" validate-config \
  --experiment "$EXPERIMENT" \
  --show-diagnostics \
  --format json

echo ""
echo "=== Step 3: Plan ==="
PLAN_OUT=$("${CLI[@]}" plan \
  --experiment "$EXPERIMENT" \
  --out "$OUT_DIR" \
  --batch-size 8 \
  --run-id "$RUN_ID" \
  --format json)
echo "$PLAN_OUT"

RUN_ROOT=$(echo "$PLAN_OUT" | "$PYTHON_BIN" -c "import sys,json; print(json.load(sys.stdin)['run_root'])")

echo ""
echo "=== Step 4: Run ==="
"${CLI[@]}" run-local \
  --run "$RUN_ROOT" \
  --profiles-file "$PROFILES_FILE" \
  --profile local_throughput \
  --format table

echo ""
echo "=== Step 5: Status ==="
"${CLI[@]}" status \
  --run "$RUN_ROOT" \
  --by-pack model \
  --format table
