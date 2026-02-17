#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${GERYON_EXAMPLE_RUN_ID:-$(basename "$EXAMPLE_DIR")_$(date +%Y%m%d_%H%M%S)_$$}"

"$EXAMPLE_DIR/../../common/run_example.sh" \
  --example-dir "$EXAMPLE_DIR" \
  --out-dir "${GERYON_EXAMPLE_OUT_DIR:-$EXAMPLE_DIR/outputs}" \
  --run-id "$RUN_ID" \
  "$@"
