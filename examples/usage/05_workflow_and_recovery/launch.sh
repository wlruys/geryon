#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${GERYON_EXAMPLE_RUN_ID:-05_workflow_and_recovery_demo}"

"$EXAMPLE_DIR/../../common/run_example.sh" \
  --example-dir "$EXAMPLE_DIR" \
  --out-dir "${GERYON_EXAMPLE_OUT_DIR:-$EXAMPLE_DIR/outputs}" \
  --run-id "$RUN_ID" \
  --batch-size "${GERYON_EXAMPLE_BATCH_SIZE:-4}" \
  --profile "${GERYON_EXAMPLE_PROFILE:-local_fail_fast}" \
  --profiles-file "$EXAMPLE_DIR/profiles.yaml" \
  "$@"
