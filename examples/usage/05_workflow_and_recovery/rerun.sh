#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${GERYON_EXAMPLE_OUT_DIR:-$EXAMPLE_DIR/outputs}"
RUN_ID="${GERYON_EXAMPLE_RUN_ID:-05_workflow_and_recovery_demo}"
RUN_ROOT="$OUT_DIR/runs/$RUN_ID"
PYTHON_BIN="${PYTHON_BIN:-python3}"

resolve_repo_root() {
  local start="$1"
  local current="$start"
  while [[ "$current" != "/" ]]; do
    if [[ -f "$current/pyproject.toml" && -d "$current/src/geryon" ]]; then
      printf "%s\n" "$current"
      return 0
    fi
    current="$(dirname "$current")"
  done
  return 1
}

if ! REPO_ROOT="$(resolve_repo_root "$EXAMPLE_DIR")"; then
  echo "failed to locate repository root from $EXAMPLE_DIR" >&2
  exit 1
fi

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "missing run root: $RUN_ROOT" >&2
  echo "run ./launch.sh first" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CLI=("$PYTHON_BIN" -m geryon)

echo "Status before rerun"
"${CLI[@]}" status --run "$RUN_ROOT"

echo "Create retry file for terminated configs"
"${CLI[@]}" rerun --run "$RUN_ROOT" --status terminated

RETRY_FILE="$(ls -1t "$RUN_ROOT"/exec/retries/retry_*.json 2>/dev/null | head -n 1 || true)"
if [[ -z "$RETRY_FILE" ]]; then
  echo "no retry file found under $RUN_ROOT/exec/retries" >&2
  exit 1
fi

echo "Run retry selection with relaxed timeout profile"
"${CLI[@]}" run-local \
  --run "$RUN_ROOT" \
  --profile local_recover \
  --profiles-file "$EXAMPLE_DIR/profiles.yaml" \
  --retry-file "$RETRY_FILE"
