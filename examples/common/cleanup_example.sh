#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example_dir> [--exec-only]" >&2
  exit 2
fi

EXAMPLE_DIR="$1"
MODE="${2:-}"
if [[ -n "$MODE" && "$MODE" != "--exec-only" ]]; then
  echo "unknown mode: $MODE" >&2
  echo "usage: $0 <example_dir> [--exec-only]" >&2
  exit 2
fi

EXAMPLE_DIR="$(cd "$EXAMPLE_DIR" && pwd)"
OUT_DIR="$EXAMPLE_DIR/outputs"

if [[ "$MODE" == "--exec-only" ]]; then
  if [[ -d "$OUT_DIR/runs" ]]; then
    while IFS= read -r -d '' RUN_DIR; do
      if [[ -d "$RUN_DIR/exec" ]]; then
        rm -rf "$RUN_DIR/exec"
        echo "removed: $RUN_DIR/exec"
      fi
    done < <(find "$OUT_DIR/runs" -mindepth 1 -maxdepth 1 -type d -print0)
  fi
else
  if [[ -d "$OUT_DIR" ]]; then
    rm -rf "$OUT_DIR"
    echo "removed: $OUT_DIR"
  fi
fi

find "$EXAMPLE_DIR" -maxdepth 1 -type f \
  \( -name "*.stdout.log" -o -name "*.stderr.log" -o -name "run.log" \) \
  -print -delete | sed 's/^/removed: /'
