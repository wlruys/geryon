#!/usr/bin/env bash
set -euo pipefail

EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$EXAMPLES_DIR/common/cleanup_example.sh"
MODE=""

if [[ "${1:-}" == "--exec-only" ]]; then
  MODE="--exec-only"
  shift
fi

TARGETS=()
if [[ $# -gt 0 ]]; then
  for value in "$@"; do
    TARGETS+=("$EXAMPLES_DIR/$value")
  done
else
  while IFS= read -r launch_file; do
    TARGETS+=("$(dirname "$launch_file")")
  done < <(find "$EXAMPLES_DIR" -mindepth 2 -maxdepth 3 -type f -name launch.sh | sort)
fi

for target in "${TARGETS[@]}"; do
  if [[ ! -d "$target" ]]; then
    echo "skip missing: $target"
    continue
  fi
  "$HELPER" "$target" ${MODE:+$MODE}
done
