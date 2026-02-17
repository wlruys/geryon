#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SUITE_DIR/.." && pwd)"
MODE="${1:-}"

if [[ -n "$MODE" && "$MODE" != "--exec-only" ]]; then
  echo "usage: $0 [--exec-only]" >&2
  exit 2
fi

TARGETS=()
while IFS= read -r launch_file; do
  TARGETS+=("$(dirname "$launch_file")")
done < <(find "$SUITE_DIR" -mindepth 2 -maxdepth 2 -type f -name launch.sh | sort)

"$EXAMPLES_DIR/cleanup.sh" ${MODE:+$MODE} "${TARGETS[@]#$EXAMPLES_DIR/}"
