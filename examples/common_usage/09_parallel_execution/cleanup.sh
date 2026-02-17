#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$EXAMPLE_DIR/../../common/cleanup_example.sh" "$EXAMPLE_DIR" "$@"
