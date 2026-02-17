#!/usr/bin/env bash
set -euo pipefail

# Example bootstrap script for local/SLURM profile usage.
# Copy and customize for your site environment.
if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate geryon || true
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate geryon || true
fi
