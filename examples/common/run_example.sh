#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_example.sh --example-dir <dir> [options]

Options:
  --example-dir <dir>       Example directory containing experiment.yaml (required)
  --out-dir <dir>           Output root (default: <example-dir>/outputs)
  --run-id <id>             Run id (default: <example>_<timestamp>_<pid>)
  --batch-size <n>          Planner batch size (default: 8)
  --profile <name>          Optional profile name for launch/run-local
  --profiles-file <path>    Optional profiles file path
  --run-set <name>          Optional run-set name
  --stages <csv>            Comma-separated stages: launch|validate|plan|run (default: launch)
  --expect-plan-failure     Expect plan stage to fail (requires --stages plan)
  --python-bin <path>       Python executable (default: $PYTHON_BIN or python3)
USAGE
}

EXAMPLE_DIR=""
OUT_DIR=""
RUN_ID=""
RUN_SET=""
BATCH_SIZE="8"
PROFILE_NAME=""
PROFILES_FILE=""
STAGES="launch"
EXPECT_PLAN_FAILURE=0
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --example-dir)
      EXAMPLE_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --run-set)
      RUN_SET="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --profile)
      PROFILE_NAME="$2"
      shift 2
      ;;
    --profiles-file)
      PROFILES_FILE="$2"
      shift 2
      ;;
    --stages)
      STAGES="$2"
      shift 2
      ;;
    --expect-plan-failure)
      EXPECT_PLAN_FAILURE=1
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$EXAMPLE_DIR" ]]; then
  echo "--example-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ "$EXPECT_PLAN_FAILURE" -eq 1 && "$STAGES" != "plan" ]]; then
  echo "--expect-plan-failure requires --stages plan" >&2
  exit 2
fi

EXAMPLE_DIR="$(cd "$EXAMPLE_DIR" && pwd)"
EXPERIMENT="$EXAMPLE_DIR/experiment.yaml"
if [[ ! -f "$EXPERIMENT" ]]; then
  echo "missing experiment file: $EXPERIMENT" >&2
  exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$EXAMPLE_DIR/outputs"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(basename "$EXAMPLE_DIR")_$(date +%Y%m%d_%H%M%S)_$$"
fi

RUN_ROOT="$OUT_DIR/runs/$RUN_ID"

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

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CLI=("$PYTHON_BIN" -m geryon)

run_validate() {
  local cmd=("${CLI[@]}" validate-config --experiment "$EXPERIMENT" --show-diagnostics)
  if [[ -n "$RUN_SET" ]]; then
    cmd+=(--run-set "$RUN_SET")
  fi
  "${cmd[@]}"
}

run_plan() {
  local cmd=("${CLI[@]}" plan --experiment "$EXPERIMENT" --out "$OUT_DIR" --batch-size "$BATCH_SIZE" --run-id "$RUN_ID")
  if [[ -n "$RUN_SET" ]]; then
    cmd+=(--run-set "$RUN_SET")
  fi

  if [[ "$EXPECT_PLAN_FAILURE" -eq 1 ]]; then
    if "${cmd[@]}"; then
      echo "expected planning to fail, but it succeeded" >&2
      exit 1
    fi
    return 0
  fi

  "${cmd[@]}"
}

run_local() {
  if [[ ! -d "$RUN_ROOT/plan" ]]; then
    echo "missing plan artifacts under $RUN_ROOT/plan; run the plan stage first" >&2
    exit 1
  fi

  local cmd=(
    "${CLI[@]}" run-local
    --run "$RUN_ROOT"
  )
  if [[ -n "$PROFILE_NAME" ]]; then
    cmd+=(--profile "$PROFILE_NAME")
  fi
  if [[ -n "$PROFILES_FILE" ]]; then
    cmd+=(--profiles-file "$PROFILES_FILE")
  fi
  "${cmd[@]}"
}

run_launch() {
  local cmd=(
    "${CLI[@]}" launch
    --experiment "$EXPERIMENT"
    --out "$OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --run-id "$RUN_ID"
    --backend local
  )
  if [[ -n "$RUN_SET" ]]; then
    cmd+=(--run-set "$RUN_SET")
  fi
  if [[ -n "$PROFILE_NAME" ]]; then
    cmd+=(--profile "$PROFILE_NAME")
  fi
  if [[ -n "$PROFILES_FILE" ]]; then
    cmd+=(--profiles-file "$PROFILES_FILE")
  fi
  "${cmd[@]}"
}

IFS=',' read -r -a STAGE_ITEMS <<< "$STAGES"
for stage in "${STAGE_ITEMS[@]}"; do
  stage="${stage//[[:space:]]/}"
  case "$stage" in
    validate)
      run_validate
      ;;
    launch)
      run_launch
      ;;
    plan)
      run_plan
      ;;
    run)
      run_local
      ;;
    "")
      ;;
    *)
      echo "unknown stage: $stage (allowed: launch,validate,plan,run)" >&2
      exit 2
      ;;
  esac
done
