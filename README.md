# Geryon

Geryon is a deterministic experiment planner and executor for pack-based Hydra sweeps.

## Quickstart

```bash
uv venv && uv sync

# Plan and run in one step
uv run geryon launch \
  --experiment ./experiment.yaml \
  --out ./outputs \
  --batch-size 16 \
  --backend local \
  --profile local_fast \
  --profiles-file ./profiles.yaml

# Or split into stages
uv run geryon plan --experiment ./experiment.yaml --out ./outputs --batch-size 16
uv run geryon run-local --run ./outputs/runs/<run_id> --profile local_fast --profiles-file ./profiles.yaml
uv run geryon collect --run ./outputs/runs/<run_id>
```

## Commands

| Command | Description |
|---|---|
| `launch` | Validate + plan + execute in one step |
| `plan` | Generate config/batch artifacts |
| `validate-config` | Check experiment schema |
| `inspect-config` | Print composed config after imports/defs expansion |
| `run-local` | Execute on the local machine |
| `run-slurm` | Submit to Slurm via Submitit |
| `status` | Summarize run progress |
| `report` | Generate a run report (table, markdown, or JSON) |
| `recover` | Retry failed/terminated configs in one step |
| `rerun` | Build a retry selection file |
| `collect` | Aggregate result records into a summary |
| `list-profiles` | Show available profiles |
| `clean` | Delete plan or exec artifacts |

Rich table output is the default terminal UX. Use `--format json` for machine-readable output.

## Artifacts

All run files live under `<out>/runs/<run_id>/`:

```
plan/
  configs.jsonl              # all planned configs
  batches.jsonl              # batch groupings
  batches/batch_000.txt      # rendered commands per batch
  manifest.{jsonl,csv}       # {id, wandb_name, tags, params}
  diagnostics.json           # composition/merge/constraint diagnostics
  run_set.json               # selected run-set metadata
exec/
  results/                   # per-task result JSONL records
  batch_logs/                # stdout/stderr/events per task
  workdirs/                  # per-command work directories
  retries/                   # retry selection files
```

## Status and Recovery

Check progress:

```bash
uv run geryon status --run ./outputs/runs/<run_id>
uv run geryon status --run ./outputs/runs/<run_id> --by-pack architecture
uv run geryon report --run ./outputs/runs/<run_id> --format markdown --out report.md
```

Retry in one step:

```bash
uv run geryon recover --run ./outputs/runs/<run_id> --status failed --backend local
```

Or with two explicit steps:

```bash
uv run geryon rerun --run ./outputs/runs/<run_id> --status failed
uv run geryon run-local --run ./outputs/runs/<run_id> \
  --retry-file ./outputs/runs/<run_id>/exec/retries/retry_<timestamp>.json \
  --profile local_fast
```

Target specific configs directly:

```bash
uv run geryon run-local --run ./outputs/runs/<run_id> --config-id <id1> --config-id <id2>
```

## Resource Profiles

Define reusable profile presets in `profiles.yaml`:

```yaml
profiles:
  local_fast:
    defaults:
      run_local:
        executor: process
        max_concurrent_tasks: 4
        cores_per_task: 1
        resume: true
        progress: true

  a100_short:
    partition: gpu
    time_min: 240
    cpus_per_task: 8
    mem_gb: 64
    gpus_per_node: 1
    mail_user: you@example.org
    mail_type: END,FAIL
    env_script: ./env/bootstrap.sh
    env_setup_cmds:
      - export OMP_PROC_BIND=true
    slurm_setup_cmds:
      - module load cuda/12.2
    slurm_additional_parameters:
      account: my_project
      qos: normal
      constraint: a100
      exclusive: true
    defaults:
      run_slurm:
        executor: tmux
        max_concurrent_tasks: 2
        cores_per_task: 4
        batches_per_task: 2
        resume: true
        query_status: true
    env:
      DATA_ROOT: /scratch/data
```

A ready-to-edit template is at `profiles.example.yaml`. Example presets for the example suite are at `examples/configs/profiles.yaml`.

Profile precedence (lowest → highest): built-in defaults → `defaults.<command>` → explicit profile fields → CLI flags.

For extra `sbatch` flags, use `slurm_additional_parameters` or `defaults.run_slurm.sbatch_option`. Option keys can use either `-` or `_`.

```bash
uv run geryon list-profiles --profiles-file ./profiles.yaml
uv run geryon run-local --run ./outputs/runs/<run_id> --profile local_fast
uv run geryon run-slurm --run ./outputs/runs/<run_id> --profile a100_short
```

## Run-Sets

Define multiple named planning variants inside one experiment file:

```yaml
run_sets:
  baseline:
    replace:
      select:
        groups: [presets.core]
  no_pbrs:
    extends: [baseline]
    replace:
      select:
        packs:
          - ref: presets.reward
            replace_options: true
            options:
              - id: sparse
                params:
                  reward:
                    mode: sparse
```

Plan one variant:

```bash
uv run geryon plan --experiment ./experiment.yaml --run-set baseline --out ./outputs --batch-size 16
```

Plan all variants:

```bash
uv run geryon plan --experiment ./experiment.yaml --all-run-sets --out ./outputs --batch-size 16
```

## Failure Policies

Configured under `defaults.run_local` / `defaults.run_slurm` in the profile:

- `command_timeout_sec` — kill a command after this many seconds
- `max_retries` — retry count per config
- `retry_on_status` — statuses that trigger a retry (`failed`, `terminated`)
- `max_failures` — absolute failure cap before stopping the batch
- `fail_fast_threshold` — failure rate cap before stopping the batch

Policy events (`retry_scheduled`, `timeout_kill`, `fail_fast_stop`) are recorded in `exec/batch_logs/`.

## Live Terminal UX

Enable `defaults.run_local.progress: true` in the profile for a live local execution dashboard showing active commands, batch/line/config IDs, assigned cores, tmux session names, and recent completions.

Enable `defaults.run_slurm.query_status: true` to poll the Slurm queue during submission.

JSONL integrity checks:

```bash
uv run geryon status --run ./outputs/runs/<run_id> --strict-jsonl
uv run geryon report --run ./outputs/runs/<run_id> --strict-jsonl --format json
uv run geryon collect --run ./outputs/runs/<run_id> --strict-jsonl
```

## Environment Bootstrap

Use one shared script for local and Slurm runs:

```bash
# env/bootstrap.sh
set -e
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate myenv
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate myenv
fi
export WANDB_API_KEY=...
```

Reference it via `env_script` in the profile:

```bash
uv run geryon run-local --run ./outputs/runs/<run_id> --profile local_fast
uv run geryon run-slurm --run ./outputs/runs/<run_id> --profile a100_short
```

## Hydra Work Directory

Geryon automatically injects `hydra.run.dir=<workdir>` into each launched command. The path is unique per attempt and lives under `exec/workdirs/`.

## Local Core Scheduling

Set `cores_per_task`, `max_concurrent_tasks`, and `max_total_cores` in profile defaults. On Linux, CPU pinning uses `taskset`. On macOS, core groups are still scheduled disjointly with thread-count env defaults for oversubscription control.

## PyLauncher Executor

Set `executor: pylauncher` in the profile. Uses `LocalLauncher` outside Slurm and `ClassicLauncher` inside, with a generated command file and one wrapper per planned command. If `LocalLauncher` is unavailable, geryon falls back to `ClassicLauncher` with a compatibility shim. Install with `uv pip install pylauncher`.

## Aliases

- `build` → `plan`
- `local` → `run-local`
- `slurm` → `run-slurm`

## Documentation

Full docs at `docs/` (build locally with `mkdocs serve`):

- [Getting Started](docs/getting-started.md) — install and first run
- [Experiment YAML Schema](docs/experiments.md) — schema-v4 reference
- [Composition and Packs](docs/packs.md) — `imports`/`option_sets`/`packs`/`groups`/`select`
- [Python DSL](docs/dsl.md) — `geryon.dsl` API
- [CLI Reference](docs/interface.md) — every command and flag
- [Profiles](docs/profiles.md) — `profiles.yaml` schema and precedence
- [Runtime Artifacts](docs/runtime-artifacts.md) — on-disk layout and status model
- [Python API](docs/python-api.md) — programmatic entry points
- [Examples](docs/examples.md) — runnable scenarios
