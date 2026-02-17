# Geryon

Geryon is a runner for giant experiments using Hydra.

Core commands:

- `geryon launch --experiment <yaml> --out <dir> --batch-size <n> [--backend local|slurm]`
- `geryon plan --experiment <yaml> --out <dir> --batch-size <n> [--run-set <name>|--all-run-sets]`
- `geryon inspect-config --experiment <yaml> [--format yaml|json]`
- `geryon run-local --run <run_dir> [--profile <name>] [--format table|json]`
- `geryon run-slurm --run <run_dir> [--profile <name>] [--format table|json]`
- `geryon status --run <run_dir> [--format table|json] [--by-pack <pack>]`
- `geryon report --run <run_dir> [--format table|markdown|json]`
- `geryon recover --run <run_dir> [--status failed|terminated|missing] [--backend local|slurm]`
- `geryon rerun --run <run_dir> [--status failed|terminated|missing]`
- `geryon list-profiles [--profiles-file <path>]`
- `geryon collect --run <run_dir> [--format table|json]`
- `geryon clean --run <run_dir> [--plan|--exec|--all]`

Rich table output is the default terminal UX. Use `--format json` for machine-readable output.

The planner preserves existing `packs x options` semantics and writes deterministic plan artifacts under:

- `<out>/runs/<run_id>/plan/`
- `<out>/runs/<run_id>/exec/`

Planner diagnostics are written to:

- `<out>/runs/<run_id>/plan/diagnostics.json`
- `<out>/runs/<run_id>/plan/diagnostics.summary.txt`
- `<out>/runs/<run_id>/plan/run_set.json`

Execution task-level logs are written to:

- `<out>/runs/<run_id>/exec/batch_logs/task_<jobid>_<arrayid>/task.stdout.log`
- `<out>/runs/<run_id>/exec/batch_logs/task_<jobid>_<arrayid>/task.stderr.log`
- `<out>/runs/<run_id>/exec/batch_logs/task_<jobid>_<arrayid>/task.events.jsonl`

Execution work directories are created per launched command under:

- `<out>/runs/<run_id>/exec/workdirs/task_<jobid>_<arrayid>/batch_<batch>/line_<line>_<config>_<attempt>/`

Retry plans are written to:

- `<out>/runs/<run_id>/exec/retries/retry_<timestamp>.json`

Compatibility aliases are available:

- `build` -> `plan`
- `local` -> `run-local`
- `slurm` -> `run-slurm`

## Detailed Documentation

- MkDocs site config: `mkdocs.yml`
- `docs/index.md`: module overview and lifecycle.
- `docs/getting-started.md`: install + first-run workflow.
- `docs/experiments.md`: complete schema-v4 YAML reference.
- `docs/packs.md`: composition model (`imports`/`option_sets`/`packs`/`groups`/`select`) and expansion rules.
- `docs/dsl.md`: complete `geryon.dsl` API reference.
- `docs/interface.md`: full CLI command/flag reference.
- `docs/profiles.md`: `profiles.yaml` schema and precedence.
- `docs/runtime-artifacts.md`: on-disk artifact contracts and status semantics.
- `docs/python-api.md`: programmatic API entry points.
- `docs/examples.md`: runnable examples map.

Build docs locally:

```bash
mkdocs serve
# or
mkdocs build
```

## Resume/Retry Workflow

Use rich table status output for quick progress checks:

```bash
uv run geryon status --run ./outputs/runs/<run_id>
uv run geryon status --run ./outputs/runs/<run_id> --format json
uv run geryon status --run ./outputs/runs/<run_id> --by-pack architecture
```

Generate markdown/json reports:

```bash
uv run geryon report --run ./outputs/runs/<run_id> --format markdown --out report.md
uv run geryon report --run ./outputs/runs/<run_id> --format json --by-pack architecture
```

Create and execute recovery in one command (failed by default):

```bash
uv run geryon recover --run ./outputs/runs/<run_id> --status failed --backend local
```

Equivalent explicit two-step flow:

```bash
uv run geryon rerun --run ./outputs/runs/<run_id> --status failed
uv run geryon run-local --run ./outputs/runs/<run_id> --retry-file ./outputs/runs/<run_id>/exec/retries/retry_<timestamp>.json --profile local_fast --profiles-file ./profiles.yaml
```

You can also force explicit IDs:

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

A ready-to-edit template is included at `geryon/profiles.example.yaml`.
Example multi-site presets for the example suite are included at
`geryon/examples/configs/profiles.yaml`.

Inspect available profiles:

```bash
uv run geryon list-profiles
uv run geryon list-profiles --profiles-file ./profiles.yaml --format json
```

Use profile defaults for Slurm:

```bash
uv run geryon run-slurm \
  --run ./outputs/runs/<run_id> \
  --profile a100_short
```

Use profile env/bootstrap settings for local runs:

```bash
uv run geryon run-local \
  --run ./outputs/runs/<run_id> \
  --profile local_fast
```

Resolution order is:
1. Built-in defaults
2. `profiles.<name>.defaults.<command>`
3. Explicit profile fields
4. CLI overrides (when available)

Defaults are merged with explicit profile values; matching fields override defaults.
For extra `sbatch` flags, use `slurm_additional_parameters` and
`defaults.run_slurm.sbatch_option` in profiles. Options are merged with profile
values taking precedence. Option keys can use either `-` or `_`.

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

## Execution Failure Policies

Failure controls are configured in profile defaults (`defaults.run_local` / `defaults.run_slurm`):

- `command_timeout_sec`
- `max_retries`
- `retry_on_status`
- `max_failures`
- `fail_fast_threshold`

## Live Terminal UX

Show live local execution progress by enabling `defaults.run_local.progress: true` in profile:

```bash
uv run geryon run-local \
  --run ./outputs/runs/<run_id> \
  --profile local_fast
```

The live dashboard includes currently running commands, batch/line/config IDs,
assigned cores (when available), active tmux session names, recent starts, and
recent completions.

Query SLURM queue state by enabling `defaults.run_slurm.query_status: true` in profile:

```bash
uv run geryon run-slurm \
  --run ./outputs/runs/<run_id> \
  --profile a100_short
```

JSONL integrity checks:

```bash
uv run geryon status --run ./outputs/runs/<run_id> --strict-jsonl
uv run geryon report --run ./outputs/runs/<run_id> --strict-jsonl --format json
uv run geryon collect --run ./outputs/runs/<run_id> --strict-jsonl
```

For Slurm runs, runtime env/bootstrap and submit preamble behavior are configured in profile fields
(`env_script`, `env_setup_cmds`, `env`, `slurm_setup_cmds` and `defaults.run_slurm.sbatch_option`).

Policy events are recorded in task logs:
- `retry_scheduled`
- `timeout_kill`
- `fail_fast_stop`

## Portable Environment Bootstrap

Use one shared shell bootstrap script for both local and Slurm runs, and keep
cluster-specific details out of Python code:

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

Then reference it from profile and run with that profile:

```bash
uv run geryon run-local --run ./outputs/runs/<run_id> --profile local_fast
uv run geryon run-slurm --run ./outputs/runs/<run_id> --profile a100_short
```

## Task Work Directory Injection

At execution time, geryon appends a task-local CLI override to every launched command:

- `hydra.run.dir=<absolute-path-under-run>/exec/workdirs/...`

This path is unique per launched command attempt and lives next to run artifacts (`plan/`, `exec/`).

## Composed Experiment Interface

`experiment.yaml` can be composed from reusable subfiles so you can build
ablations without rewriting full pack lists each time.

Supported reusable definition blocks:

- `option_sets`: reusable lists of options
- `packs`: reusable pack templates
- `groups`: reusable groups of pack selectors
- `imports[].package`: optional namespace for definitions loaded through that import edge
- `imports`: recursive file imports (relative to current file)

Top-level experiment selectors:

- `select.groups`: include one or more group refs
- `select.packs`: include pack refs or inline packs

Selector features for ablations:

- `ref`: pack reference
- `replace_options: true`: replace referenced pack options
- `options_from`: append option sets
- `options`: append inline options
- `filter.include_ids`, `filter.exclude_ids`: filter options

Example:

```yaml
# defs/options.yaml
option_sets:
  arch_family:
    - id: arch_a
      tag: arch-a
      params:
        model: {type: A}
    - id: arch_b
      tag: arch-b
      params:
        model: {type: B}
  seeds:
    - id: seed1
      tag: seed1
      params:
        seed: 1
    - id: seed2
      tag: seed2
      params:
        seed: 2
```

```yaml
# defs/packs.yaml
packs:
  architecture:
    name: architecture
    options_from:
      - ref: arch_family
  seed_pack:
    name: seed
    options_from:
      - ref: seeds
```

```yaml
# defs/groups.yaml
groups:
  core:
    - ref: seed_pack
```

```yaml
# experiment.yaml
imports:
  - path: defs/options.yaml
    package: presets
  - path: defs/packs.yaml
    package: presets
  - path: defs/groups.yaml
    package: presets
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('run')"]
select:
  groups: [presets.core]
  packs:
    - ref: presets.architecture
      replace_options: true
      options_from:
        - ref: presets.arch_family
          include_ids: [arch_b]
```

This runs only `arch_b` with the shared `core` group, while reusing all
predefined sets across experiments.

Preview composed config before planning:

```bash
uv run geryon inspect-config \
  --experiment ./experiment.yaml \
  --format yaml \
  --show-diagnostics
```

## Quickstart

```bash
cd geryon
uv venv
uv sync --extra dev --extra slurm --extra pylauncher
uv run geryon plan --experiment ../src/jac_mlp_feature/experiment.yaml --out ./outputs --batch-size 16
uv run geryon run-local --run ./outputs/runs/<run_id> --profile local_fast --profiles-file ./profiles.yaml
uv run geryon collect --run ./outputs/runs/<run_id>
```

## Local Core Scheduling

- Configure `cores`, `cores_per_task`, `max_concurrent_tasks`, and `max_total_cores` in profile defaults.
- Linux pinning uses `taskset` when available, with `sched_setaffinity` as a best-effort fallback path.
- On macOS and other platforms without Linux affinity APIs, core groups are still scheduled disjointly and thread-count env defaults are applied for safer oversubscription control.

## PyLauncher Executor

- `executor: pylauncher` uses `pylauncher.ClassicLauncher` in Slurm environments and `pylauncher.LocalLauncher` outside Slurm, with a generated command file and one wrapper script per planned command.
- Wrapper scripts capture per-command exit codes and preserve geryon stdout/stderr log paths.
- If `LocalLauncher` is unavailable in the installed PyLauncher version, geryon falls back to `ClassicLauncher` with a local compatibility shim.
- Install dependency: `uv pip install pylauncher`.
