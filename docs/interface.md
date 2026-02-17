# CLI Reference

Entry point:

- `geryon` -> `geryon.cli:main`

Deprecated aliases (still accepted):

- `build` -> `plan`
- `local` -> `run-local`
- `slurm` -> `run-slurm`

Output modes:

- Most commands support `--format table|json`
- `inspect-config` supports `yaml|json`
- `report` supports `table|markdown|json`

## `plan`

Plan configs and batch files.

Required:

- `--experiment`
- `--out`
- `--batch-size`

Optional:

- `--run-id`
- `--run-set`
- `--all-run-sets` (mutually exclusive with `--run-set`)
- `--dry-run`
- `--format table|json` (default `table`)

Notes:

- `--all-run-sets` fails if no run sets exist.
- With `--all-run-sets` and explicit `--run-id`, each run ID is suffixed with sanitized run-set name.

## `validate-config`

Validate schema + planner normalization.

Required:

- `--experiment`

Optional:

- `--run-set`
- `--show-diagnostics`
- `--format table|json`

## `inspect-config`

Render composed config after imports/defs/select expansion.

Required:

- `--experiment`

Optional:

- `--run-set`
- `--show-diagnostics`
- `--format yaml|json` (default `yaml`)
- `--out` (writes rendered content to file)

## `run-local`

Execute planned work on current machine.

Required:

- `--run` (path to run root)

Selection flags:

- `--batch-index` (repeatable)
- `--config-id` (repeatable)
- `--retry-file` (JSON from `rerun`)
- retry-file implies resume semantics during execution.

Profile flags:

- `--profile`
- `--profiles-file`

Output flags:

- `--dry-run`
- `--format table|json`

Help groups:

- Selection
- Profile
- Output

## `run-slurm`

Submit task payloads to Slurm via Submitit.

Required:

- `--run`
- resolved Slurm resources from profile values/defaults (`partition`, `time_min`, etc.)

Profile flags:

- `--profile`
- `--profiles-file`

Output flags:

- `--dry-run`
- `--format table|json`

Help groups:

- Selection
- Profile
- Output

## `launch`

Single command for validate + plan + execute.

Required:

- `--experiment`
- `--out`
- `--batch-size`

Optional:

- `--run-id`
- `--run-set`
- `--backend local|slurm` (default `local`)
- `--skip-validate`
- optional selection/profile flags from run commands
- `--dry-run`
- `--format table|json`

## `recover`

Single command for retry selection + execution.

Required:

- `--run`

Optional:

- `--status failed|terminated|missing` (default `failed`)
- `--backend local|slurm` (default `local`)
- optional selection/profile flags from run commands
- `--dry-run`
- `--format table|json`

## `list-profiles`

List profile presets.

Flags:

- `--profiles-file`
- `--format table|json`

## `status`

Summarize run status from plan + result files.

Flags:

- `--run`
- `--by-pack` (repeatable or comma-separated)
- `--strict-jsonl` (fail if corrupt JSONL result lines are detected)
- `--format table|json`

Grouping fails if requested pack is unknown in selected options.

## `report`

Generate run report.

Flags:

- `--run`
- `--by-pack` (repeatable or comma-separated)
- `--strict-jsonl` (fail if corrupt JSONL result lines are detected)
- `--format table|markdown|json`
- `--out` (optional)

When `--format table` and `--out` is provided, markdown is written to `--out`.

## `rerun`

Build retry metadata from current status.

Flags:

- `--run`
- `--status failed|terminated|missing` (default `failed`)
- `--config-id` (repeatable explicit additions)
- `--format table|json`
- `--dry-run`

Writes `exec/retries/retry_<timestamp>.json` unless dry-run.

## `collect`

Aggregate `exec/results/task_*.jsonl` into summary.

Flags:

- `--run`
- `--strict-jsonl` (fail if corrupt JSONL result lines are detected)
- `--dry-run`
- `--format table|json`

Writes `exec/results/summary.json` unless dry-run.

## `clean`

Delete artifacts under run root.

Required:

- `--run`
- one of `--plan`, `--exec`, `--all`

If no scope flag is set, command fails.

## Exit Codes

- `0`: success
- `1`: runtime/system/command error
- `2`: configuration/validation error (`ConfigError`)
- `130`: interrupted (`KeyboardInterrupt`)
