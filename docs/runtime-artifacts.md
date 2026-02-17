# Runtime Artifacts and Status Model

Geryon uses an append-only artifact model under each run root.

Run root layout:

```text
<out>/runs/<run_id>/
  run.json
  plan/
    experiment.snapshot.yaml
    configs.jsonl
    batches.jsonl
    batches/batch_000.txt
    manifest.jsonl
    manifest.csv
    diagnostics.json
    diagnostics.summary.txt
    run_set.json
  exec/
    results/task_<job>_<task>.jsonl
    results/summary.json
    batch_logs/task_<job>_<task>/
      task.stdout.log
      task.stderr.log
      task.events.jsonl
      batch_000.jsonl
    cmd_logs/task_<job>_<task>/batch_000/*.stdout.log
    cmd_logs/task_<job>_<task>/batch_000/*.stderr.log
    workdirs/task_<job>_<task>/batch_000/line_0000_<cfg>_<attempt>/
    retries/retry_<timestamp>.json
    task_payloads.jsonl
    task_payloads/task_payloads_<snapshot>.jsonl
    submitit_jobs.json
    submitit_jobs/submitit_jobs_<snapshot>.json
    submitit/
```

## Plan Artifacts

- `plan/configs.jsonl`: one `PlannedConfig` per config.
- `plan/batches.jsonl`: one `PlannedBatch` per batch.
- `plan/batches/batch_XXX.txt`: rendered commands for each batch.
- `plan/manifest.{jsonl,csv}`: `{id, wandb_name, tags, params}`.
- `plan/diagnostics.json`: composition/merge/constraint/predicate diagnostics.
- `plan/run_set.json`: selected run-set metadata.
- `run.json`: run metadata (`run_id`, paths, counts, schema version, etc.).

## Execution Records

Each attempt record in `exec/results/task_*.jsonl` contains:

- run/config identity: `run_id`, `config_id`, `batch_index`, `line_index`
- attempt graph: `attempt_id`, `attempt_index`, `parent_attempt_id`
- runtime identity: `hostname`, `pid`, `tmux_session`, `job_id`, `array_task_id`
- timing/outcome: `start_time`, `end_time`, `duration_sec`, `exit_code`, `status`, `status_reason`
- logs/command: `stdout_path`, `stderr_path`, `command`, `executor`
- plan context: `selected_options`

Status values emitted by executors:

- `success`
- `failed`
- `terminated`

`status_reason` may include `timeout_kill`.

## Task Event Stream

`exec/batch_logs/task_<job>_<task>/task.events.jsonl` includes lifecycle events:

- `task_start`
- `batch_start`
- `batch_filter`
- `batch_launch`
- `command_start`
- `command_complete`
- `retry_scheduled`
- `timeout_kill`
- `fail_fast_stop`
- `fail_fast_drop_pending`
- `batch_end`
- `batch_skipped`
- `batch_skipped_fail_fast`
- `executor_error` (if executor loop throws)
- `task_end`

Every task-event record is validated before write and includes these required
top-level fields:

- `schema_version` (current value: `1`)
- `event`
- `time`
- `run_id`
- `job_id`
- `array_task_id`
- `submission_id` (nullable)

Event-specific required fields:

| Event | Required fields |
| --- | --- |
| `task_start` | `executor`, `batches`, `resume`, `selected_config_ids`, `policy` |
| `batch_start` | `batch_index`, `num_commands`, `executor` |
| `batch_filter` | `batch_index`, `planned_commands`, `selected_commands`, `skipped_not_selected`, `skipped_resume` |
| `batch_launch` | `batch_index`, `wave_index`, `num_commands` |
| `command_start` | `batch_index`, `line_index`, `config_id`, `attempt_id`, `wave_index`, `executor` |
| `command_complete` | `batch_index`, `line_index`, `config_id`, `status`, `attempt_id`, `attempt_index`, `duration_sec` |
| `timeout_kill` | `batch_index`, `line_index`, `config_id`, `attempt_id`, `attempt_index` |
| `retry_scheduled` | `batch_index`, `line_index`, `config_id`, `previous_attempt_id`, `next_attempt_index`, `retry_count`, `max_retries`, `status` |
| `fail_fast_stop` | `reason`, `batch_index`, `line_index`, `config_id`, `terminal_total`, `terminal_failed`, `failure_rate` |
| `fail_fast_drop_pending` | `reason`, `dropped_commands`, `wave_index`, `batch_index` |
| `batch_end` | `batch_index`, `status` |
| `batch_skipped` | `batch_index`, `skipped_not_selected`, `skipped_resume` |
| `batch_skipped_fail_fast` | `batch_index`, `num_commands`, `reason` |
| `executor_error` | `error` |
| `task_end` | `summary` |

## Batch Event Stream

`exec/batch_logs/task_<job>_<task>/batch_<index>.jsonl` is also schema-validated.

Every batch-event record includes these required top-level fields:

- `schema_version` (current value: `1`)
- `event`
- `time`
- `run_id`
- `job_id`
- `array_task_id`
- `submission_id` (nullable)

Batch event-specific required fields:

| Event | Required fields |
| --- | --- |
| `batch_start` | `batch_index`, `num_commands`, `executor` |
| `batch_filter` | `batch_index`, `planned_commands`, `selected_commands`, `skipped_not_selected`, `skipped_resume` |
| `command_start` | `batch_index`, `line_index`, `config_id`, `attempt_id`, `wave_index`, `executor` |
| `command_complete` | `batch_index`, `line_index`, `config_id`, `status`, `attempt_id`, `attempt_index` |
| `retry_scheduled` | `batch_index`, `line_index`, `config_id`, `retry_count`, `max_retries`, `status` |
| `fail_fast_stop` | `reason`, `terminal_total`, `terminal_failed` |
| `fail_fast_drop_pending` | `reason`, `dropped_commands`, `wave_index`, `batch_index` |
| `batch_end` | `batch_index`, `status` |

## Resume and Retry Semantics

- Resume mode skips configs with **any prior success** (`status.successful_config_ids`).
  It is enabled via profile defaults (`defaults.run_local.resume` / `defaults.run_slurm.resume`)
  and is automatically enabled when `--retry-file` is used.
- `rerun` selects target IDs by latest status (`failed`, `terminated`, `missing`) plus explicit `--config-id`.
- `run-local/run-slurm --retry-file` restricts execution to retry selection IDs.
- retries append new attempt records; old records are preserved.

## Status Model

`status` builds `RunStatusIndex` from:

- planned configs (`plan/configs.jsonl`)
- attempt records (`exec/results/task_*.jsonl`)

Per config, status tracks:

- attempt count
- latest status
- latest attempt id
- whether any success exists (`has_success`)

Completion definition:

- complete: `has_success == true`
- pending: no successful attempt yet

`missing` means planned config with no attempts.

## Collect Summary

`collect` aggregates all result files into:

- `attempts.total`
- `attempts.by_status`
- final latest status per config (`configs.final_by_status`)
- `failed_config_ids` (latest status failed or terminated)
- per-batch status counts

When not dry-run, writes `exec/results/summary.json`.

## Integration Surfaces

If you automate around geryon, use these stable machine-readable files:

- `run.json`
- `plan/configs.jsonl`
- `plan/batches.jsonl`
- `plan/diagnostics.json`
- `exec/results/task_*.jsonl`
- `exec/retries/retry_*.json`
- `exec/task_payloads.jsonl`
- `exec/submitit_jobs.json`
