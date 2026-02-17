# Python API Reference

This page summarizes primary programmatic entry points in `geryon`.

## Planning APIs

Module: `geryon.planner`

- `parse_experiment_yaml(path, run_set=None) -> ExperimentSpec`
- `parse_experiment_yaml_with_diagnostics(path, run_set=None) -> (ExperimentSpec, diagnostics)`
- `get_experiment_run_sets(path) -> list[str]`
- `plan_experiment(experiment_path, out_dir, batch_size, run_id=None, run_set=None, dry_run=False) -> PlanSummary`
- `summarize_plan_diagnostics(diagnostics) -> dict`

`PlanSummary` fields:

- `run_id`
- `run_root`
- `total_configs`
- `total_batches`

## Composition API

Module: `geryon.config_compose`

- `compose_experiment_data(experiment_path, run_set=None) -> (composed_doc, diagnostics)`
- `list_run_set_names(experiment_path) -> list[str]`

Use this when you need expanded `packs` before planning.

## Execution APIs

Module: `geryon.execution`

- `available_batch_indices(store) -> list[int]`
- `select_batch_indices(store, requested) -> list[int]`
- `build_task_payloads(...) -> list[TaskPayload]`
- `execute_task_payload(payload, job_id=None, array_task_id=None, progress_callback=None) -> dict`

`TaskPayload` includes executor choice, selected batches/config IDs, env setup, and retry/fail-fast policy.

## Status and Reporting APIs

Module: `geryon.status`

- `build_run_status_index(run_dir) -> RunStatusIndex`
- `summarize_run_status(index) -> dict`
- `summarize_status_groups(index, by_packs=...) -> dict`
- `build_run_report_payload(index, by_packs=...) -> dict`
- `render_report_markdown(report_payload) -> str`
- `select_rerun_config_ids(index, status_filter, explicit_config_ids) -> list[str]`
- `successful_config_ids(store_or_run) -> set[str]`

## Collect API

Module: `geryon.collect`

- `collect_run(run_dir, dry_run=False) -> dict`

## Profiles APIs

Module: `geryon.profiles`

- `load_profiles(path, required) -> (resolved_path, dict[str, RunProfile])`
- `resolve_profile(profile_name, profiles_file) -> (resolved_path, RunProfile | None)`
- `merge_profile_env(profile, cli_env_script, cli_env_setup_cmds, cli_env_vars) -> (env_script, env_setup_cmds, env_vars)`

## Storage API

Module: `geryon.store`

- `ArtifactStore.from_run_dir(run_dir)`
- Plan readers/writers:
  - `read_planned_configs`, `read_planned_batches`, `read_batch_commands`
  - `write_planned_configs`, `write_planned_batches`, `write_manifest`
- Layout helpers for paths under `plan/` and `exec/`

## CLI API

Module: `geryon.cli`

- `main(argv: Sequence[str] | None = None) -> int`

Programmatic CLI invocation is used in tests and returns process-style exit code.

## Error Type

Module: `geryon.models`

- `GeryonError` (base runtime error)
- `ConfigError` (validation/configuration errors)

`ConfigError` is the expected exception type for invalid YAML/DSL/CLI configuration.
