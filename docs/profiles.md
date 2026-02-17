# Profiles (`profiles.yaml`)

Profiles provide reusable runtime defaults for `run-local` and `run-slurm`.

Default path:

- `./profiles.yaml`

Override path:

- `--profiles-file <path>`

Load behavior:

- If `--profile` is provided, profiles file must exist.
- If `--profile` is omitted, missing profiles file is tolerated.

## File Schema

```yaml
profiles:
  local_fast:
    env_script: ./env/bootstrap.sh
    defaults:
      run_local:
        executor: process
        max_concurrent_tasks: 4
        cores_per_task: 1
        max_retries: 1
        retry_on_status: [failed, terminated]
        resume: true
        progress: true

  a100_short:
    partition: gpu
    time_min: 240
    cpus_per_task: 8
    mem_gb: 64
    gpus_per_node: 1
    job_name: geryon
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
        slurm_setup_cmds:
          - module load cuda/12.2
        sbatch_option:
          qos: normal
          constraint: a100
    env:
      DATA_ROOT: /scratch/data
```

Allowed profile keys:

- `partition`
- `time_min`
- `cpus_per_task`
- `mem_gb`
- `gpus_per_node`
- `job_name`
- `mail_user`
- `mail_type`
- `env_script`
- `env_setup_cmds`
- `slurm_setup_cmds`
- `slurm_additional_parameters`
- `env`
- `defaults`

Unknown keys are errors.

`defaults` supports only:

- `defaults.run_local`:
  - `executor`
  - `max_concurrent_tasks`
  - `cores_per_task`
  - `max_total_cores`
  - `cores`
  - `batches_per_task`
  - `command_timeout_sec`
  - `max_retries`
  - `retry_on_status`
  - `max_failures`
  - `fail_fast_threshold`
  - `resume`
  - `progress`
- `defaults.run_slurm`:
  - all common fields above except `progress`
  - `slurm_setup_cmds`
  - `sbatch_option`
  - `partition`
  - `time_min`
  - `cpus_per_task`
  - `mem_gb`
  - `gpus_per_node`
  - `job_name`
  - `mail_user`
  - `mail_type`
  - `query_status`

## Types and Normalization

- numeric fields must be integers (bool is rejected).
- `mail_type` may be:
  - comma-separated string
  - list of strings
  both normalize to canonical comma string.
- `env` values may be scalar (`str|int|float|bool`) and are stringified.
- env variable names must match `[A-Za-z_][A-Za-z0-9_]*`.
- `slurm_additional_parameters` keys normalize `-` to `_`.
- `env_script` relative path resolves relative to profiles file directory.

## Precedence

For `run-local`/`run-slurm` fields covered by `defaults`:

1. built-in defaults
2. `profiles.<name>.defaults.<command>`
3. explicit profile fields (`partition`, `time_min`, env/slurm setup, etc.)
4. CLI overrides (when available)

Defaults are merged with explicit profile values. When fields overlap, explicit
profile values win.

## Slurm resource params (`run-slurm`)

Resolution order follows precedence above.

Required after resolution:

- `partition`
- `time_min`

Defaults after profile resolution:

- `cpus_per_task`: `1`
- `mem_gb`: `16`
- `gpus_per_node`: `0`
- `job_name`: `geryon`

## Environment bootstrap (`run-local` and `run-slurm`)

Resolution:

- `env_script`: from profile.
- `env_setup_cmds`: from profile.
- `env`: from profile.

In `run-slurm`, runtime setup commands are converted into `slurm_setup` preamble lines:

1. `export KEY='VALUE'` for merged env map
2. `source <env_script>` if set
3. profile `env_setup_cmds`
4. `defaults.run_slurm.slurm_setup_cmds` then profile `slurm_setup_cmds`

## CLI Integration

List profiles:

```bash
geryon list-profiles --profiles-file ./profiles.yaml
geryon list-profiles --profiles-file ./profiles.yaml --format json
```

Use with local run:

```bash
geryon run-local --run ./outputs/runs/<run_id> --profile local_fast --profiles-file ./profiles.yaml
```

Use with Slurm run:

```bash
geryon run-slurm --run ./outputs/runs/<run_id> --profile a100_short --profiles-file ./profiles.yaml
```

Workflow wrappers using profile defaults:

```bash
geryon launch --experiment ./experiment.yaml --out ./outputs --batch-size 16 --profile local_fast
geryon recover --run ./outputs/runs/<run_id> --profile local_fast
```

## Common Errors

- profiles file missing while `--profile` is set
- unknown profile name
- unknown keys in profile entry
- invalid env variable names
- non-scalar values in `env` or `slurm_additional_parameters`
