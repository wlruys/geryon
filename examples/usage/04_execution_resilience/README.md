# 04 Execution Resilience

Demonstrates timeout, retry, fail-fast, and progress UI with flaky commands.

## Feature Set

- Parallel local execution (`max_concurrent_tasks`).
- Timeout kill (`command_timeout_sec`).
- Retry policy (`max_retries`, `retry_on_status`).
- Fail-fast stop (`max_failures`).
- Live progress dashboard via profile default (`progress: true`).

## Run (Local)

```bash
./launch.sh
```

Run root (default): `./outputs/runs/04_execution_resilience_demo`

## Dry-Run and Validation

```bash
geryon validate-config --experiment ./experiment.yaml --format table
geryon launch \
  --experiment ./experiment.yaml \
  --out ./outputs \
  --run-id 04_execution_resilience_demo \
  --batch-size 4 \
  --backend local \
  --profile local_resiliency \
  --profiles-file ./profiles.yaml \
  --dry-run
```

## UI/UX Options

```bash
geryon run-local \
  --run ./outputs/runs/04_execution_resilience_demo \
  --profile local_resiliency \
  --profiles-file ./profiles.yaml \
  --format table

geryon collect --run ./outputs/runs/04_execution_resilience_demo --format json
```

## Optional Slurm + Queue Status

```bash
geryon run-slurm \
  --run ./outputs/runs/04_execution_resilience_demo \
  --profile slurm_demo \
  --profiles-file ./profiles.yaml \
  --dry-run \
  --format json
```
