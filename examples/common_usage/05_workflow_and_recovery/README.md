# 05 Workflow and Recovery

Operator workflow example: dry-run, launch, status, rerun selection, targeted retry, and recover.

## Feature Set

- Two-phase execution behavior via profile timeouts.
- Deterministic `terminated` failures in the initial run.
- `rerun` + `--retry-file` retry targeting.
- Recovery with a relaxed retry profile.

## Run (Initial)

```bash
./launch.sh
```

Run root (default): `./outputs/runs/05_workflow_and_recovery_demo`

## Dry-Run and Validation

```bash
geryon validate-config --experiment ./experiment.yaml --format table
geryon inspect-config --experiment ./experiment.yaml --format yaml
geryon launch \
  --experiment ./experiment.yaml \
  --out ./outputs \
  --run-id 05_workflow_and_recovery_demo \
  --batch-size 4 \
  --backend local \
  --profile local_fail_fast \
  --profiles-file ./profiles.yaml \
  --dry-run
```

## Rerun Flow (Targeted Retry)

```bash
./rerun.sh
```

Equivalent manual flow:

```bash
geryon status --run ./outputs/runs/05_workflow_and_recovery_demo
geryon rerun --run ./outputs/runs/05_workflow_and_recovery_demo --status terminated
geryon run-local \
  --run ./outputs/runs/05_workflow_and_recovery_demo \
  --profile local_recover \
  --profiles-file ./profiles.yaml \
  --retry-file ./outputs/runs/05_workflow_and_recovery_demo/exec/retries/retry_<timestamp>.json
```

One-command alternative:

```bash
geryon recover \
  --run ./outputs/runs/05_workflow_and_recovery_demo \
  --status terminated \
  --backend local \
  --profile local_recover \
  --profiles-file ./profiles.yaml
```

## UI/UX Options

```bash
geryon status --run ./outputs/runs/05_workflow_and_recovery_demo --format table
geryon status --run ./outputs/runs/05_workflow_and_recovery_demo --format json
```

## Optional Slurm + Queue Status

```bash
geryon run-slurm \
  --run ./outputs/runs/05_workflow_and_recovery_demo \
  --profile slurm_demo \
  --profiles-file ./profiles.yaml \
  --dry-run \
  --format json
```
