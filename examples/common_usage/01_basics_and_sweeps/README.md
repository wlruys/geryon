# 01 Basics and Sweeps

Minimal starting point for pack-based sweeps, defaults, and tags.

## Feature Set

- Basic `schema.version: 4` experiment shape.
- Cartesian product over two packs (`model x optimizer`).
- Shared defaults (`defaults.params`) and tags (`defaults.tags`).

## Run (Local)

```bash
./launch.sh
```

Run root (default): `./outputs/runs/01_basics_and_sweeps_demo`

## Dry-Run and Validation

```bash
geryon validate-config --experiment ./experiment.yaml --format table
geryon inspect-config --experiment ./experiment.yaml --format yaml
geryon launch \
  --experiment ./experiment.yaml \
  --out ./outputs \
  --run-id 01_basics_and_sweeps_demo \
  --batch-size 4 \
  --backend local \
  --profile local_dev \
  --profiles-file ./profiles.yaml \
  --dry-run \
  --format json
```

## UI/UX Options

```bash
geryon status --run ./outputs/runs/01_basics_and_sweeps_demo --format table
geryon status --run ./outputs/runs/01_basics_and_sweeps_demo --format json
```

## Optional Slurm + Queue Status

```bash
geryon run-slurm \
  --run ./outputs/runs/01_basics_and_sweeps_demo \
  --profile slurm_demo \
  --profiles-file ./profiles.yaml \
  --dry-run \
  --format json

# Real submission (cluster required):
# geryon run-slurm --run ./outputs/runs/01_basics_and_sweeps_demo --profile slurm_demo --profiles-file ./profiles.yaml --query-status
# geryon queue --run ./outputs/runs/01_basics_and_sweeps_demo
# geryon queue-refresh --run ./outputs/runs/01_basics_and_sweeps_demo
```
