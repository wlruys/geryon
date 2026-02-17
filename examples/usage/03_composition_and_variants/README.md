# 03 Composition and Variants

Teaches modular composition (`imports`) and variant execution (`run_sets`).

## Feature Set

- Import reusable pack definitions from `defs/packs.yaml`.
- Expand packs via referenced definitions (`ref: catalog.*_pack`).
- Switch variants with `run_sets` (`baseline`, `no_dropout`, `sgd_only`).

## Run (Local)

```bash
./launch.sh
```

Run root (default): `./outputs/runs/03_composition_and_variants_demo`

Run a specific variant:

```bash
./launch.sh --run-set no_dropout
./launch.sh --run-set sgd_only
```

## Dry-Run and Validation

```bash
geryon validate-config --experiment ./experiment.yaml --run-set baseline --format table
geryon inspect-config --experiment ./experiment.yaml --run-set no_dropout --format yaml
geryon launch \
  --experiment ./experiment.yaml \
  --out ./outputs \
  --run-id 03_composition_and_variants_demo \
  --run-set sgd_only \
  --batch-size 4 \
  --backend local \
  --profile local_dev \
  --profiles-file ./profiles.yaml \
  --dry-run
```

## UI/UX Options

```bash
geryon status --run ./outputs/runs/03_composition_and_variants_demo --format table
geryon status --run ./outputs/runs/03_composition_and_variants_demo --by-pack model --by-pack optimizer
```

## Optional Slurm + Queue Status

```bash
geryon run-slurm \
  --run ./outputs/runs/03_composition_and_variants_demo \
  --profile slurm_demo \
  --profiles-file ./profiles.yaml \
  --dry-run \
  --format json
```
