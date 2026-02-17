# 02 Constraints and Filtering

Shows how `include`, `exclude`, and predicates shrink a search space.

## Feature Set

- `constraints.include` as allowlist.
- `constraints.exclude` for known-invalid combinations.
- Predicate guardrail based on resolved params.

## Run (Local)

```bash
./launch.sh
```

Run root (default): `./outputs/runs/02_constraints_and_filtering_demo`

## Dry-Run and Validation

```bash
geryon validate-config --experiment ./experiment.yaml --format table
geryon inspect-config --experiment ./experiment.yaml --format yaml --show-diagnostics
geryon launch \
  --experiment ./experiment.yaml \
  --out ./outputs \
  --run-id 02_constraints_and_filtering_demo \
  --batch-size 4 \
  --backend local \
  --profile local_throughput \
  --profiles-file ./profiles.yaml \
  --dry-run \
  --format json
```

## UI/UX Options

```bash
geryon status --run ./outputs/runs/02_constraints_and_filtering_demo --format table
geryon report --run ./outputs/runs/02_constraints_and_filtering_demo --format markdown
```

## Optional Slurm + Queue Status

```bash
geryon run-slurm \
  --run ./outputs/runs/02_constraints_and_filtering_demo \
  --profile slurm_demo \
  --profiles-file ./profiles.yaml \
  --dry-run \
  --format json
```
