# Merge Policy Examples

Each directory is a focused, standalone fixture for one merge behavior.

## Map

- `01_conflict_behavior`: higher-priority value overrides in `merge.mode=merge`.
- `02_key_strategies`: path-level strategies (`deep_merge`, `append_unique`, `set_union`, `replace`).
- `03_intentional_overrides`: explicit ablation-style override.
- `04_duplicate_pack_extension`: duplicate pack names merged in `mode=merge`.
- `05_delete_sentinel`: field removal via `merge.delete_sentinel`.
- `06_policy_gates`: strict merge-mode guardrail behavior.

## Run One Example

```bash
cd geryon/examples/merge_policies/01_conflict_behavior
./launch.sh
```

## Dry-Run + UI/UX + Optional Slurm

Use any merge example directory below as `<example>`:

```bash
geryon launch --experiment <example>/experiment.yaml --out <example>/outputs --run-id demo --batch-size 8 --backend local --dry-run --format json
geryon status --run <example>/outputs/runs/demo --format table
geryon run-slurm --run <example>/outputs/runs/demo --partition debug --time-min 20 --dry-run --format json
```

Queue commands after real Slurm submission:

```bash
geryon queue --run <example>/outputs/runs/demo
geryon queue-refresh --run <example>/outputs/runs/demo
```

## Cleanup

```bash
./geryon/examples/merge_policies/cleanup.sh
```
