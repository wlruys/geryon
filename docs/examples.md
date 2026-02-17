# Examples

Examples live under `examples/`.

## Suites

- `usage/`: five canonical workflows for planning, execution, resilience, and recovery.
- `merge_policies/`: targeted merge-mode and strategy fixtures.

## Common Usage Map

- `01_basics_and_sweeps`: schema basics, packs, defaults/tags, and cartesian sweeps.
- `02_constraints_and_filtering`: allowlist/blocklist constraints plus predicates.
- `03_composition_and_variants`: imports, groups, and `run_sets` variants.
- `04_execution_resilience`: timeout/retry/fail-fast with progress UI profile defaults.
- `05_workflow_and_recovery`: rerun selection, retry-file execution, and recover flow.

## Merge Policy Map

- `01_conflict_behavior`: merge-mode conflict resolution with priorities.
- `02_key_strategies`: path-level merge strategies.
- `03_intentional_overrides`: explicit higher-priority overrides.
- `04_duplicate_pack_extension`: duplicate pack-name extension behavior.
- `05_delete_sentinel`: deletion via merge sentinel value.
- `06_policy_gates`: strict merge-mode guardrail behavior.

## Operator Features Demonstrated

Examples include command paths for:

- validation + dry-run,
- output UX (`--format table|json`, status/report),
- optional Slurm submission (`run-slurm --dry-run`) and queue status checks.

## Run an Example

```bash
cd examples/usage/01_basics_and_sweeps
./launch.sh
```

## Cleanup

```bash
bash examples/cleanup.sh
```
