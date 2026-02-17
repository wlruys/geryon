# Merge Policy Example Suite

Each directory in this suite is a runnable experiment fixture:

- `01_conflict_behavior/`: `merge.mode=merge` with priority-based conflict resolution
- `02_key_strategies/`: path-level merge strategies via `merge.strategies`
- `03_intentional_overrides/`: intentional non-equal overrides via higher-priority pack
- `04_duplicate_pack_extension/`: duplicate pack names auto-merge in `mode=merge`
- `05_delete_sentinel/`: key deletion via `merge.delete_sentinel`
- `06_policy_gates/`: `merge.mode=none` policy gate (intentionally fails at plan time)

Shared utilities live under `../common/`.

## Run A Single Example

```bash
cd geryon/examples/merge_policies/01_conflict_behavior
./launch.sh
```

## Cleanup

```bash
./cleanup.sh
```

## Run From Tests

`tests/test_examples_merge_policies.py` executes all launch scripts as integration checks.
