# Examples

Examples live under `/Users/wlruys/workspace/scheduling/geryon/geryon/examples`.

## Suites

- `common_usage/`: typical workflows and operations.
- `merge_policies/`: merge behavior and conflict cases.

## Common Usage Map

- `01_reused_packs`: imports + definitions + groups.
- `02_ablation_study`: run-set variants (`extends`, `replace`).
- `03_parameter_sweep`: cartesian sweep.
- `04_controlled_arch_hparams`: predicate-constrained pairings.
- `05_hello_world`: minimal experiment.
- `06_defaults_and_tags`: defaults + tag behavior.
- `07_include_exclude`: include/exclude constraints.
- `08_dry_run_and_validation`: validate/inspect/launch dry-run flow.
- `09_parallel_execution`: process executor with parallel workers.
- `10_timeout_and_retry`: timeout + retry policy.
- `11_resume_and_rerun`: launch -> status -> recover lifecycle.
- `12_fail_fast`: max failure count/rate stop policy.
- `13_dsl_parameter_sweep_predicates`: generate experiment from `geryon.dsl`.

## Merge Policy Map

- `01_conflict_behavior`: `merge.mode=merge` resolved via pack priorities.
- `02_key_strategies`: path-level merge strategies.
- `03_intentional_overrides`: intentional override via higher-priority pack.
- `04_duplicate_pack_extension`: duplicate pack names auto-merge in `mode=merge`.
- `05_delete_sentinel`: key deletion with custom sentinel.
- `06_policy_gates`: `merge.mode=none` overlap failure gate.

## Run an Example

```bash
cd /Users/wlruys/workspace/scheduling/geryon/geryon/examples/common_usage/05_hello_world
./launch.sh
```

## Cleanup

```bash
bash /Users/wlruys/workspace/scheduling/geryon/geryon/examples/cleanup.sh
```
