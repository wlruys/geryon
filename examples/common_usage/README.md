# Common Usage Examples

This suite contains runnable, practical experiment patterns:

## Getting Started

- `05_hello_world/`: minimal experiment — one pack, two options, two configs.
- `06_defaults_and_tags/`: shared default parameters and per-option tags.
- `07_include_exclude/`: include/exclude constraints to prune the cartesian product.

## Planning & Validation

- `08_dry_run_and_validation/`: `validate-config`, `inspect-config`, and `launch --dry-run` before committing.

## Execution

- `09_parallel_execution/`: running configs in parallel with profile defaults (`defaults.run_local.max_concurrent_tasks`).

## Error Handling & Recovery

- `10_timeout_and_retry/`: configure `command_timeout_sec` and `max_retries` in profiles for retry/timeout policy.
- `11_resume_and_rerun/`: full recovery lifecycle — `launch` → `status` → `recover`.
- `12_fail_fast/`: configure `max_failures` and `fail_fast_threshold` in profiles to stop early.

## Composition & Advanced Patterns

- `01_reused_packs/`: reusable definitions split across files (`imports`, `option_sets`/`packs`/`groups`).
- `02_ablation_study/`: baseline + ablation variants with `run_sets`.
- `03_parameter_sweep/`: plain cartesian hyperparameter sweep.
- `04_controlled_arch_hparams/`: controlled architecture comparison with matched hyperparameter profiles via `constraints.predicates`.
- `13_dsl_parameter_sweep_predicates/`: build `experiment.yaml` from `geryon.dsl`, then apply predicate filtering over a parameter sweep.

## Shared Utilities

- `flaky_app.py`: test application that simulates failures, timeouts, and flaky behavior (used by examples 10-12).

All examples that don't use `flaky_app.py` execute the shared dummy app at
`geryon/examples/common/dummy_hydra_app.py`.
During execution, geryon also injects `hydra.run.dir=...` so each launched command gets
a dedicated work directory under `outputs/runs/<run_id>/exec/workdirs/`.

## Run One Example

```bash
cd geryon/examples/common_usage/05_hello_world
./launch.sh
```

## Cleanup

```bash
# from this suite
./cleanup.sh

# from a single example directory
./cleanup.sh
```

## Ablation Run-Sets

`02_ablation_study/experiment.yaml` defines:

- `baseline`
- `no_augmentation`
- `no_dropout`
- `no_aug_no_dropout`

Run a specific variant:

```bash
uv run geryon plan \
  --experiment ./geryon/examples/common_usage/02_ablation_study/experiment.yaml \
  --run-set no_dropout \
  --out ./geryon/examples/common_usage/02_ablation_study/outputs \
  --batch-size 8
```
