# Common Usage Examples

Five canonical, standalone examples that teach the core Geryon workflow.

| Directory | Teaches | Local profile |
| --- | --- | --- |
| `01_basics_and_sweeps/` | schema basics, packs, cartesian sweeps, defaults/tags | `local_dev` |
| `02_constraints_and_filtering/` | include/exclude constraints and predicates | `local_throughput` |
| `03_composition_and_variants/` | imports, groups, and run-set variants | `local_dev` |
| `04_execution_resilience/` | timeout/retry/fail-fast and progress UI | `local_resiliency` |
| `05_workflow_and_recovery/` | rerun, retry-file targeting, and recover flow | `local_fail_fast` / `local_recover` |

## Common Operator Commands

Each example README includes tailored commands for these workflows:

- Validation + dry-run planning.
- UI/UX output modes (`--format table|json`, report/status commands).
- Optional Slurm submission (`run-slurm --dry-run`) and queue checks (`queue`, `queue-refresh`).

Example queue check commands (after real Slurm submission):

```bash
geryon queue --run ./outputs/runs/<run_id>
geryon queue-refresh --run ./outputs/runs/<run_id>
```

## Run One Example

```bash
cd geryon/examples/usage/01_basics_and_sweeps
./launch.sh
```

## Suite Cleanup

```bash
./geryon/examples/usage/cleanup.sh
```
