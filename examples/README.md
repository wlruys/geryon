# Examples

The example suite is split into:

- `usage/`: five canonical teaching examples for day-to-day workflows.
- `merge_policies/`: focused examples for merge semantics and policy behavior.

## Teaching Coverage

Across these examples you can learn and practice:

- Planning and local execution.
- Dry-run and validation before execution.
- UI/UX output modes (`table`/`json`, status/report views, progress dashboard via profile defaults).
- Recovery workflows (`rerun`, `--retry-file`, `recover`).
- Optional Slurm flows (`run-slurm --dry-run`, `--query-status`, `queue`, `queue-refresh`).

## Quick Start

```bash
cd geryon/examples/usage/01_basics_and_sweeps
./launch.sh
```

## Cleanup

```bash
./geryon/examples/cleanup.sh
./geryon/examples/cleanup.sh --exec-only
```
