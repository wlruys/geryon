# Example Runtime Configs

Shared profiles used by example suites.

## Files

- `profiles.yaml`: reusable profile presets (`local_*`, `slurm_*`).
- `env/bootstrap.example.sh`: optional environment bootstrap script.

## Quick Start

```bash
uv run geryon list-profiles \
  --profiles-file ./geryon/examples/configs/profiles.yaml
```

Local run with profile defaults:

```bash
uv run geryon run-local \
  --run ./geryon/examples/common_usage/01_basics_and_sweeps/outputs/runs/01_basics_and_sweeps_demo \
  --profiles-file ./geryon/examples/configs/profiles.yaml \
  --profile local_dev
```

Optional Slurm dry-run:

```bash
uv run geryon run-slurm \
  --run ./geryon/examples/common_usage/01_basics_and_sweeps/outputs/runs/01_basics_and_sweeps_demo \
  --profiles-file ./geryon/examples/configs/profiles.yaml \
  --profile slurm_cpu_debug \
  --dry-run \
  --format json
```

Queue status after real submission:

```bash
uv run geryon queue --run ./geryon/examples/common_usage/01_basics_and_sweeps/outputs/runs/01_basics_and_sweeps_demo
uv run geryon queue-refresh --run ./geryon/examples/common_usage/01_basics_and_sweeps/outputs/runs/01_basics_and_sweeps_demo
```

## Site Customization

Adjust cluster-specific settings (`partition`, `account`, `qos`, `constraint`, modules)
for your environment before real Slurm submissions.
