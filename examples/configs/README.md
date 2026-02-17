# Example Runtime Configs

This folder contains reusable runtime profiles for example launches across
local machines and Slurm clusters.

## Files

- `profiles.yaml`: profile presets consumed by `--profiles-file`.
- `env/bootstrap.example.sh`: optional shared env bootstrap script referenced by profiles.

## Quick Start

List profiles:

```bash
uv run geryon list-profiles \
  --profiles-file ./geryon/examples/configs/profiles.yaml
```

Use a local profile:

```bash
uv run geryon run-local \
  --run ./geryon/examples/common_usage/05_hello_world/outputs/runs/<run_id> \
  --profiles-file ./geryon/examples/configs/profiles.yaml \
  --profile local_dev
```

Use a Slurm profile:

```bash
uv run geryon run-slurm \
  --run ./geryon/examples/common_usage/05_hello_world/outputs/runs/<run_id> \
  --profiles-file ./geryon/examples/configs/profiles.yaml \
  --profile slurm_cpu_debug
```

## Site Customization Notes

- Update `partition`, `account`, `qos`, `constraint`, and `time_min` per site policy.
- Replace module commands and environment activation in `bootstrap.example.sh`.
- Keep secrets out of this repository; inject credentials via environment at runtime.
