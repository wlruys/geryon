# Examples

Available suites:

- `common_usage/`: practical patterns — getting started, execution, error handling, composition.
- `merge_policies/`: merge and policy behavior validation examples.

Shared helper scripts are under `geryon/examples/common/`:

- `run_example.sh`: wrapper for stage-based example execution (`launch` by default, plus `validate`, `plan`, `run`).
- `cleanup_example.sh`: remove generated artifacts for one example.

## Shared Profiles

Reusable machine/runtime presets for examples are in:

- `geryon/examples/configs/profiles.yaml`
- `geryon/examples/configs/env/bootstrap.example.sh`

Inspect available profiles:

```bash
uv run geryon list-profiles \
  --profiles-file ./geryon/examples/configs/profiles.yaml
```

Launch an example locally with a profile:

```bash
uv run geryon launch \
  --experiment ./geryon/examples/common_usage/05_hello_world/experiment.yaml \
  --out ./geryon/examples/common_usage/05_hello_world/outputs \
  --batch-size 8 \
  --profiles-file ./geryon/examples/configs/profiles.yaml \
  --profile local_throughput
```

Recover failed configs with one command:

```bash
uv run geryon recover \
  --run ./geryon/examples/common_usage/05_hello_world/outputs/runs/<run_id> \
  --profiles-file ./geryon/examples/configs/profiles.yaml \
  --profile local_throughput
```

Submit any planned example to Slurm with a profile:

```bash
uv run geryon run-slurm \
  --run ./geryon/examples/common_usage/05_hello_world/outputs/runs/<run_id> \
  --profiles-file ./geryon/examples/configs/profiles.yaml \
  --profile slurm_gpu_debug
```

Runtime configuration is profile-driven; adjust profile fields/defaults instead of per-run CLI overrides.

## Cleanup

Use cleanup scripts to remove generated outputs and execution logs:

```bash
# clean all examples
./geryon/examples/cleanup.sh

# clean only execution artifacts under outputs/runs/*/exec
./geryon/examples/cleanup.sh --exec-only
```
