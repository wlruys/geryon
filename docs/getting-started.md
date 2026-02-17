# Getting Started

## Install

Requirements:

- Python `>=3.10`
- `PyYAML`, `rich` (installed from `pyproject.toml`)

Install:

```bash
cd geryon
uv sync
```

CLI entry point:

- `geryon` (maps to `geryon.cli:main`)

Optional extras:

- Slurm support: `uv sync --extra slurm`
- PyLauncher executor: `uv sync --extra pylauncher`

## First Experiment

Create `experiment.yaml`:

```yaml
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('hello')"]
select:
  packs:
    - name: seed
      options:
        - id: s1
          params:
            seed: 1
        - id: s2
          params:
            seed: 2
```

Validate + inspect composed config:

```bash
geryon validate-config --experiment ./experiment.yaml --show-diagnostics
geryon inspect-config --experiment ./experiment.yaml --format yaml --show-diagnostics
```

Launch in one step:

```bash
geryon launch \
  --experiment ./experiment.yaml \
  --out ./outputs \
  --run-id demo \
  --batch-size 8 \
  --backend local \
  --profile local_dev \
  --profiles-file ./profiles.yaml
```

Plan artifacts (manual flow):

```bash
geryon plan \
  --experiment ./experiment.yaml \
  --out ./outputs \
  --run-id demo \
  --batch-size 8
```

Run locally (manual flow):

```bash
geryon run-local --run ./outputs/runs/demo --profile local_dev --profiles-file ./profiles.yaml
```

Check status:

```bash
geryon status --run ./outputs/runs/demo
geryon report --run ./outputs/runs/demo --format markdown --out ./report.md
```

## Standard Workflow

1. `validate-config` before edits are committed.
2. `inspect-config` when using imports/registries/run-sets.
3. `launch` for the default happy path.
4. `plan` + `run-local`/`run-slurm` when you need split stages.
5. `status` and `report` during/after runs.
6. `recover` for retry execution (or `rerun` + `run-* --retry-file`).
7. `collect` for summary JSON.

## Aliases

Deprecated aliases still available:

- `build` -> `plan`
- `local` -> `run-local`
- `slurm` -> `run-slurm`
