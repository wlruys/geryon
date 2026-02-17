# Geryon

Geryon is a deterministic experiment planner + executor for pack-based sweeps.

Core capabilities:

- Compose experiments from reusable YAML definitions (`imports`, `option_sets`, `packs`, `groups`, `select`).
- Plan deterministic config/batch artifacts (`plan/configs.jsonl`, `plan/batches.jsonl`, manifests, diagnostics).
- Execute locally (`process`, `tmux`, `pylauncher`) or on Slurm via Submitit.
- Track status from append-only result records and generate retry selections.
- Generate experiments from a typed Python DSL (`geryon.dsl`).

## Lifecycle

1. Compose: load imports, resolve refs, expand groups/packs/options into canonical planner input.
2. Validate: normalize schema and policy contracts (merge, constraints, predicates, run-set overlays).
3. Plan: cartesian product, merge params, apply constraints/predicates, emit artifacts.
4. Execute: launch commands with retry/timeout/fail-fast policy.
5. Observe: status/report/collect over result JSONL files.
6. Recover: `recover` (or `rerun` + `run-* --retry-file`) retries failed/terminated/missing configs.

## Documentation Map

- **Getting Started**: install + first run.
- **Experiment YAML / Schema Reference**: complete schema-v4 contract.
- **Experiment YAML / Composition and Packs**: import/defs/select resolution rules.
- **Python DSL**: full `geryon.dsl` API and helpers.
- **CLI Reference**: every command and flag.
- **Runtime / Profiles**: `profiles.yaml` schema and precedence.
- **Runtime / Artifacts and Status**: on-disk interfaces and status semantics.
- **Python API**: stable programmatic entry points.
- **Examples**: runnable scenarios in `examples/`.

## Build These Docs

```bash
mkdocs serve
# or
mkdocs build
```

From the repository root: `/Users/wlruys/workspace/scheduling/geryon/geryon`.
