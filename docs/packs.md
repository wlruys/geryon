# Composition and Packs

This page describes how reusable YAML is expanded into final planner `packs`.

Implementation source: `geryon.config_compose.compose_experiment_data`.

## Composition Pipeline

1. Load root file.
2. Recursively load `imports` (cycle-checked).
3. Register definitions from each file:
   - `option_sets`
   - `packs`
   - `groups`
4. Optionally apply a run-set overlay (`extends` + `append` + `replace` + `patch_packs`).
5. Expand `select.groups` and `select.packs` into concrete pack entries.
6. Drop composition-only keys and hand final `packs` to planner.

## Pack Definition Shape

```yaml
packs:
  optimizer_pack:
    name: optimizer
    priority: 10
    options_from:
      - ref: common.optimizers
```

Allowed pack keys:

- `name` (required)
- `priority` (optional integer, default `0`)
- `options_from` (optional list)
- `options` (optional list)

A pack must resolve to at least one option.

## Pack Selector Shape (`select.packs`)

Selector forms:

- Group selector: `{group: <ref>}`
- Pack ref selector: `{ref: <ref>, ...}`
- Inline pack selector: same fields as a pack definition

For ref selectors, allowed override keys:

- `name`
- `priority`
- `replace_options`
- `options_from`
- `options`
- `filter.include_ids`
- `filter.exclude_ids`

## Priority and Merge

`priority` is consumed by planner merge logic:

- Higher priority wins conflicts in `merge.mode: merge`.
- Equal priority + different values is an error.
- In `merge.mode: none`, any overlap between packs is an error.

Duplicate pack names:

- `merge.mode: merge`: duplicate names auto-merge into one pack.
- `merge.mode: none`: duplicate names are rejected.

## Option Sources and Filters

`options_from` entries:

- `ref` (required)
- `include_ids` (optional)
- `exclude_ids` (optional)

`filter` on selectors applies after all options are collected. If filtering removes all options, composition fails.

## Diagnostics

`compose_experiment_data` emits diagnostics with:

- loaded files/imports
- definition registry contents
- expanded group refs, pack refs, and option sources
- final pack count
- run-set overlay resolution details
