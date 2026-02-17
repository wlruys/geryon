# Experiment YAML Schema (v4)

This is the configuration contract consumed by:

- `geryon validate-config`
- `geryon inspect-config`
- `geryon plan`

Geryon processes experiment files in two stages:

1. Composition (`geryon.config_compose`): handles `imports`, `option_sets`, `packs`, `groups`, `select`, and optional `run_sets` overlays.
2. Planning (`geryon.planner`): validates/normalizes `command`, `defaults`, `constraints`, `merge`, and final expanded `packs`.

## Top-Level Keys

Allowed top-level keys:

- `schema`
- `imports`
- `option_sets`
- `packs`
- `groups`
- `command`
- `select`
- `defaults`
- `constraints`
- `merge`
- `run_sets`

Removed in v4: `registry`, `package`.

## `schema`

```yaml
schema:
  version: 4
  unknown_key_behavior: error  # error | warn
```

Rules:

- `version` must be `4`.
- `unknown_key_behavior` must be `error` or `warn`.
- If omitted, defaults are injected (`version: 4`, `unknown_key_behavior: error`).

## `command` (required)

```yaml
command:
  program: python3
  args: [train.py]
```

Rules:

- `program`: required non-empty string.
- `args`: optional list of strings (default `[]`).

## `select` (composition stage)

```yaml
select:
  groups: [presets.core]
  packs:
    - ref: presets.architecture
    - name: seed
      options:
        - id: s1
          params: {seed: 1}
```

Selector forms:

- `{group: <ref>}`
- `{ref: <ref>, ...overrides...}`
- Inline pack mapping with `name` + options.

Selector keys:

- `ref`
- `name`
- `priority`
- `replace_options`
- `options_from`
- `options`
- `filter` (`include_ids`, `exclude_ids`)

## Final `packs` Contract (planner stage)

After composition, planner receives expanded `packs`.

Pack keys:

- `name` (required)
- `options` (required non-empty list)
- `priority` (optional integer, default `0`)

Option keys:

- `id` (required non-empty string)
- `params` (required mapping)
- `tag` (optional string)

Rules:

- Dotted keys are rejected under `params`.
- Duplicate pack names:
  - `merge.mode: merge` -> allowed, options are merged.
  - `merge.mode: none` -> error.
- Duplicate option ids within a merged pack are errors.

## `defaults` (optional)

```yaml
defaults:
  params:
    train:
      epochs: 100
  tags: [project:baseline]
```

Rules:

- Allowed keys: `params`, `tags`.
- `params` must be a mapping; dotted keys are rejected.
- `tags` must be a list.

## `constraints`

Allowed keys:

- `include`
- `exclude`
- `predicates`

Matching is ID-based for selected options.

## `merge`

Canonical merge schema:

```yaml
merge:
  mode: merge                    # none | merge
  strategies: {}                 # path -> error|replace|deep_merge|append_unique|set_union
  delete_sentinel: "__delete__" # exact-value delete marker
```

### Modes

- `none`:
  - Any overlap between packs is an error (equal values included).
  - Duplicate pack names are rejected.
- `merge` (default):
  - Conflicts resolve by pack `priority`.
  - Higher priority wins.
  - Equal priority + different values is an error.
  - Equal values are allowed.

Defaults are implicitly lowest priority and can be overridden by any pack.

### Strategies

Supported per-path strategies:

- `error`
- `replace`
- `deep_merge`
- `append_unique`
- `set_union`

### Delete sentinel

If a value equals `merge.delete_sentinel`, that key is removed.

## Removed Merge Features (breaking)

The following are removed:

- Modes: `strict`, `balanced`, `permissive`
- `merge.allow.*`
- `merge.advanced.*`
- `pack.extend`
- `pack.override_paths` / `option.override_paths`

Use pack `priority` with `merge.mode: merge` instead.
