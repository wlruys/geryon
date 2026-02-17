# Python DSL (`geryon.dsl`)

`geryon.dsl` builds canonical schema-v4 experiment documents in Python.

Exports:

- `Experiment`
- Specs: `Option`, `OptionSource`, `Pack`, `PackSelector`, `DefaultsSpec`, `MergeSpec`, `ConstraintRule`, `PredicateArg`, `Predicate`, `RunSetSpec`
- Helpers: `pack_map`, `match_ids`, `pack_param_values`, `pack_param_linspace`, `pack_param_logspace`

## Core Builder: `Experiment`

Key mutators:

- `command(program, *args)`
- `defaults(params=..., tags=...)`
- `merge(spec=None, **kwargs)`
- `select_group(group_ref)`
- `add_pack(pack)`
- `add_pack_selector(selector)`
- `add_pack_ref(...)`
- `include(...)`, `exclude(...)`, `add_predicate(...)`
- definition helpers: `add_option_set`, `add_pack_def`, `add_group_def`
- run sets: `add_run_set`, `run_set`, `variant`

Emission helpers:

- `to_dict(strict=True)`
- `to_yaml(strict=True, sort_keys=False)`
- `to_yaml_file(...)`

## Spec Highlights

### `Option`

```python
Option(option_id="resnet18", params={"model": {"name": "resnet18"}}, tag="arch-resnet")
```

- `option_id` required.
- `params` required mapping (dotted keys rejected).
- `tag` optional.

### `Pack`

```python
Pack(name="architecture", priority=10, options=(Option(...),))
```

- `name` required.
- `priority` optional integer (default `0`).
- At least one of `options` or `options_from` must be set.

### `PackSelector`

```python
PackSelector(
  ref="presets.architecture",
  name="arch_ablation",
  priority=20,
  replace_options=True,
  options_from=(OptionSource(ref="presets.arch", include_ids=("resnet18",)),),
  options=(Option(...),),
)
```

### `MergeSpec`

```python
MergeSpec(
  mode="merge",  # none | merge
  strategies={"callbacks": "append_unique"},
  delete_sentinel="__delete__",
)
```

Allowed keys:

- `mode`: `none | merge`
- `strategies`: `error | replace | deep_merge | append_unique | set_union`
- `delete_sentinel`

### `RunSetSpec`

```python
RunSetSpec(
  extends=("baseline",),
  append={"merge": {"mode": "merge"}},
  replace={"select": {"packs": [...]}},
)
```

`append`/`replace` roots are limited to `select`, `defaults`, `constraints`, `merge`.

## Template Helpers

- `pack_param_values`: one path + iterable values -> `Pack`
- `pack_param_linspace`: linear spacing variant
- `pack_param_logspace`: logarithmic spacing variant
- `pack_map`: mapping `option_id -> params` or `option_id -> {params, tag}`
- `match_ids`: builds predicate for allowed `(left_id, right_id)` pairings

## Notes

- `strict=True` validates the generated document schema.
- `validate_with_geryon=True` in `to_yaml_file` runs planner-level semantic validation.
