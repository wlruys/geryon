from __future__ import annotations

import itertools
import json
import logging
import os
import re
import shlex
from collections.abc import Container
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from geryon.config_compose import compose_experiment_data, list_run_set_names
from geryon.models import (
    ConfigError,
    ExcludeConstraint,
    ExperimentSpec,
    MergePolicy,
    OptionSpec,
    ParameterPredicateSpec,
    PackSpec,
    PlanSummary,
    PlannedBatch,
    PlannedConfig,
    PredicateArgSpec,
)
from geryon.store import ArtifactStore
from geryon.utils import (
    flatten_dict,
    render_hydra_override,
    sanitize_for_path,
    sha1_hex,
    stable_json,
    utc_now_iso,
)

_log = logging.getLogger("geryon.planner")


def _validate_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _option_key(option: OptionSpec) -> str:
    return stable_json(
        {
            "id": option.option_id,
            "tag": option.tag,
            "params": option.params,
        }
    )


_MERGE_MODES = {"none", "merge"}
_KEY_STRATEGIES = {"error", "replace", "deep_merge", "append_unique", "set_union"}
_MERGE_ROOT_KEYS = {"mode", "strategies", "delete_sentinel"}
_PREDICATE_ON_ERROR_BEHAVIORS = {"error", "false"}
_PREDICATE_OPERATORS = {
    "and",
    "or",
    "not",
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "in",
    "not_in",
    "between",
}
_ARG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MISSING = object()


def _normalize_str_list(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list of strings")
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f"{label}[{index}] must be a string")
        trimmed = item.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        out.append(trimmed)
    return tuple(out)


def _normalize_behavior(
    raw: Any,
    *,
    label: str,
    allowed: set[str],
    aliases: Mapping[str, str] | None = None,
) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    normalized = raw.strip()
    mapped = dict(aliases or {}).get(normalized, normalized)
    if mapped not in allowed:
        raise ConfigError(f"{label} must be one of {sorted(allowed)}")
    return mapped


def _normalize_key_strategies(raw: Any, *, label: str) -> dict[str, str]:
    if raw is None:
        return {}
    key_strategies_map = _validate_mapping(raw, label=label)
    key_strategies: dict[str, str] = {}
    for raw_path, raw_strategy in key_strategies_map.items():
        path = str(raw_path).strip()
        if not path:
            raise ConfigError(f"{label} keys must be non-empty strings")
        strategy = _normalize_behavior(
            raw_strategy,
            label=f"{label}['{path}']",
            allowed=_KEY_STRATEGIES,
        )
        key_strategies[path] = strategy
    return key_strategies


def _normalize_merge_policy(
    data: Mapping[str, Any],
) -> tuple[MergePolicy, dict[str, Any], dict[str, Any]]:
    merge_raw = data.get("merge", {})
    if merge_raw is None:
        merge_raw = {}
    merge_map = _validate_mapping(merge_raw, label="merge")

    unknown_keys = sorted(set(str(key) for key in merge_map.keys()) - _MERGE_ROOT_KEYS)
    if unknown_keys:
        raise ConfigError(f"merge has unknown keys: {unknown_keys}")

    mode = _normalize_behavior(
        merge_map.get("mode", "merge"),
        label="merge.mode",
        allowed=_MERGE_MODES,
    )

    key_strategies = _normalize_key_strategies(
        merge_map.get("strategies", {}), label="merge.strategies"
    )

    delete_sentinel = merge_map.get("delete_sentinel", "__delete__")
    if not isinstance(delete_sentinel, str) or not delete_sentinel.strip():
        raise ConfigError("merge.delete_sentinel must be a non-empty string")
    delete_sentinel = delete_sentinel.strip()

    merge_input = deepcopy(dict(merge_map))
    merge_effective = {
        "mode": mode,
        "strategies": dict(key_strategies),
        "delete_sentinel": delete_sentinel,
    }
    return (
        MergePolicy(
            mode=mode,
            key_strategies=key_strategies,
            delete_sentinel=delete_sentinel,
        ),
        merge_input,
        merge_effective,
    )


def _normalize_packs(
    data: Mapping[str, Any],
    *,
    merge_policy: MergePolicy,
) -> tuple[tuple[PackSpec, ...], dict[str, Any]]:
    raw_packs = data.get("packs")
    if not isinstance(raw_packs, list) or not raw_packs:
        raise ConfigError("'packs' must be a non-empty list")

    diagnostics: dict[str, Any] = {
        "duplicate_pack_names": [],
        "duplicate_options_removed": [],
        "duplicate_option_ids": [],
        "duplicate_tags": [],
        "pack_catalog": {},
    }

    pack_name_to_indices: dict[str, list[int]] = defaultdict(list)
    pack_name_to_priority: dict[str, int] = {}
    merged_pack_options: dict[str, list[OptionSpec]] = defaultdict(list)
    merged_seen_option_keys: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    merged_seen_option_ids: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    merged_seen_tags: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ordered_pack_names: list[str] = []

    for pack_idx, raw_pack in enumerate(raw_packs):
        pack_map = _validate_mapping(raw_pack, label=f"packs[{pack_idx}]")
        unknown_pack_keys = sorted(
            set(str(key) for key in pack_map.keys()) - {"name", "options", "priority"}
        )
        if unknown_pack_keys:
            raise ConfigError(
                f"packs[{pack_idx}] has unknown keys: {unknown_pack_keys}"
            )

        name = pack_map.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"packs[{pack_idx}].name must be a non-empty string")
        name = name.strip()

        priority_raw = pack_map.get("priority", 0)
        if not isinstance(priority_raw, int):
            raise ConfigError(f"packs[{pack_idx}].priority must be an integer")
        priority = int(priority_raw)

        if name in pack_name_to_indices and merge_policy.mode == "none":
            raise ConfigError(
                f"Duplicate pack '{name}' at packs[{pack_idx}] is not allowed "
                f"under merge.mode=none. Use merge.mode=merge to allow duplicate pack names."
            )

        options = pack_map.get("options")
        if not isinstance(options, list) or not options:
            raise ConfigError(f"Pack '{name}' options must be a non-empty list")

        if name not in ordered_pack_names:
            ordered_pack_names.append(name)
        pack_name_to_indices[name].append(pack_idx)
        pack_name_to_priority[name] = priority

        for option_idx, option_data in enumerate(options):
            option = _normalize_option(
                name,
                option_data,
                option_idx,
                priority=priority,
            )
            option_key = _option_key(option)
            merged_key = merged_seen_option_keys[name].get(option_key)
            if merged_key is not None:
                diagnostics["duplicate_options_removed"].append(
                    {
                        "pack": name,
                        "canonical_pack_index": merged_key["pack_index"],
                        "canonical_option_index": merged_key["option_index"],
                        "duplicate_pack_index": pack_idx,
                        "duplicate_option_index": option_idx,
                        "option_id": option.option_id,
                        "tag": option.tag,
                    }
                )
                continue

            reindexed_option = OptionSpec(
                pack_name=name,
                index=len(merged_pack_options[name]),
                params=deepcopy(option.params),
                option_id=option.option_id,
                tag=option.tag,
                priority=priority,
            )
            merged_pack_options[name].append(reindexed_option)
            merged_seen_option_keys[name][option_key] = {
                "pack_index": pack_idx,
                "option_index": option_idx,
            }

            merged_seen_option_ids[name][reindexed_option.option_id].append(
                reindexed_option.index
            )
            if reindexed_option.tag:
                merged_seen_tags[name][reindexed_option.tag].append(
                    reindexed_option.index
                )

    for pack_name, indices in sorted(pack_name_to_indices.items()):
        if len(indices) > 1:
            diagnostics["duplicate_pack_names"].append(
                {"pack": pack_name, "pack_indices": indices}
            )

    for pack_name, option_map in sorted(merged_seen_option_ids.items()):
        for option_id, indices in sorted(option_map.items()):
            if len(indices) > 1:
                diagnostics["duplicate_option_ids"].append(
                    {
                        "pack": pack_name,
                        "option_id": option_id,
                        "option_indices": indices,
                    }
                )

    for pack_name, tag_map in sorted(merged_seen_tags.items()):
        for tag, indices in sorted(tag_map.items()):
            if len(indices) > 1:
                diagnostics["duplicate_tags"].append(
                    {"pack": pack_name, "tag": tag, "option_indices": indices}
                )

    packs: list[PackSpec] = []
    for pack_name in ordered_pack_names:
        options = tuple(merged_pack_options[pack_name])
        priority = pack_name_to_priority.get(pack_name, 0)
        packs.append(PackSpec(name=pack_name, options=options, priority=priority))
        diagnostics["pack_catalog"][pack_name] = [
            {
                "index": option.index,
                "id": option.option_id,
                "tag": option.tag,
                "priority": option.priority,
            }
            for option in options
        ]

    if diagnostics["duplicate_option_ids"]:
        raise ConfigError(
            f"Duplicate option ids detected: {diagnostics['duplicate_option_ids']}"
        )

    return tuple(packs), diagnostics


def _assert_no_dotted_keys(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            if "." in key:
                raise ConfigError(
                    f"{label} key '{key}' is invalid: dotted keys are not allowed"
                )
            _assert_no_dotted_keys(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_dotted_keys(item, label=f"{label}[{index}]")


def _normalize_defaults(
    data: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    defaults = data.get("defaults", {})
    if defaults is None:
        return {}, tuple()
    defaults_map = _validate_mapping(defaults, label="defaults")
    unknown_defaults = sorted(
        set(str(key) for key in defaults_map.keys()) - {"params", "tags"}
    )
    if unknown_defaults:
        raise ConfigError(f"defaults has unknown keys: {unknown_defaults}")

    raw_params = defaults_map.get("params", {})
    if raw_params is None:
        raw_params = {}
    raw_params_map = _validate_mapping(raw_params, label="defaults.params")
    _assert_no_dotted_keys(raw_params_map, label="defaults.params")

    params = deepcopy(dict(raw_params_map))

    raw_tags = defaults_map.get("tags", [])
    if raw_tags is None:
        raw_tags = []
    if not isinstance(raw_tags, list):
        raise ConfigError("defaults.tags must be a list")

    tags = tuple(str(tag) for tag in raw_tags if str(tag).strip())
    return params, tags


def _normalize_option(
    pack_name: str,
    option_data: Any,
    index: int,
    *,
    priority: int = 0,
) -> OptionSpec:
    option_map = _validate_mapping(
        option_data, label=f"packs[{pack_name}] option #{index}"
    )

    unknown_option_keys = sorted(set(option_map.keys()) - {"id", "tag", "params"})
    if unknown_option_keys:
        raise ConfigError(
            f"Pack '{pack_name}' option #{index}: unknown keys {unknown_option_keys}"
        )

    option_id = option_map.get("id")
    if not isinstance(option_id, str) or not option_id.strip():
        raise ConfigError(
            f"Pack '{pack_name}' option #{index}: 'id' must be a non-empty string"
        )
    option_id = option_id.strip()

    tag = option_map.get("tag")
    if tag is not None and not isinstance(tag, str):
        raise ConfigError(f"Pack '{pack_name}' option #{index}: 'tag' must be a string")
    if isinstance(tag, str):
        tag = tag.strip() or None

    if "params" not in option_map:
        raise ConfigError(f"Pack '{pack_name}' option #{index}: 'params' is required")
    params = option_map["params"]
    params_map = _validate_mapping(
        params, label=f"Pack '{pack_name}' option #{index} params"
    )
    _assert_no_dotted_keys(
        params_map, label=f"packs[{pack_name}].options[{index}].params"
    )
    normalized_params = deepcopy(dict(params_map))

    return OptionSpec(
        pack_name=pack_name,
        index=index,
        params=normalized_params,
        option_id=option_id,
        tag=tag,
        priority=priority,
    )


def _normalize_predicate_operand(value: Any, *, label: str) -> Any:
    if isinstance(value, Mapping):
        raise ConfigError(f"{label} cannot be a mapping")
    if isinstance(value, list):
        return [
            _normalize_predicate_operand(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and value.startswith("$"):
        var_name = value[1:]
        if not _ARG_NAME_RE.fullmatch(var_name):
            raise ConfigError(f"{label} has invalid variable reference '{value}'")
        return f"${var_name}"
    return value


def _normalize_predicate_expr(expr: Any, *, label: str) -> Any:
    if isinstance(expr, Mapping):
        expr_map = _validate_mapping(expr, label=label)
        if len(expr_map) != 1:
            raise ConfigError(f"{label} must contain exactly one operator")
        raw_op, raw_operand = next(iter(expr_map.items()))
        op = str(raw_op)
        if op not in _PREDICATE_OPERATORS:
            raise ConfigError(f"{label} has unsupported operator '{op}'")

        if op in {"and", "or"}:
            if not isinstance(raw_operand, list) or not raw_operand:
                raise ConfigError(f"{label}.{op} must be a non-empty list")
            return {
                op: [
                    _normalize_predicate_expr(item, label=f"{label}.{op}[{index}]")
                    for index, item in enumerate(raw_operand)
                ]
            }

        if op == "not":
            return {op: _normalize_predicate_expr(raw_operand, label=f"{label}.not")}

        if not isinstance(raw_operand, list):
            raise ConfigError(f"{label}.{op} must be a list")

        expected = 3 if op == "between" else 2
        if len(raw_operand) != expected:
            raise ConfigError(f"{label}.{op} must have {expected} operands")
        return {
            op: [
                _normalize_predicate_operand(item, label=f"{label}.{op}[{index}]")
                for index, item in enumerate(raw_operand)
            ]
        }

    return _normalize_predicate_operand(expr, label=label)


def _collect_predicate_variable_refs(node: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(node, str) and node.startswith("$"):
        refs.add(node[1:])
        return refs
    if isinstance(node, list):
        for item in node:
            refs.update(_collect_predicate_variable_refs(item))
        return refs
    if isinstance(node, Mapping):
        for value in node.values():
            refs.update(_collect_predicate_variable_refs(value))
    return refs


def _normalize_predicates(
    raw_predicates: Any,
    *,
    known_pack_names: set[str],
) -> tuple[tuple[ParameterPredicateSpec, ...], dict[str, Any]]:
    if raw_predicates is None:
        raw_predicates = []
    if not isinstance(raw_predicates, list):
        raise ConfigError("constraints.predicates must be a list")

    seen_ids: set[str] = set()
    predicates: list[ParameterPredicateSpec] = []
    unknown_predicate_pack_refs: list[dict[str, Any]] = []

    for predicate_index, raw_predicate in enumerate(raw_predicates):
        label_prefix = f"constraints.predicates[{predicate_index}]"
        predicate_map = _validate_mapping(raw_predicate, label=label_prefix)
        unknown_keys = sorted(
            set(str(key) for key in predicate_map.keys())
            - {"id", "args", "expr", "on_error"}
        )
        if unknown_keys:
            raise ConfigError(f"{label_prefix} has unknown keys: {unknown_keys}")

        predicate_id_raw = predicate_map.get("id")
        if not isinstance(predicate_id_raw, str) or not predicate_id_raw.strip():
            raise ConfigError(f"{label_prefix}.id must be a non-empty string")
        predicate_id = predicate_id_raw.strip()
        if predicate_id in seen_ids:
            raise ConfigError(f"Duplicate constraints.predicates id '{predicate_id}'")
        seen_ids.add(predicate_id)

        args_raw = predicate_map.get("args")
        args_map = _validate_mapping(args_raw, label=f"{label_prefix}.args")
        if not args_map:
            raise ConfigError(f"{label_prefix}.args must not be empty")

        arg_specs: list[PredicateArgSpec] = []
        arg_names: set[str] = set()
        for raw_arg_name, raw_arg_spec in args_map.items():
            arg_name = str(raw_arg_name).strip()
            if not _ARG_NAME_RE.fullmatch(arg_name):
                raise ConfigError(
                    f"{label_prefix}.args has invalid arg name '{raw_arg_name}'. "
                    "Use pattern [A-Za-z_][A-Za-z0-9_]*"
                )
            if arg_name in arg_names:
                raise ConfigError(f"{label_prefix}.args has duplicate arg '{arg_name}'")
            arg_names.add(arg_name)

            arg_spec_map = _validate_mapping(
                raw_arg_spec, label=f"{label_prefix}.args.{arg_name}"
            )
            unknown_arg_spec_keys = sorted(
                set(str(key) for key in arg_spec_map.keys())
                - {"pack_id", "param", "default"}
            )
            if unknown_arg_spec_keys:
                raise ConfigError(
                    f"{label_prefix}.args.{arg_name} has unknown keys: {unknown_arg_spec_keys}"
                )

            has_pack_id = "pack_id" in arg_spec_map
            has_param = "param" in arg_spec_map
            if has_pack_id == has_param:
                raise ConfigError(
                    f"{label_prefix}.args.{arg_name} must define exactly one of pack_id or param"
                )

            has_default = "default" in arg_spec_map
            default_value = deepcopy(arg_spec_map.get("default"))

            if has_pack_id:
                pack_name_raw = arg_spec_map.get("pack_id")
                if not isinstance(pack_name_raw, str) or not pack_name_raw.strip():
                    raise ConfigError(
                        f"{label_prefix}.args.{arg_name}.pack_id must be a non-empty string"
                    )
                pack_name = pack_name_raw.strip()
                if pack_name not in known_pack_names:
                    unknown_predicate_pack_refs.append(
                        {
                            "predicate_id": predicate_id,
                            "arg": arg_name,
                            "pack": pack_name,
                        }
                    )
                arg_specs.append(
                    PredicateArgSpec(
                        name=arg_name,
                        source_kind="pack_id",
                        source_key=pack_name,
                        has_default=has_default,
                        default_value=default_value,
                    )
                )
                continue

            param_raw = arg_spec_map.get("param")
            if not isinstance(param_raw, str) or not param_raw.strip():
                raise ConfigError(
                    f"{label_prefix}.args.{arg_name}.param must be a non-empty string"
                )
            param_path = param_raw.strip()
            parts = param_path.split(".")
            if any(not part for part in parts):
                raise ConfigError(
                    f"{label_prefix}.args.{arg_name}.param has invalid dotted path '{param_path}'"
                )
            arg_specs.append(
                PredicateArgSpec(
                    name=arg_name,
                    source_kind="param",
                    source_key=param_path,
                    has_default=has_default,
                    default_value=default_value,
                )
            )

        if "expr" not in predicate_map:
            raise ConfigError(f"{label_prefix}.expr is required")
        expr = _normalize_predicate_expr(
            predicate_map.get("expr"), label=f"{label_prefix}.expr"
        )
        unknown_expr_args = sorted(_collect_predicate_variable_refs(expr) - arg_names)
        if unknown_expr_args:
            raise ConfigError(
                f"{label_prefix}.expr references unknown args: {unknown_expr_args}. "
                f"Known args: {sorted(arg_names)}"
            )

        on_error_raw = predicate_map.get("on_error", "error")
        if isinstance(on_error_raw, bool):
            on_error_raw = "false" if on_error_raw is False else "error"
        on_error = _normalize_behavior(
            on_error_raw,
            label=f"{label_prefix}.on_error",
            allowed=_PREDICATE_ON_ERROR_BEHAVIORS,
        )

        predicates.append(
            ParameterPredicateSpec(
                predicate_id=predicate_id,
                args=tuple(arg_specs),
                expr=expr,
                on_error=on_error,
            )
        )

    diagnostics = {
        "predicate_ids": [predicate.predicate_id for predicate in predicates],
        "unknown_predicate_pack_refs": unknown_predicate_pack_refs,
    }
    return tuple(predicates), diagnostics


def _normalize_constraints(
    data: Mapping[str, Any],
    *,
    packs: tuple[PackSpec, ...],
) -> tuple[
    tuple[ExcludeConstraint, ...], tuple[ParameterPredicateSpec, ...], dict[str, Any]
]:
    constraints_raw = data.get("constraints", {})
    if constraints_raw is None:
        return (
            tuple(),
            tuple(),
            {
                "unknown_constraint_packs": [],
                "unknown_constraint_values": [],
                "unknown_predicate_pack_refs": [],
                "predicate_ids": [],
            },
        )
    constraints_map = _validate_mapping(constraints_raw, label="constraints")

    unknown_constraints_keys = sorted(
        set(str(key) for key in constraints_map.keys())
        - {"include", "exclude", "predicates"}
    )
    if unknown_constraints_keys:
        raise ConfigError(f"constraints has unknown keys: {unknown_constraints_keys}")

    excludes_raw = constraints_map.get("exclude", [])
    if excludes_raw is None:
        excludes_raw = []
    if not isinstance(excludes_raw, list):
        raise ConfigError("constraints.exclude must be a list")

    includes_raw = constraints_map.get("include", [])
    if includes_raw is None:
        includes_raw = []
    if not isinstance(includes_raw, list):
        raise ConfigError("constraints.include must be a list")

    known_pack_names = {pack.name for pack in packs}
    known_identifiers: dict[str, set[str]] = {
        pack.name: set().union(*(option.identifiers for option in pack.options))
        for pack in packs
    }
    unknown_constraint_packs: list[dict[str, Any]] = []
    unknown_constraint_values: list[dict[str, Any]] = []

    def _collect_rules(
        raw_rules: list[Any], *, mode: str, label_prefix: str
    ) -> list[ExcludeConstraint]:
        rules: list[ExcludeConstraint] = []
        for idx, raw_rule in enumerate(raw_rules):
            rule_map = _validate_mapping(raw_rule, label=f"{label_prefix}[{idx}]")
            unknown_rule_keys = sorted(
                set(str(key) for key in rule_map.keys()) - {"when"}
            )
            if unknown_rule_keys:
                raise ConfigError(
                    f"{label_prefix}[{idx}] has unknown keys: {unknown_rule_keys}"
                )
            when_raw = rule_map.get("when")
            when_map = _validate_mapping(when_raw, label=f"{label_prefix}[{idx}].when")

            normalized: dict[str, tuple[str, ...]] = {}
            for pack_name, values in when_map.items():
                pack_name = str(pack_name)
                if isinstance(values, list):
                    normalized_values = tuple(str(v) for v in values)
                else:
                    normalized_values = (str(values),)

                normalized[pack_name] = normalized_values

                if pack_name not in known_pack_names:
                    unknown_constraint_packs.append(
                        {"rule_index": idx, "pack": pack_name, "mode": mode}
                    )
                    continue

                known = known_identifiers.get(pack_name, set())
                unknown_values = sorted(
                    {value for value in normalized_values if value not in known}
                )
                if unknown_values:
                    unknown_constraint_values.append(
                        {
                            "rule_index": idx,
                            "pack": pack_name,
                            "unknown_values": unknown_values,
                            "mode": mode,
                        }
                    )
            rules.append(ExcludeConstraint(when=normalized, mode=mode))
        return rules

    rules: list[ExcludeConstraint] = []
    rules.extend(
        _collect_rules(includes_raw, mode="include", label_prefix="constraints.include")
    )
    rules.extend(
        _collect_rules(excludes_raw, mode="exclude", label_prefix="constraints.exclude")
    )

    predicates, predicate_diagnostics = _normalize_predicates(
        constraints_map.get("predicates", []),
        known_pack_names=known_pack_names,
    )

    diagnostics = {
        "unknown_constraint_packs": unknown_constraint_packs,
        "unknown_constraint_values": unknown_constraint_values,
        "unknown_predicate_pack_refs": list(
            predicate_diagnostics.get("unknown_predicate_pack_refs", [])
        ),
        "predicate_ids": list(predicate_diagnostics.get("predicate_ids", [])),
    }
    return tuple(rules), predicates, diagnostics


def _resolve_base_command(data: Mapping[str, Any]) -> str:
    command_raw = data.get("command")
    command_map = _validate_mapping(command_raw, label="command")
    unknown_command_keys = sorted(
        set(str(key) for key in command_map.keys()) - {"program", "args"}
    )
    if unknown_command_keys:
        raise ConfigError(f"command has unknown keys: {unknown_command_keys}")

    program = command_map.get("program")
    if not isinstance(program, str) or not program.strip():
        raise ConfigError("command.program must be a non-empty string")

    args = command_map.get("args", [])
    if args is None:
        args = []
    if not isinstance(args, list):
        raise ConfigError("command.args must be a list")

    argv = [program] + [str(arg) for arg in args]
    return shlex.join(argv)


def parse_experiment_yaml_with_diagnostics(
    path: str | Path,
    *,
    run_set: str | None = None,
) -> tuple[ExperimentSpec, dict[str, Any]]:
    experiment_path = Path(path)
    raw, composition_diagnostics = compose_experiment_data(
        experiment_path, run_set=run_set
    )
    if not isinstance(raw, Mapping):
        raise ConfigError("Composed experiment configuration must be a mapping")

    allowed_top_keys = {
        "schema",
        "command",
        "defaults",
        "constraints",
        "merge",
        "packs",
    }
    unknown_top_keys = sorted(set(str(key) for key in raw.keys()) - allowed_top_keys)
    if unknown_top_keys:
        raise ConfigError(
            f"Composed experiment has unknown top-level keys: {unknown_top_keys}"
        )

    merge_policy, merge_input, merge_effective = _normalize_merge_policy(raw)
    base_command = _resolve_base_command(raw)
    packs, pack_diagnostics = _normalize_packs(raw, merge_policy=merge_policy)
    defaults_params, defaults_tags = _normalize_defaults(raw)
    constraints, predicates, constraint_diagnostics = _normalize_constraints(
        raw, packs=packs
    )

    unknown_packs = list(constraint_diagnostics.get("unknown_constraint_packs", []))
    unknown_values = list(constraint_diagnostics.get("unknown_constraint_values", []))
    unknown_predicate_packs = list(
        constraint_diagnostics.get("unknown_predicate_pack_refs", [])
    )
    if unknown_packs or unknown_values or unknown_predicate_packs:
        raise ConfigError(
            "Unknown constraint references detected: "
            f"unknown_packs={unknown_packs}, unknown_values={unknown_values}, "
            f"unknown_predicate_pack_refs={unknown_predicate_packs}"
        )

    spec = ExperimentSpec(
        base_command=base_command,
        packs=packs,
        defaults_params=defaults_params,
        defaults_tags=defaults_tags,
        constraints=constraints,
        predicates=predicates,
        merge_policy=merge_policy,
    )
    diagnostics: dict[str, Any] = {
        "composition": composition_diagnostics,
        "merge_input": merge_input,
        "merge_effective": merge_effective,
        "packs": pack_diagnostics,
        "constraints": constraint_diagnostics,
    }
    return spec, diagnostics


def parse_experiment_yaml(
    path: str | Path, *, run_set: str | None = None
) -> ExperimentSpec:
    spec, _ = parse_experiment_yaml_with_diagnostics(path, run_set=run_set)
    return spec


def get_experiment_run_sets(path: str | Path) -> list[str]:
    return list_run_set_names(path)


def _register_provenance(
    provenance: dict[tuple[str, ...], tuple[str, int]],
    value: Any,
    path: tuple[str, ...],
    source: str,
    priority: int,
) -> None:
    provenance[path] = (source, priority)
    if isinstance(value, dict):
        for key, sub_value in value.items():
            _register_provenance(
                provenance, sub_value, path + (str(key),), source, priority
            )


def _remove_registered_provenance(
    provenance: dict[tuple[str, ...], tuple[str, int]], path: tuple[str, ...]
) -> None:
    to_remove = [
        item for item in provenance.keys() if item == path or item[: len(path)] == path
    ]
    for item in to_remove:
        provenance.pop(item, None)


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"


def _dedupe_list_preserve(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in values:
        key = stable_json(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(deepcopy(item))
    return out


def _set_union_sorted(values: list[Any]) -> list[Any]:
    keyed: dict[str, Any] = {}
    for item in values:
        key = stable_json(item)
        if key not in keyed:
            keyed[key] = deepcopy(item)
    return [keyed[key] for key in sorted(keyed.keys())]


def _is_delete_value(value: Any, *, sentinel: str) -> bool:
    return isinstance(value, str) and value == sentinel


@dataclass(frozen=True)
class _MergeSource:
    label: str
    priority: int = 0


def _record_merge_event(
    diagnostics: dict[str, Any] | None,
    *,
    kind: str,
    path: tuple[str, ...],
    previous_source: str | None,
    current_source: str,
    previous_value: Any,
    current_value: Any,
) -> None:
    if diagnostics is None:
        return

    path_str = _format_path(path)
    counters = diagnostics.setdefault("counters", {})
    path_counts = diagnostics.setdefault("path_counts", {})
    samples = diagnostics.setdefault("samples", {})

    counters[kind] = int(counters.get(kind, 0)) + 1
    if kind not in path_counts:
        path_counts[kind] = {}
    path_counts[kind][path_str] = int(path_counts[kind].get(path_str, 0)) + 1

    if kind not in samples:
        samples[kind] = []
    if len(samples[kind]) < 64:
        samples[kind].append(
            {
                "path": path_str,
                "previous_source": previous_source,
                "current_source": current_source,
                "previous_value": previous_value,
                "current_value": current_value,
            }
        )


def _merge_option_params(
    target: dict[str, Any],
    updates: Mapping[str, Any],
    *,
    source: _MergeSource,
    provenance: dict[tuple[str, ...], tuple[str, int]],
    merge_policy: MergePolicy,
    diagnostics: dict[str, Any] | None = None,
    path: tuple[str, ...] = (),
) -> None:
    """Priority-based merge of option params into target.

    provenance maps path -> (source_label, priority).
    Defaults have priority = -inf (represented by no entry in provenance).
    """
    for key, value in updates.items():
        key = str(key)
        current_path = path + (key,)
        current_path_str = _format_path(current_path)
        strategy = merge_policy.key_strategies.get(current_path_str)

        if _is_delete_value(value, sentinel=merge_policy.delete_sentinel):
            prev_info = provenance.get(current_path)
            previous_source = prev_info[0] if prev_info else None
            if key in target:
                previous_value = deepcopy(target[key])
                target.pop(key, None)
                _remove_registered_provenance(provenance, current_path)
                _record_merge_event(
                    diagnostics,
                    kind="delete_applied",
                    path=current_path,
                    previous_source=previous_source or "defaults",
                    current_source=source.label,
                    previous_value=previous_value,
                    current_value=None,
                )
            else:
                _record_merge_event(
                    diagnostics,
                    kind="delete_noop",
                    path=current_path,
                    previous_source=previous_source,
                    current_source=source.label,
                    previous_value=None,
                    current_value=None,
                )
            continue

        if key not in target:
            target[key] = deepcopy(value)
            _register_provenance(
                provenance, value, current_path, source.label, source.priority
            )
            continue

        existing = target[key]
        prev_info = provenance.get(current_path)
        previous_source = prev_info[0] if prev_info else None
        previous_priority = prev_info[1] if prev_info else None  # None means defaults

        if strategy:
            if strategy == "error":
                raise ConfigError(
                    "Parameter overlap blocked by merge.strategies:\n"
                    f"  path: {_format_path(current_path)}\n"
                    f"  strategy: error\n"
                    f"  previous source: {previous_source or 'defaults'} value={existing!r}\n"
                    f"  current source : {source.label} value={value!r}"
                )

            if strategy == "replace":
                _record_merge_event(
                    diagnostics,
                    kind="strategy_replace",
                    path=current_path,
                    previous_source=previous_source or "defaults",
                    current_source=source.label,
                    previous_value=deepcopy(existing),
                    current_value=deepcopy(value),
                )
                target[key] = deepcopy(value)
                _remove_registered_provenance(provenance, current_path)
                _register_provenance(
                    provenance, value, current_path, source.label, source.priority
                )
                continue

            if strategy == "deep_merge":
                if not isinstance(existing, dict) or not isinstance(value, Mapping):
                    raise ConfigError(
                        "merge.strategies requested deep_merge but values are not both mappings:\n"
                        f"  path: {_format_path(current_path)}\n"
                        f"  existing type: {type(existing).__name__}\n"
                        f"  incoming type: {type(value).__name__}"
                    )
                nested_source = _MergeSource(
                    label=source.label,
                    priority=source.priority,
                )
                _merge_option_params(
                    existing,
                    value,
                    source=nested_source,
                    provenance=provenance,
                    merge_policy=merge_policy,
                    diagnostics=diagnostics,
                    path=current_path,
                )
                continue

            if strategy == "append_unique":
                if not isinstance(existing, list) or not isinstance(value, list):
                    raise ConfigError(
                        "merge.strategies requested append_unique but values are not both lists:\n"
                        f"  path: {_format_path(current_path)}"
                    )
                merged_list = _dedupe_list_preserve([*existing, *value])
                _record_merge_event(
                    diagnostics,
                    kind="strategy_append_unique",
                    path=current_path,
                    previous_source=previous_source or "defaults",
                    current_source=source.label,
                    previous_value=deepcopy(existing),
                    current_value=deepcopy(merged_list),
                )
                target[key] = merged_list
                _remove_registered_provenance(provenance, current_path)
                _register_provenance(
                    provenance, merged_list, current_path, source.label, source.priority
                )
                continue

            if strategy == "set_union":
                if not isinstance(existing, list) or not isinstance(value, list):
                    raise ConfigError(
                        "merge.strategies requested set_union but values are not both lists:\n"
                        f"  path: {_format_path(current_path)}"
                    )
                merged_set = _set_union_sorted([*existing, *value])
                _record_merge_event(
                    diagnostics,
                    kind="strategy_set_union",
                    path=current_path,
                    previous_source=previous_source or "defaults",
                    current_source=source.label,
                    previous_value=deepcopy(existing),
                    current_value=deepcopy(merged_set),
                )
                target[key] = merged_set
                _remove_registered_provenance(provenance, current_path)
                _register_provenance(
                    provenance, merged_set, current_path, source.label, source.priority
                )
                continue

            raise ConfigError(
                f"Unsupported key strategy '{strategy}' at path {_format_path(current_path)}"
            )

        if isinstance(existing, dict) and isinstance(value, Mapping):
            _merge_option_params(
                existing,
                value,
                source=source,
                provenance=provenance,
                merge_policy=merge_policy,
                diagnostics=diagnostics,
                path=current_path,
            )
            continue

        # Defaults always overridden by any pack
        if previous_source is None:
            if existing != value:
                _record_merge_event(
                    diagnostics,
                    kind="override_default",
                    path=current_path,
                    previous_source="defaults",
                    current_source=source.label,
                    previous_value=deepcopy(existing),
                    current_value=deepcopy(value),
                )
            target[key] = deepcopy(value)
            _register_provenance(
                provenance, value, current_path, source.label, source.priority
            )
            continue

        if merge_policy.mode == "none":
            raise ConfigError(
                "Parameter conflict detected (merge.mode=none does not allow overlaps):\n"
                f"  path: {_format_path(current_path)}\n"
                f"  previous source: {previous_source} value={existing!r}\n"
                f"  current source : {source.label} value={value!r}\n"
                "  hint: use merge.mode=merge with priority to resolve conflicts"
            )

        # Equal values are always allowed in merge mode.
        if existing == value:
            _record_merge_event(
                diagnostics,
                kind="equal_overlap",
                path=current_path,
                previous_source=previous_source,
                current_source=source.label,
                previous_value=deepcopy(existing),
                current_value=deepcopy(value),
            )
            continue

        # mode == "merge": resolve by priority
        assert previous_priority is not None
        if source.priority > previous_priority:
            _record_merge_event(
                diagnostics,
                kind="priority_override",
                path=current_path,
                previous_source=previous_source,
                current_source=source.label,
                previous_value=deepcopy(existing),
                current_value=deepcopy(value),
            )
            target[key] = deepcopy(value)
            _remove_registered_provenance(provenance, current_path)
            _register_provenance(
                provenance, value, current_path, source.label, source.priority
            )
        elif source.priority == previous_priority:
            raise ConfigError(
                "Parameter conflict at equal priority:\n"
                f"  path: {_format_path(current_path)}\n"
                f"  previous source: {previous_source} (priority={previous_priority}) value={existing!r}\n"
                f"  current source : {source.label} (priority={source.priority}) value={value!r}\n"
                "  hint: assign different priorities to resolve the conflict"
            )
        else:
            # Lower priority — keep existing value
            _record_merge_event(
                diagnostics,
                kind="priority_kept",
                path=current_path,
                previous_source=previous_source,
                current_source=source.label,
                previous_value=deepcopy(existing),
                current_value=deepcopy(value),
            )


def _matches_exclusion(
    selected: Mapping[str, OptionSpec],
    rules: Iterable[ExcludeConstraint],
) -> bool:
    def _rule_matches(rule: ExcludeConstraint) -> bool:
        matched = True
        for pack_name, expected_values in rule.when.items():
            option = selected.get(pack_name)
            if option is None:
                matched = False
                break
            if option.identifiers.isdisjoint(set(expected_values)):
                matched = False
                break
        return matched

    include_rules = [rule for rule in rules if rule.mode == "include"]
    exclude_rules = [rule for rule in rules if rule.mode != "include"]

    if include_rules and not any(_rule_matches(rule) for rule in include_rules):
        return True
    for rule in exclude_rules:
        if _rule_matches(rule):
            return True
    return False


class _PredicateEvaluationError(RuntimeError):
    pass


def _lookup_param_value(params: Mapping[str, Any], dotted_path: str) -> Any:
    cursor: Any = params
    for part in dotted_path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return _MISSING
        cursor = cursor[part]
    return cursor


def _format_predicate_context(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def _resolve_predicate_operand(value: Any, *, args: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        arg_name = value[1:]
        return args.get(arg_name, _MISSING)
    if isinstance(value, list):
        resolved: list[Any] = []
        for item in value:
            item_value = _resolve_predicate_operand(item, args=args)
            if item_value is _MISSING:
                return _MISSING
            resolved.append(item_value)
        return resolved
    return value


def _evaluate_predicate_expr(expr: Any, *, args: Mapping[str, Any]) -> bool:
    if isinstance(expr, Mapping):
        op, operand = next(iter(expr.items()))

        if op == "and":
            results = [_evaluate_predicate_expr(item, args=args) for item in operand]
            return all(results)

        if op == "or":
            results = [_evaluate_predicate_expr(item, args=args) for item in operand]
            return any(results)

        if op == "not":
            return not _evaluate_predicate_expr(operand, args=args)

        values = [_resolve_predicate_operand(item, args=args) for item in operand]
        if any(value is _MISSING for value in values):
            operand_render = _format_predicate_context(operand)
            args_render = _format_predicate_context(args)
            raise _PredicateEvaluationError(
                "predicate operand resolved to missing argument value "
                f"(op={op}, operand={operand_render}, args={args_render})"
            )

        left = values[0]
        right = values[1] if len(values) > 1 else None
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "lt":
            try:
                return left < right
            except TypeError as exc:
                raise _PredicateEvaluationError(
                    f"lt comparison failed: {left!r} < {right!r}"
                ) from exc
        if op == "lte":
            try:
                return left <= right
            except TypeError as exc:
                raise _PredicateEvaluationError(
                    f"lte comparison failed: {left!r} <= {right!r}"
                ) from exc
        if op == "gt":
            try:
                return left > right
            except TypeError as exc:
                raise _PredicateEvaluationError(
                    f"gt comparison failed: {left!r} > {right!r}"
                ) from exc
        if op == "gte":
            try:
                return left >= right
            except TypeError as exc:
                raise _PredicateEvaluationError(
                    f"gte comparison failed: {left!r} >= {right!r}"
                ) from exc
        if op == "in":
            try:
                if right is None:
                    raise TypeError("right operand is None")
                return left in cast(Container[object], right)
            except TypeError as exc:
                raise _PredicateEvaluationError(
                    f"in comparison failed: {left!r} in {right!r}"
                ) from exc
        if op == "not_in":
            try:
                if right is None:
                    raise TypeError("right operand is None")
                return left not in cast(Container[object], right)
            except TypeError as exc:
                raise _PredicateEvaluationError(
                    f"not_in comparison failed: {left!r} not in {right!r}"
                ) from exc
        if op == "between":
            target = values[0]
            lower = values[1]
            upper = values[2]
            try:
                return lower <= target <= upper
            except TypeError as exc:
                raise _PredicateEvaluationError(
                    f"between comparison failed: {lower!r} <= {target!r} <= {upper!r}"
                ) from exc

        raise _PredicateEvaluationError(f"unsupported predicate operator '{op}'")

    raw = _resolve_predicate_operand(expr, args=args)
    if raw is _MISSING:
        expr_render = _format_predicate_context(expr)
        args_render = _format_predicate_context(args)
        raise _PredicateEvaluationError(
            "predicate expression resolved to missing argument value "
            f"(expr={expr_render}, args={args_render})"
        )
    if not isinstance(raw, bool):
        raise _PredicateEvaluationError(
            f"predicate expression leaf must resolve to bool, got {type(raw).__name__}"
        )
    return raw


def _evaluate_predicate(
    predicate: ParameterPredicateSpec,
    *,
    selected: Mapping[str, OptionSpec],
    merged_params: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    resolved_args: dict[str, Any] = {}
    for arg in predicate.args:
        value: Any = _MISSING
        if arg.source_kind == "pack_id":
            option = selected.get(arg.source_key)
            if option is not None:
                value = option.option_id
        elif arg.source_kind == "param":
            value = _lookup_param_value(merged_params, arg.source_key)
        else:
            raise _PredicateEvaluationError(
                f"unsupported predicate arg source '{arg.source_kind}'"
            )

        if value is _MISSING:
            if arg.has_default:
                resolved_args[arg.name] = deepcopy(arg.default_value)
                continue
            raise _PredicateEvaluationError(
                f"missing predicate arg '{arg.name}' from source {arg.source_kind}:{arg.source_key}"
            )
        resolved_args[arg.name] = value

    return _evaluate_predicate_expr(predicate.expr, args=resolved_args), resolved_args


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "tag"


def _build_wandb_name(tags: tuple[str, ...], config_id: str) -> str:
    prefix = "_".join(_sanitize_tag(tag) for tag in tags) if tags else "exp"
    name = f"{prefix}_{config_id[:8]}"
    return name[:128]


def _render_command(base_command: str, params: Mapping[str, Any]) -> str:
    flat = flatten_dict(params)
    overrides = [render_hydra_override(key, flat[key]) for key in sorted(flat.keys())]
    if not overrides:
        return base_command
    return f"{base_command} {' '.join(overrides)}"


def _build_configs(
    spec: ExperimentSpec,
    *,
    run_id: str,
    batch_size: int,
) -> tuple[list[PlannedConfig], dict[str, Any], dict[str, Any]]:
    combos = itertools.product(*[pack.options for pack in spec.packs])
    merge_diagnostics: dict[str, Any] = {
        "counters": {},
        "path_counts": {},
        "samples": {},
    }
    predicate_diagnostics: dict[str, Any] = {
        "counters": {
            "evaluated": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "dropped_candidates": 0,
        },
        "by_predicate": {
            predicate.predicate_id: {
                "evaluated": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
            }
            for predicate in spec.predicates
        },
        "drop_samples": [],
        "error_samples": [],
    }

    merge_policy = spec.merge_policy

    planned: list[PlannedConfig] = []
    for combo_index, combo in enumerate(combos):
        selected = {option.pack_name: option for option in combo}
        if _matches_exclusion(selected, spec.constraints):
            continue
        selected_labels = {item.pack_name: item.label for item in combo}

        merged: dict[str, Any] = deepcopy(spec.defaults_params)
        provenance: dict[tuple[str, ...], tuple[str, int]] = {}

        # Sort by priority ascending so higher priority applies last and wins
        sorted_combo = sorted(combo, key=lambda opt: opt.priority)
        for option in sorted_combo:
            source = _MergeSource(
                label=f"{option.pack_name}[{option.index}]",
                priority=option.priority,
            )
            try:
                _merge_option_params(
                    merged,
                    option.params,
                    source=source,
                    provenance=provenance,
                    merge_policy=merge_policy,
                    diagnostics=merge_diagnostics,
                )
            except ConfigError as exc:
                raise ConfigError(
                    f"{exc}\n  selected options: {selected_labels}"
                ) from exc

        drop_reason: str | None = None
        for predicate in spec.predicates:
            predicate_diagnostics["counters"]["evaluated"] += 1
            per_predicate = predicate_diagnostics["by_predicate"][
                predicate.predicate_id
            ]
            per_predicate["evaluated"] += 1
            resolved_args: dict[str, Any] = {}
            try:
                passed, resolved_args = _evaluate_predicate(
                    predicate,
                    selected=selected,
                    merged_params=merged,
                )
            except _PredicateEvaluationError as exc:
                predicate_diagnostics["counters"]["errors"] += 1
                per_predicate["errors"] += 1
                sample_errors = predicate_diagnostics["error_samples"]
                if len(sample_errors) < 64:
                    sample_errors.append(
                        {
                            "predicate_id": predicate.predicate_id,
                            "selected_options": selected_labels,
                            "resolved_args": deepcopy(resolved_args),
                            "error": str(exc),
                            "on_error": predicate.on_error,
                            "expr": deepcopy(predicate.expr),
                        }
                    )
                elif len(sample_errors) == 64:
                    _log.warning(
                        "Predicate error samples capped at 64; "
                        "further errors will not be recorded in diagnostics"
                    )
                    sample_errors.append({"truncated": True})
                if predicate.on_error == "false":
                    passed = False
                else:
                    raise ConfigError(
                        "Predicate evaluation failed:\n"
                        f"  predicate_id: {predicate.predicate_id}\n"
                        f"  selected options: {selected_labels}\n"
                        f"  resolved args: {_format_predicate_context(resolved_args)}\n"
                        f"  expr: {_format_predicate_context(predicate.expr)}\n"
                        f"  error: {exc}"
                    ) from exc

            if passed:
                predicate_diagnostics["counters"]["passed"] += 1
                per_predicate["passed"] += 1
                continue

            predicate_diagnostics["counters"]["failed"] += 1
            per_predicate["failed"] += 1
            drop_reason = f"predicate:{predicate.predicate_id}"
            break

        if drop_reason is not None:
            predicate_diagnostics["counters"]["dropped_candidates"] += 1
            sample_drops = predicate_diagnostics["drop_samples"]
            if len(sample_drops) < 64:
                sample_drops.append(
                    {
                        "reason": drop_reason,
                        "predicate_id": predicate.predicate_id,
                        "selected_options": selected_labels,
                        "resolved_args": deepcopy(resolved_args),
                        "expr": deepcopy(predicate.expr),
                    }
                )
            elif len(sample_drops) == 64:
                _log.warning(
                    "Predicate drop samples capped at 64; "
                    "further drops will not be recorded in diagnostics"
                )
                sample_drops.append({"truncated": True})
            continue

        tags = sorted(
            {*spec.defaults_tags, *(option.tag for option in combo if option.tag)}
        )
        canonical = {"params": merged, "tags": tags}
        config_id = sha1_hex(canonical)
        wandb_name = _build_wandb_name(tuple(tags), config_id)

        params_with_wandb = deepcopy(merged)
        wandb_section = params_with_wandb.get("wandb", {})
        if wandb_section is None:
            wandb_section = {}
        if not isinstance(wandb_section, Mapping):
            raise ConfigError("wandb must be a mapping when present")
        wandb_dict = dict(wandb_section)
        wandb_dict["name"] = wandb_name
        wandb_dict["tags"] = tags
        params_with_wandb["wandb"] = wandb_dict

        batch_index = combo_index // batch_size
        line_index = combo_index % batch_size

        planned.append(
            PlannedConfig(
                run_id=run_id,
                config_id=config_id,
                batch_index=batch_index,
                line_index=line_index,
                command=_render_command(spec.base_command, params_with_wandb),
                params=params_with_wandb,
                tags=tuple(tags),
                wandb_name=wandb_name,
                selected_options=selected_labels,
            )
        )

    # Re-index batches/lines after exclusion filtering.
    reindexed: list[PlannedConfig] = []
    for index, cfg in enumerate(planned):
        reindexed.append(
            PlannedConfig(
                run_id=cfg.run_id,
                config_id=cfg.config_id,
                batch_index=index // batch_size,
                line_index=index % batch_size,
                command=cfg.command,
                params=cfg.params,
                tags=cfg.tags,
                wandb_name=cfg.wandb_name,
                selected_options=cfg.selected_options,
            )
        )

    return reindexed, merge_diagnostics, predicate_diagnostics


def _default_run_id(*, run_set: str | None = None) -> str:
    stamp = utc_now_iso().replace(":", "").replace("-", "")
    if run_set:
        suffix = sanitize_for_path(run_set)
        return f"run_{stamp}_{suffix}"
    return f"run_{stamp}"


def _sorted_count_items(
    counts: Mapping[str, int], limit: int = 32
) -> list[dict[str, Any]]:
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"path": key, "count": int(value)} for key, value in items[:limit]]


def summarize_plan_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    composition = dict(diagnostics.get("composition", {}))
    composition_registry = dict(composition.get("registry", {}))
    composition_expansion = dict(composition.get("expansion", {}))
    packs = dict(diagnostics.get("packs", {}))
    constraints = dict(diagnostics.get("constraints", {}))
    predicates = dict(diagnostics.get("predicates", {}))
    predicate_counters = dict(predicates.get("counters", {}))
    merge = dict(diagnostics.get("merge", {}))
    counters = dict(merge.get("counters", {}))
    return {
        "composition_files": len(composition.get("files", [])),
        "composition_option_sets": len(composition_registry.get("option_sets", [])),
        "composition_pack_defs": len(composition_registry.get("packs", [])),
        "composition_group_defs": len(composition_registry.get("groups", [])),
        "composition_selected_group_refs": len(
            composition_expansion.get("group_refs", [])
        ),
        "composition_selected_pack_refs": len(
            composition_expansion.get("pack_refs", [])
        ),
        "duplicate_pack_names": len(packs.get("duplicate_pack_names", [])),
        "duplicate_options_removed": len(packs.get("duplicate_options_removed", [])),
        "duplicate_option_ids": len(packs.get("duplicate_option_ids", [])),
        "duplicate_tags": len(packs.get("duplicate_tags", [])),
        "unknown_constraint_packs": len(
            constraints.get("unknown_constraint_packs", [])
        ),
        "unknown_constraint_values": len(
            constraints.get("unknown_constraint_values", [])
        ),
        "unknown_predicate_pack_refs": len(
            constraints.get("unknown_predicate_pack_refs", [])
        ),
        "predicates_declared": len(constraints.get("predicate_ids", [])),
        "predicate_evaluated": int(predicate_counters.get("evaluated", 0)),
        "predicate_passed": int(predicate_counters.get("passed", 0)),
        "predicate_failed": int(predicate_counters.get("failed", 0)),
        "predicate_errors": int(predicate_counters.get("errors", 0)),
        "predicate_dropped_candidates": int(
            predicate_counters.get("dropped_candidates", 0)
        ),
        "override_default_events": int(counters.get("override_default", 0)),
        "equal_overlap_events": int(counters.get("equal_overlap", 0)),
        "priority_override_events": int(counters.get("priority_override", 0)),
        "delete_applied_events": int(counters.get("delete_applied", 0)),
    }


def _render_diagnostics_summary_text(diagnostics: Mapping[str, Any]) -> str:
    summary = summarize_plan_diagnostics(diagnostics)
    merge = dict(diagnostics.get("merge", {}))
    top_override = merge.get("top_override_default_paths", [])
    top_equal = merge.get("top_equal_overlap_paths", [])

    lines = [
        "Plan Diagnostics Summary",
        f"composition_files: {summary['composition_files']}",
        f"composition_option_sets: {summary['composition_option_sets']}",
        f"composition_pack_defs: {summary['composition_pack_defs']}",
        f"composition_group_defs: {summary['composition_group_defs']}",
        f"composition_selected_group_refs: {summary['composition_selected_group_refs']}",
        f"composition_selected_pack_refs: {summary['composition_selected_pack_refs']}",
        f"duplicate_pack_names: {summary['duplicate_pack_names']}",
        f"duplicate_options_removed: {summary['duplicate_options_removed']}",
        f"duplicate_option_ids: {summary['duplicate_option_ids']}",
        f"duplicate_tags: {summary['duplicate_tags']}",
        f"unknown_constraint_packs: {summary['unknown_constraint_packs']}",
        f"unknown_constraint_values: {summary['unknown_constraint_values']}",
        f"unknown_predicate_pack_refs: {summary['unknown_predicate_pack_refs']}",
        f"predicates_declared: {summary['predicates_declared']}",
        f"predicate_evaluated: {summary['predicate_evaluated']}",
        f"predicate_passed: {summary['predicate_passed']}",
        f"predicate_failed: {summary['predicate_failed']}",
        f"predicate_errors: {summary['predicate_errors']}",
        f"predicate_dropped_candidates: {summary['predicate_dropped_candidates']}",
        f"override_default_events: {summary['override_default_events']}",
        f"equal_overlap_events: {summary['equal_overlap_events']}",
        f"priority_override_events: {summary['priority_override_events']}",
        f"delete_applied_events: {summary['delete_applied_events']}",
        "",
        "Top override_default paths:",
    ]
    if top_override:
        lines.extend(
            f"  - {item['path']}: {item['count']}" for item in top_override[:10]
        )
    else:
        lines.append("  - none")

    lines.append("")
    lines.append("Top equal_overlap paths:")
    if top_equal:
        lines.extend(f"  - {item['path']}: {item['count']}" for item in top_equal[:10])
    else:
        lines.append("  - none")
    return "\n".join(lines) + "\n"


def plan_experiment(
    *,
    experiment_path: str | Path,
    out_dir: str | Path,
    batch_size: int,
    run_id: str | None = None,
    run_set: str | None = None,
    dry_run: bool = False,
    preview_count: int = 0,
) -> PlanSummary:
    if batch_size <= 0:
        raise ConfigError("batch_size must be positive")
    if preview_count < 0:
        raise ConfigError("preview_count must be >= 0")

    spec, parse_diagnostics = parse_experiment_yaml_with_diagnostics(
        experiment_path, run_set=run_set
    )
    run_id = run_id or _default_run_id(run_set=run_set)
    run_root = Path(out_dir).resolve() / "runs" / run_id
    store = ArtifactStore(run_root)

    if run_root.exists() and not dry_run:
        raise ConfigError(f"Run directory already exists: {run_root}")

    planned_configs, merge_diagnostics, predicate_diagnostics = _build_configs(
        spec,
        run_id=run_id,
        batch_size=batch_size,
    )
    if not planned_configs:
        constraint_count = len(spec.constraints)
        predicate_count = len(spec.predicates)
        pred_counters = predicate_diagnostics.get("counters", {})
        pred_dropped = pred_counters.get("dropped_candidates", 0)
        merge_counters = merge_diagnostics.get("counters", {})
        merge_excluded = merge_counters.get("constraint_excluded", 0)
        total_combos = merge_counters.get("total_combinations", 0)
        raise ConfigError(
            f"Planning generated zero configurations after applying constraints. "
            f"Total combinations: {total_combos}, "
            f"constraint-excluded: {merge_excluded} ({constraint_count} constraints), "
            f"predicate-dropped: {pred_dropped} ({predicate_count} predicates). "
            f"Use 'validate-config --show-diagnostics' for details."
        )

    num_batches = 1 + planned_configs[-1].batch_index
    planned_batches: list[PlannedBatch] = []
    for batch_index in range(num_batches):
        records = [cfg for cfg in planned_configs if cfg.batch_index == batch_index]
        planned_batches.append(
            PlannedBatch(
                batch_index=batch_index,
                path=str(Path("plan") / "batches" / f"batch_{batch_index:03d}.txt"),
                num_commands=len(records),
            )
        )

    merge_path_counts = merge_diagnostics.get("path_counts", {})
    plan_diagnostics = {
        "summary": {
            "num_packs": len(spec.packs),
            "num_constraints": len(spec.constraints),
            "num_configs": len(planned_configs),
            "num_batches": num_batches,
        },
        "composition": parse_diagnostics.get("composition", {}),
        "merge_input": parse_diagnostics.get("merge_input", {}),
        "merge_effective": parse_diagnostics.get("merge_effective", {}),
        "run_set": dict(parse_diagnostics.get("composition", {})).get("run_set", {}),
        "packs": parse_diagnostics.get("packs", {}),
        "constraints": parse_diagnostics.get("constraints", {}),
        "predicates": predicate_diagnostics,
        "merge": {
            "counters": merge_diagnostics.get("counters", {}),
            "top_override_default_paths": _sorted_count_items(
                merge_path_counts.get("override_default", {})
            ),
            "top_equal_overlap_paths": _sorted_count_items(
                merge_path_counts.get("equal_overlap", {})
            ),
            "samples": merge_diagnostics.get("samples", {}),
        },
    }

    if not dry_run:
        store.ensure_plan_layout()
        store.ensure_exec_layout()

        for batch in planned_batches:
            lines = [
                cfg.command
                for cfg in planned_configs
                if cfg.batch_index == batch.batch_index
            ]
            store.batch_file(batch.batch_index).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

        store.write_planned_configs(planned_configs)
        store.write_planned_batches(planned_batches)
        store.write_manifest(planned_configs)
        store.plan_run_set_path.write_text(
            json.dumps(
                dict(parse_diagnostics.get("composition", {})).get("run_set", {}),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        store.plan_diagnostics_path.write_text(
            json.dumps(plan_diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        store.plan_diagnostics_summary_path.write_text(
            _render_diagnostics_summary_text(plan_diagnostics),
            encoding="utf-8",
        )

        run_meta = {
            "run_id": run_id,
            "run_root": str(store.run_root),
            "created_at": utc_now_iso(),
            "experiment_path": str(Path(experiment_path).resolve()),
            "run_set": run_set,
            "batch_size": batch_size,
            "schema_version": 3,
            "base_command": spec.base_command,
            "num_configs": len(planned_configs),
            "num_batches": num_batches,
            "diagnostics_path": str(store.plan_diagnostics_path),
            "diagnostics_summary_path": str(store.plan_diagnostics_summary_path),
            "planner_version": "v2",
            "host_user": os.environ.get("USER")
            or os.environ.get("USERNAME")
            or "unknown",
        }
        store.write_run_meta(run_meta)

        snapshot_path = store.plan_dir / "experiment.snapshot.yaml"
        snapshot_path.write_text(
            Path(experiment_path).read_text(encoding="utf-8"), encoding="utf-8"
        )

    preview_configs: tuple[dict[str, Any], ...] = tuple(
        cfg.to_json() for cfg in planned_configs[:preview_count]
    )

    return PlanSummary(
        run_id=run_id,
        run_root=run_root,
        total_configs=len(planned_configs),
        total_batches=num_batches,
        preview_configs=preview_configs,
    )
