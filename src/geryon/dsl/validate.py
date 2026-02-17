from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from geryon.models import ConfigError

from geryon.dsl.specs import assert_no_dotted_keys

_ALLOWED_TOP_LEVEL_KEYS = {
    "schema",
    "imports",
    "option_sets",
    "packs",
    "groups",
    "command",
    "select",
    "defaults",
    "constraints",
    "merge",
    "run_sets",
}

_ALLOWED_MERGE_KEYS = {"mode", "strategies", "delete_sentinel"}
_ALLOWED_MERGE_MODES = {"none", "merge"}
_ALLOWED_STRATEGIES = {"error", "replace", "deep_merge", "append_unique", "set_union"}

_ALLOWED_PREDICATE_OPERATORS = {
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


def _ensure_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _ensure_non_empty_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _ensure_string_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list")
    out: list[str] = []
    for index, item in enumerate(value):
        out.append(_ensure_non_empty_str(item, label=f"{label}[{index}]"))
    return out


def _validate_imports(imports: Any, *, label: str = "imports") -> None:
    if imports is None:
        return
    if not isinstance(imports, list):
        raise ConfigError(f"{label} must be a list")
    for index, item in enumerate(imports):
        if not isinstance(item, Mapping):
            raise ConfigError(f"{label}[{index}] must be a mapping")
        item_map = _ensure_mapping(item, label=f"{label}[{index}]")
        unknown = sorted(set(item_map.keys()) - {"path", "package"})
        if unknown:
            raise ConfigError(f"{label}[{index}] has unknown keys: {unknown}")
        _ensure_non_empty_str(item_map.get("path"), label=f"{label}[{index}].path")
        if "package" in item_map and item_map.get("package") is not None:
            _ensure_non_empty_str(
                item_map.get("package"), label=f"{label}[{index}].package"
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
        _ensure_non_empty_str(var_name, label=f"{label} variable")
        return f"${var_name}"
    return value


def normalize_predicate_expr(expr: Any, *, label: str) -> Any:
    if isinstance(expr, Mapping):
        expr_map = _ensure_mapping(expr, label=label)
        if len(expr_map) != 1:
            raise ConfigError(f"{label} must contain exactly one operator")
        raw_op, raw_operand = next(iter(expr_map.items()))
        op = _ensure_non_empty_str(raw_op, label=f"{label} operator")
        if op not in _ALLOWED_PREDICATE_OPERATORS:
            raise ConfigError(f"{label} has unsupported operator '{op}'")

        if op in {"and", "or"}:
            if not isinstance(raw_operand, list) or not raw_operand:
                raise ConfigError(f"{label}.{op} must be a non-empty list")
            return {
                op: [
                    normalize_predicate_expr(item, label=f"{label}.{op}[{index}]")
                    for index, item in enumerate(raw_operand)
                ]
            }

        if op == "not":
            return {"not": normalize_predicate_expr(raw_operand, label=f"{label}.not")}

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


def _collect_predicate_refs(node: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(node, str) and node.startswith("$"):
        refs.add(node[1:])
        return refs
    if isinstance(node, list):
        for item in node:
            refs.update(_collect_predicate_refs(item))
        return refs
    if isinstance(node, Mapping):
        for value in node.values():
            refs.update(_collect_predicate_refs(value))
    return refs


def _validate_option(option: Any, *, label: str) -> None:
    option_map = _ensure_mapping(option, label=label)
    unknown = sorted(set(option_map.keys()) - {"id", "tag", "params"})
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {unknown}")

    _ensure_non_empty_str(option_map.get("id"), label=f"{label}.id")
    if "params" not in option_map:
        raise ConfigError(f"{label}.params is required")
    params = _ensure_mapping(option_map["params"], label=f"{label}.params")
    assert_no_dotted_keys(params, label=f"{label}.params")

    tag = option_map.get("tag")
    if tag is not None:
        _ensure_non_empty_str(tag, label=f"{label}.tag")


def _validate_option_source(source: Any, *, label: str) -> None:
    source_map = _ensure_mapping(source, label=label)
    unknown = sorted(set(source_map.keys()) - {"ref", "include_ids", "exclude_ids"})
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {unknown}")
    _ensure_non_empty_str(source_map.get("ref"), label=f"{label}.ref")
    if "include_ids" in source_map:
        _ensure_string_list(source_map.get("include_ids"), label=f"{label}.include_ids")
    if "exclude_ids" in source_map:
        _ensure_string_list(source_map.get("exclude_ids"), label=f"{label}.exclude_ids")


def _validate_pack_selector(pack: Any, *, label: str) -> None:
    pack_map = _ensure_mapping(pack, label=label)
    known_keys = {
        "ref",
        "name",
        "priority",
        "replace_options",
        "options_from",
        "options",
        "filter",
    }
    unknown = sorted(set(pack_map.keys()) - known_keys - {"options", "name"})
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {unknown}")

    if "ref" in pack_map:
        _ensure_non_empty_str(pack_map.get("ref"), label=f"{label}.ref")

    if "name" in pack_map:
        _ensure_non_empty_str(pack_map.get("name"), label=f"{label}.name")

    if "priority" in pack_map and not isinstance(pack_map.get("priority"), int):
        raise ConfigError(f"{label}.priority must be an integer")
    if "replace_options" in pack_map and not isinstance(
        pack_map.get("replace_options"), bool
    ):
        raise ConfigError(f"{label}.replace_options must be boolean")

    options_from = pack_map.get("options_from", [])
    if options_from is not None:
        if not isinstance(options_from, list):
            raise ConfigError(f"{label}.options_from must be a list")
        for index, source in enumerate(options_from):
            _validate_option_source(source, label=f"{label}.options_from[{index}]")

    options = pack_map.get("options", [])
    if options is not None:
        if not isinstance(options, list):
            raise ConfigError(f"{label}.options must be a list")
        for index, option in enumerate(options):
            _validate_option(option, label=f"{label}.options[{index}]")

    if "filter" in pack_map:
        filter_map = _ensure_mapping(pack_map.get("filter"), label=f"{label}.filter")
        unknown_filter = sorted(set(filter_map.keys()) - {"include_ids", "exclude_ids"})
        if unknown_filter:
            raise ConfigError(f"{label}.filter has unknown keys: {unknown_filter}")
        if "include_ids" in filter_map:
            _ensure_string_list(
                filter_map.get("include_ids"), label=f"{label}.filter.include_ids"
            )
        if "exclude_ids" in filter_map:
            _ensure_string_list(
                filter_map.get("exclude_ids"), label=f"{label}.filter.exclude_ids"
            )


def _validate_constraints(constraints: Any, *, label: str = "constraints") -> None:
    constraints_map = _ensure_mapping(constraints, label=label)
    unknown = sorted(set(constraints_map.keys()) - {"include", "exclude", "predicates"})
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {unknown}")

    for mode in ("include", "exclude"):
        rules = constraints_map.get(mode, [])
        if rules is None:
            continue
        if not isinstance(rules, list):
            raise ConfigError(f"{label}.{mode} must be a list")
        for rule_index, rule in enumerate(rules):
            rule_map = _ensure_mapping(rule, label=f"{label}.{mode}[{rule_index}]")
            unknown_rule = sorted(set(rule_map.keys()) - {"when"})
            if unknown_rule:
                raise ConfigError(
                    f"{label}.{mode}[{rule_index}] has unknown keys: {unknown_rule}"
                )
            when = _ensure_mapping(
                rule_map.get("when"), label=f"{label}.{mode}[{rule_index}].when"
            )
            for pack_name, ids in when.items():
                _ensure_non_empty_str(
                    pack_name, label=f"{label}.{mode}[{rule_index}].when pack"
                )
                if isinstance(ids, str):
                    _ensure_non_empty_str(
                        ids, label=f"{label}.{mode}[{rule_index}].when value"
                    )
                else:
                    _ensure_string_list(
                        ids, label=f"{label}.{mode}[{rule_index}].when values"
                    )

    predicates = constraints_map.get("predicates", [])
    if predicates is None:
        return
    if not isinstance(predicates, list):
        raise ConfigError(f"{label}.predicates must be a list")

    ids: set[str] = set()
    for pred_index, pred in enumerate(predicates):
        pred_map = _ensure_mapping(pred, label=f"{label}.predicates[{pred_index}]")
        unknown_pred = sorted(set(pred_map.keys()) - {"id", "args", "expr", "on_error"})
        if unknown_pred:
            raise ConfigError(
                f"{label}.predicates[{pred_index}] has unknown keys: {unknown_pred}"
            )

        pred_id = _ensure_non_empty_str(
            pred_map.get("id"), label=f"{label}.predicates[{pred_index}].id"
        )
        if pred_id in ids:
            raise ConfigError(f"Duplicate {label}.predicates id '{pred_id}'")
        ids.add(pred_id)

        args_map = _ensure_mapping(
            pred_map.get("args"), label=f"{label}.predicates[{pred_index}].args"
        )
        if not args_map:
            raise ConfigError(
                f"{label}.predicates[{pred_index}].args must not be empty"
            )

        arg_names: set[str] = set()
        for arg_name, arg_spec in args_map.items():
            arg_name_s = _ensure_non_empty_str(
                arg_name, label=f"{label}.predicates[{pred_index}] arg name"
            )
            if arg_name_s in arg_names:
                raise ConfigError(
                    f"{label}.predicates[{pred_index}] has duplicate arg '{arg_name_s}'"
                )
            arg_names.add(arg_name_s)
            arg_spec_map = _ensure_mapping(
                arg_spec, label=f"{label}.predicates[{pred_index}].args.{arg_name_s}"
            )
            unknown_arg = sorted(
                set(arg_spec_map.keys()) - {"pack_id", "param", "default"}
            )
            if unknown_arg:
                raise ConfigError(
                    f"{label}.predicates[{pred_index}].args.{arg_name_s} has unknown keys: {unknown_arg}"
                )
            has_pack_id = "pack_id" in arg_spec_map
            has_param = "param" in arg_spec_map
            if has_pack_id == has_param:
                raise ConfigError(
                    f"{label}.predicates[{pred_index}].args.{arg_name_s} must define exactly one of pack_id or param"
                )
            if has_pack_id:
                _ensure_non_empty_str(
                    arg_spec_map.get("pack_id"),
                    label=f"{label}.predicates[{pred_index}].args.{arg_name_s}.pack_id",
                )
            if has_param:
                path = _ensure_non_empty_str(
                    arg_spec_map.get("param"),
                    label=f"{label}.predicates[{pred_index}].args.{arg_name_s}.param",
                )
                if any(not part for part in path.split(".")):
                    raise ConfigError(
                        f"{label}.predicates[{pred_index}].args.{arg_name_s}.param has invalid dotted path '{path}'"
                    )

        if "expr" not in pred_map:
            raise ConfigError(f"{label}.predicates[{pred_index}].expr is required")
        expr = normalize_predicate_expr(
            pred_map.get("expr"), label=f"{label}.predicates[{pred_index}].expr"
        )

        unknown_refs = sorted(_collect_predicate_refs(expr) - arg_names)
        if unknown_refs:
            raise ConfigError(
                f"{label}.predicates[{pred_index}].expr references unknown args: {unknown_refs}"
            )

        on_error = pred_map.get("on_error", "error")
        if isinstance(on_error, bool):
            on_error = "false" if on_error is False else "error"
        if on_error not in {"error", "false"}:
            raise ConfigError(
                f"{label}.predicates[{pred_index}].on_error must be 'error' or 'false'"
            )


def _validate_registry(registry: Any, *, label: str = "registry") -> None:
    registry_map = _ensure_mapping(registry, label=label)
    unknown = sorted(set(registry_map.keys()) - {"option_sets", "packs", "groups"})
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {unknown}")

    option_sets = registry_map.get("option_sets", {})
    if option_sets is not None:
        option_sets_map = _ensure_mapping(option_sets, label=f"{label}.option_sets")
        for set_name, options in option_sets_map.items():
            _ensure_non_empty_str(set_name, label=f"{label}.option_sets key")
            if not isinstance(options, list):
                raise ConfigError(f"{label}.option_sets['{set_name}'] must be a list")
            for index, option in enumerate(options):
                _validate_option(
                    option, label=f"{label}.option_sets['{set_name}'][{index}]"
                )

    packs = registry_map.get("packs", {})
    if packs is not None:
        packs_map = _ensure_mapping(packs, label=f"{label}.packs")
        for pack_name, pack in packs_map.items():
            _ensure_non_empty_str(pack_name, label=f"{label}.packs key")
            _validate_pack_selector(pack, label=f"{label}.packs['{pack_name}']")

    groups = registry_map.get("groups", {})
    if groups is not None:
        groups_map = _ensure_mapping(groups, label=f"{label}.groups")
        for group_name, selectors in groups_map.items():
            _ensure_non_empty_str(group_name, label=f"{label}.groups key")
            if not isinstance(selectors, list):
                raise ConfigError(f"{label}.groups['{group_name}'] must be a list")
            for index, selector in enumerate(selectors):
                _validate_pack_selector(
                    selector, label=f"{label}.groups['{group_name}'][{index}]"
                )


def _validate_definition_sections(doc: Mapping[str, Any]) -> None:
    _validate_registry(
        {
            "option_sets": doc.get("option_sets", {}),
            "packs": doc.get("packs", {}),
            "groups": doc.get("groups", {}),
        },
        label="definitions",
    )


def _validate_strategy_map(value: Any, *, label: str) -> None:
    strategy_map = _ensure_mapping(value, label=label)
    for raw_path, raw_strategy in strategy_map.items():
        path = _ensure_non_empty_str(raw_path, label=f"{label} key")
        strategy = _ensure_non_empty_str(raw_strategy, label=f"{label}['{path}']")
        if strategy not in _ALLOWED_STRATEGIES:
            raise ConfigError(
                f"{label}['{path}'] must be one of {sorted(_ALLOWED_STRATEGIES)}"
            )


def validate_document(document: Mapping[str, Any]) -> dict[str, Any]:
    doc = deepcopy(dict(_ensure_mapping(document, label="experiment document")))

    if "registry" in doc:
        raise ConfigError(
            "registry is removed in schema v4; move definitions to top-level option_sets/packs/groups"
        )
    if "package" in doc:
        raise ConfigError(
            "package is removed in schema v4; move package scope to imports[].package"
        )

    unknown_top = sorted(set(doc.keys()) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown_top:
        raise ConfigError(
            f"experiment document has unknown top-level keys: {unknown_top}"
        )

    schema = _ensure_mapping(doc.get("schema", {}), label="schema")
    unknown_schema = sorted(set(schema.keys()) - {"version", "unknown_key_behavior"})
    if unknown_schema:
        raise ConfigError(f"schema has unknown keys: {unknown_schema}")
    version = schema.get("version", 4)
    if version != 4:
        raise ConfigError("schema.version must be 4")
    unknown_behavior = schema.get("unknown_key_behavior", "error")
    if unknown_behavior not in {"error", "warn"}:
        raise ConfigError("schema.unknown_key_behavior must be 'error' or 'warn'")

    if "imports" in doc:
        _validate_imports(doc.get("imports"), label="imports")

    if "command" in doc:
        command = _ensure_mapping(doc.get("command"), label="command")
        unknown_command = sorted(set(command.keys()) - {"program", "args"})
        if unknown_command:
            raise ConfigError(f"command has unknown keys: {unknown_command}")
        _ensure_non_empty_str(command.get("program"), label="command.program")
        _ensure_string_list(command.get("args", []), label="command.args")

    if "defaults" in doc:
        defaults = _ensure_mapping(doc.get("defaults"), label="defaults")
        unknown_defaults = sorted(set(defaults.keys()) - {"params", "tags"})
        if unknown_defaults:
            raise ConfigError(f"defaults has unknown keys: {unknown_defaults}")
        params = defaults.get("params", {})
        if params is not None:
            params_map = _ensure_mapping(params, label="defaults.params")
            assert_no_dotted_keys(params_map, label="defaults.params")
        _ensure_string_list(defaults.get("tags", []), label="defaults.tags")

    if "merge" in doc:
        merge = _ensure_mapping(doc.get("merge"), label="merge")
        unknown_merge = sorted(set(merge.keys()) - _ALLOWED_MERGE_KEYS)
        if unknown_merge:
            raise ConfigError(f"merge has unknown keys: {unknown_merge}")

        mode = merge.get("mode", "merge")
        if mode is not None:
            mode_s = _ensure_non_empty_str(mode, label="merge.mode")
            if mode_s not in _ALLOWED_MERGE_MODES:
                raise ConfigError(
                    f"merge.mode must be one of {sorted(_ALLOWED_MERGE_MODES)}"
                )

        if "strategies" in merge:
            _validate_strategy_map(merge.get("strategies"), label="merge.strategies")

        if "delete_sentinel" in merge:
            _ensure_non_empty_str(
                merge.get("delete_sentinel"), label="merge.delete_sentinel"
            )

    if "constraints" in doc:
        _validate_constraints(doc.get("constraints"), label="constraints")

    _validate_definition_sections(doc)

    if "select" in doc:
        select = _ensure_mapping(doc.get("select"), label="select")
        unknown_select = sorted(set(select.keys()) - {"groups", "packs"})
        if unknown_select:
            raise ConfigError(f"select has unknown keys: {unknown_select}")
        _ensure_string_list(select.get("groups", []), label="select.groups")
        packs = select.get("packs", [])
        if packs is not None:
            if not isinstance(packs, list):
                raise ConfigError("select.packs must be a list")
            for index, pack in enumerate(packs):
                _validate_pack_selector(pack, label=f"select.packs[{index}]")

    if "run_sets" in doc:
        run_sets = _ensure_mapping(doc.get("run_sets"), label="run_sets")
        for run_set_name, run_set in run_sets.items():
            name = _ensure_non_empty_str(run_set_name, label="run_sets key")
            run_set_map = _ensure_mapping(run_set, label=f"run_sets.{name}")
            unknown_run_set = sorted(
                set(run_set_map.keys())
                - {"extends", "append", "replace", "patch_packs"}
            )
            if unknown_run_set:
                raise ConfigError(
                    f"run_sets.{name} has unknown keys: {unknown_run_set}"
                )
            _ensure_string_list(
                run_set_map.get("extends", []), label=f"run_sets.{name}.extends"
            )

            for section in ("append", "replace"):
                payload = run_set_map.get(section, {})
                if payload is None:
                    continue
                payload_map = _ensure_mapping(
                    payload, label=f"run_sets.{name}.{section}"
                )
                unknown_roots = sorted(
                    set(payload_map.keys())
                    - {"select", "defaults", "constraints", "merge"}
                )
                if unknown_roots:
                    raise ConfigError(
                        f"run_sets.{name}.{section} has unknown roots: {unknown_roots}"
                    )

            patch_packs = run_set_map.get("patch_packs", {})
            if patch_packs is not None:
                patch_map = _ensure_mapping(
                    patch_packs, label=f"run_sets.{name}.patch_packs"
                )
                for pack_name, replacement in patch_map.items():
                    pack_name_s = _ensure_non_empty_str(
                        pack_name, label=f"run_sets.{name}.patch_packs key"
                    )
                    _validate_pack_selector(
                        replacement,
                        label=f"run_sets.{name}.patch_packs['{pack_name_s}']",
                    )

    return doc
