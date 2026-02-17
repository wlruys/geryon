from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from geryon.models import ConfigError


@dataclass
class _Registry:
    option_sets: dict[str, dict[str, Any]] = field(default_factory=dict)
    packs: dict[str, dict[str, Any]] = field(default_factory=dict)
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class _FileLoadKey:
    path: Path
    package: str | None


def _validate_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _normalize_package(
    package: Any,
    *,
    source_path: Path,
    label: str = "package",
) -> str | None:
    if package is None:
        return None
    if not isinstance(package, str):
        raise ConfigError(f"{source_path}: {label} must be a string")
    package = package.strip()
    return package or None


def _schema_unknown_key_behavior(raw: Mapping[str, Any], *, source_path: Path) -> str:
    schema_raw = raw.get("schema", {})
    if schema_raw is None:
        schema_raw = {}
    schema_map = _validate_mapping(schema_raw, label=f"{source_path}: schema")
    unknown_schema = sorted(
        set(str(key) for key in schema_map.keys()) - {"version", "unknown_key_behavior"}
    )
    if unknown_schema:
        raise ConfigError(f"{source_path}: schema has unknown keys: {unknown_schema}")

    version = schema_map.get("version", 4)
    if not isinstance(version, int) or version != 4:
        raise ConfigError(f"{source_path}: schema.version must be integer 4")

    behavior = schema_map.get("unknown_key_behavior", "error")
    if not isinstance(behavior, str) or behavior not in {"error", "warn"}:
        raise ConfigError(
            f"{source_path}: schema.unknown_key_behavior must be 'error' or 'warn'"
        )
    return behavior


def _handle_unknown_keys(
    unknown_keys: list[str],
    *,
    label: str,
    behavior: str,
    diagnostics: dict[str, Any],
) -> None:
    if not unknown_keys:
        return
    message = f"{label} has unknown keys: {unknown_keys}"
    if behavior == "error":
        raise ConfigError(message)
    diagnostics.setdefault("warnings", []).append(message)


def _qualify_definition_name(
    name: Any, *, package: str | None, source_path: Path
) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{source_path}: definition key must be a non-empty string")
    name = name.strip()
    if name.startswith("/"):
        return name[1:]
    if package:
        return f"{package}.{name}"
    return name


def _resolve_reference_name(
    ref: Any, *, package: str | None, source_path: Path, label: str
) -> str:
    if not isinstance(ref, str) or not ref.strip():
        raise ConfigError(f"{source_path}: {label} must be a non-empty string")
    ref = ref.strip()
    if ref.startswith("/"):
        return ref[1:]
    if package and "." not in ref:
        return f"{package}.{ref}"
    return ref


def _merge_mappings(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        key_s = str(key)
        if (
            key_s in merged
            and isinstance(merged[key_s], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key_s] = _merge_mappings(
                _validate_mapping(merged[key_s], label=f"merge[{key_s}]"),
                _validate_mapping(value, label=f"merge[{key_s}]"),
            )
        else:
            merged[key_s] = deepcopy(value)
    return merged


def _append_mappings(
    base: Mapping[str, Any], patch: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in patch.items():
        key_s = str(key)
        if key_s not in merged:
            merged[key_s] = deepcopy(value)
            continue
        existing = merged[key_s]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key_s] = _append_mappings(
                _validate_mapping(existing, label=f"append[{key_s}]"),
                _validate_mapping(value, label=f"append[{key_s}]"),
            )
            continue
        if isinstance(existing, list) and isinstance(value, list):
            merged[key_s] = [*deepcopy(existing), *deepcopy(value)]
            continue
        merged[key_s] = deepcopy(value)
    return merged


def _render_cycle_chain(chain: list[str]) -> str:
    lines = ["  cycle chain:"]
    for index in range(len(chain) - 1):
        lines.append(f"    {index + 1}. {chain[index]} -> {chain[index + 1]}")
    return "\n".join(lines)


def _coerce_str_list(value: Any, *, source_path: Path, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{source_path}: {label} must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f"{source_path}: {label}[{index}] must be a string")
        trimmed = item.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        out.append(trimmed)
    return out


def _register_definition(
    registry_map: dict[str, dict[str, Any]],
    *,
    key: str,
    value: Any,
    package: str | None,
    source_path: Path,
    kind: str,
) -> None:
    if key in registry_map:
        previous = registry_map[key]["source_path"]
        raise ConfigError(
            f"Duplicate {kind} definition '{key}' in {source_path}; first defined in {previous}"
        )
    registry_map[key] = {
        "value": deepcopy(value),
        "package": package,
        "source_path": str(source_path),
    }


def _validate_file_top_level(
    raw: Mapping[str, Any],
    *,
    source_path: Path,
    behavior: str,
    diagnostics: dict[str, Any],
) -> None:
    if "registry" in raw:
        raise ConfigError(
            f"{source_path}: 'registry' is removed in schema v4. "
            "Move definitions to top-level keys: option_sets, packs, groups."
        )
    if "package" in raw:
        raise ConfigError(
            f"{source_path}: 'package' is removed in schema v4. "
            "Define package scope at each imports entry via imports[].package."
        )

    allowed = {
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
    unknown = sorted(set(str(key) for key in raw.keys()) - allowed)
    _handle_unknown_keys(
        unknown, label=f"{source_path}", behavior=behavior, diagnostics=diagnostics
    )


def _normalize_import_entries(
    raw: Any, *, source_path: Path
) -> list[dict[str, str | None]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError(f"{source_path}: imports must be a list")

    out: list[dict[str, str | None]] = []
    for index, entry_raw in enumerate(raw):
        if not isinstance(entry_raw, Mapping):
            raise ConfigError(
                f"{source_path}: imports[{index}] must be a mapping with keys 'path' and optional 'package'"
            )
        entry = _validate_mapping(entry_raw, label=f"{source_path}: imports[{index}]")
        unknown_keys = sorted(
            set(str(key) for key in entry.keys()) - {"path", "package"}
        )
        if unknown_keys:
            raise ConfigError(
                f"{source_path}: imports[{index}] has unknown keys: {unknown_keys}"
            )

        path_raw = entry.get("path")
        if not isinstance(path_raw, str) or not path_raw.strip():
            raise ConfigError(
                f"{source_path}: imports[{index}].path must be a non-empty string"
            )

        out.append(
            {
                "path": path_raw.strip(),
                "package": _normalize_package(
                    entry.get("package"),
                    source_path=source_path,
                    label=f"imports[{index}].package",
                ),
            }
        )
    return out


def _load_file_into_registry(
    *,
    source_path: Path,
    package: str | None,
    registry: _Registry,
    diagnostics: dict[str, Any],
    loaded: set[_FileLoadKey],
    stack: list[_FileLoadKey],
) -> dict[str, Any]:
    path = source_path.resolve()
    load_key = _FileLoadKey(path=path, package=package)
    if load_key in loaded:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigError(f"Failed re-reading YAML '{path}': {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{path}: YAML root must be a mapping")
        return dict(raw)

    if load_key in stack:
        chain = [
            f"{item.path} (package={item.package if item.package is not None else '<none>'})"
            for item in [*stack, load_key]
        ]
        raise ConfigError(
            "Import cycle detected.\n"
            f"{_render_cycle_chain(chain)}\n"
            "  fix: remove one import edge so the chain is acyclic."
        )

    if not path.exists():
        if stack:
            parent = stack[-1]
            context = (
                f" (imported from {parent.path}"
                f" with package={parent.package if parent.package is not None else '<none>'})"
            )
        else:
            context = ""
        raise ConfigError(f"File not found: {path}{context}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"Failed parsing YAML '{path}': {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: YAML root must be a mapping")

    behavior = _schema_unknown_key_behavior(raw, source_path=path)
    _validate_file_top_level(
        raw, source_path=path, behavior=behavior, diagnostics=diagnostics
    )

    imports = _normalize_import_entries(raw.get("imports", []), source_path=path)

    imported_paths: list[str] = []
    imports_detail: list[dict[str, str | None]] = []
    stack.append(load_key)
    for item in imports:
        child = Path(str(item["path"]))
        child_path = (
            (path.parent / child).resolve()
            if not child.is_absolute()
            else child.resolve()
        )
        imported_paths.append(str(child_path))
        imports_detail.append({"path": str(child_path), "package": item["package"]})
        _load_file_into_registry(
            source_path=child_path,
            package=item["package"],
            registry=registry,
            diagnostics=diagnostics,
            loaded=loaded,
            stack=stack,
        )
    stack.pop()

    option_sets_raw = raw.get("option_sets", {})
    if option_sets_raw is None:
        option_sets_raw = {}
    option_sets = _validate_mapping(option_sets_raw, label=f"{path}: option_sets")

    packs_raw = raw.get("packs", {})
    if packs_raw is None:
        packs_raw = {}
    packs = _validate_mapping(packs_raw, label=f"{path}: packs")

    groups_raw = raw.get("groups", {})
    if groups_raw is None:
        groups_raw = {}
    groups = _validate_mapping(groups_raw, label=f"{path}: groups")

    registered_option_sets: list[str] = []
    for name, options in option_sets.items():
        key = _qualify_definition_name(name, package=package, source_path=path)
        if not isinstance(options, list):
            raise ConfigError(f"{path}: option_sets['{name}'] must be a list")
        for index, option in enumerate(options):
            if not isinstance(option, Mapping):
                raise ConfigError(
                    f"{path}: option_sets['{name}'][{index}] must be a mapping"
                )
        _register_definition(
            registry.option_sets,
            key=key,
            value=list(options),
            package=package,
            source_path=path,
            kind="option_set",
        )
        registered_option_sets.append(key)

    registered_packs: list[str] = []
    for name, pack_spec in packs.items():
        key = _qualify_definition_name(name, package=package, source_path=path)
        if not isinstance(pack_spec, Mapping):
            raise ConfigError(f"{path}: packs['{name}'] must be a mapping")
        _register_definition(
            registry.packs,
            key=key,
            value=dict(pack_spec),
            package=package,
            source_path=path,
            kind="pack",
        )
        registered_packs.append(key)

    registered_groups: list[str] = []
    for name, group_entries in groups.items():
        key = _qualify_definition_name(name, package=package, source_path=path)
        if not isinstance(group_entries, list):
            raise ConfigError(f"{path}: groups['{name}'] must be a list")
        for index, selector in enumerate(group_entries):
            if not isinstance(selector, Mapping):
                raise ConfigError(
                    f"{path}: groups['{name}'][{index}] must be a mapping"
                )
        _register_definition(
            registry.groups,
            key=key,
            value=list(group_entries),
            package=package,
            source_path=path,
            kind="group",
        )
        registered_groups.append(key)

    diagnostics["files"].append(
        {
            "path": str(path),
            "package": package,
            "imports": imported_paths,
            "imports_detail": imports_detail,
            "registered": {
                "option_sets": sorted(registered_option_sets),
                "packs": sorted(registered_packs),
                "groups": sorted(registered_groups),
            },
        }
    )

    loaded.add(load_key)
    return dict(raw)


def _normalized_run_sets(
    root_raw: Mapping[str, Any], *, source_path: Path
) -> dict[str, dict[str, Any]]:
    run_sets_raw = root_raw.get("run_sets", {})
    if run_sets_raw is None:
        return {}
    if not isinstance(run_sets_raw, Mapping):
        raise ConfigError(f"{source_path}: run_sets must be a mapping")

    normalized: dict[str, dict[str, Any]] = {}
    for name, value in run_sets_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{source_path}: run_set names must be non-empty strings")
        if not isinstance(value, Mapping):
            raise ConfigError(f"{source_path}: run_sets.{name} must be a mapping")
        normalized[name.strip()] = dict(value)
    return normalized


def _resolve_run_set_order(
    *,
    run_set_name: str,
    run_sets: Mapping[str, dict[str, Any]],
    source_path: Path,
    stack: list[str],
    cache: dict[str, list[str]],
) -> list[str]:
    if run_set_name in cache:
        return list(cache[run_set_name])

    if run_set_name not in run_sets:
        known = ", ".join(sorted(run_sets.keys())) or "<none>"
        raise ConfigError(
            f"{source_path}: unknown run_set '{run_set_name}'. Known run_sets: {known}"
        )

    if run_set_name in stack:
        chain = [*stack, run_set_name]
        raise ConfigError(
            f"{source_path}: run_set inheritance cycle detected.\n"
            f"{_render_cycle_chain(chain)}\n"
            "  fix: remove one extends edge from the cycle."
        )

    run_set = run_sets[run_set_name]
    unknown_keys = sorted(
        set(str(key) for key in run_set.keys())
        - {"extends", "append", "replace", "patch_packs"}
    )
    if unknown_keys:
        raise ConfigError(
            f"{source_path}: run_sets.{run_set_name} has unknown keys: {unknown_keys}"
        )

    extends_raw = run_set.get("extends", [])
    if extends_raw is None:
        extends_raw = []
    if not isinstance(extends_raw, list):
        raise ConfigError(
            f"{source_path}: run_sets.{run_set_name}.extends must be a list"
        )

    order: list[str] = []
    seen: set[str] = set()
    next_stack = [*stack, run_set_name]
    for index, parent in enumerate(extends_raw):
        if not isinstance(parent, str) or not parent.strip():
            raise ConfigError(
                f"{source_path}: run_sets.{run_set_name}.extends[{index}] must be a non-empty string"
            )
        parent_name = parent.strip()
        for resolved in _resolve_run_set_order(
            run_set_name=parent_name,
            run_sets=run_sets,
            source_path=source_path,
            stack=next_stack,
            cache=cache,
        ):
            if resolved not in seen:
                seen.add(resolved)
                order.append(resolved)

    if run_set_name not in seen:
        order.append(run_set_name)

    cache[run_set_name] = list(order)
    return order


def _validate_run_set_operation_payload(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    run_set_name: str,
    section: str,
) -> None:
    allowed_roots = {"select", "defaults", "constraints", "merge"}
    unknown_roots = sorted(set(str(key) for key in payload.keys()) - allowed_roots)
    if unknown_roots:
        raise ConfigError(
            f"{source_path}: run_sets.{run_set_name}.{section} has unknown roots: {unknown_roots}. "
            f"Allowed: {sorted(allowed_roots)}"
        )


def _apply_run_set(
    *,
    root_raw: Mapping[str, Any],
    run_set_name: str,
    source_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_sets = _normalized_run_sets(root_raw, source_path=source_path)
    cache: dict[str, list[str]] = {}
    order = _resolve_run_set_order(
        run_set_name=run_set_name,
        run_sets=run_sets,
        source_path=source_path,
        stack=[],
        cache=cache,
    )

    merged = deepcopy(dict(root_raw))
    applied_ops: list[dict[str, Any]] = []
    for current in order:
        run_set = run_sets[current]

        append_raw = run_set.get("append", {})
        if append_raw is None:
            append_raw = {}
        append_map = _validate_mapping(
            append_raw, label=f"{source_path}: run_sets.{current}.append"
        )
        _validate_run_set_operation_payload(
            append_map,
            source_path=source_path,
            run_set_name=current,
            section="append",
        )
        if append_map:
            merged = _append_mappings(merged, append_map)
            applied_ops.append(
                {
                    "run_set": current,
                    "operation": "append",
                    "payload": deepcopy(dict(append_map)),
                }
            )

        replace_raw = run_set.get("replace", {})
        if replace_raw is None:
            replace_raw = {}
        replace_map = _validate_mapping(
            replace_raw, label=f"{source_path}: run_sets.{current}.replace"
        )
        _validate_run_set_operation_payload(
            replace_map,
            source_path=source_path,
            run_set_name=current,
            section="replace",
        )
        if replace_map:
            merged = _merge_mappings(merged, replace_map)
            applied_ops.append(
                {
                    "run_set": current,
                    "operation": "replace",
                    "payload": deepcopy(dict(replace_map)),
                }
            )

        patch_raw = run_set.get("patch_packs", {})
        if patch_raw is None:
            patch_raw = {}
        patch_map = _validate_mapping(
            patch_raw, label=f"{source_path}: run_sets.{current}.patch_packs"
        )
        if patch_map:
            select_map = dict(
                _validate_mapping(
                    merged.get("select"),
                    label=f"{source_path}: run_sets.{current}.patch_packs requires select",
                )
            )
            packs_raw = select_map.get("packs")
            if not isinstance(packs_raw, list):
                raise ConfigError(
                    f"{source_path}: run_sets.{current}.patch_packs requires select.packs to be a list"
                )
            patched_packs = deepcopy(list(packs_raw))
            for raw_pack_name, raw_replacement in patch_map.items():
                if not isinstance(raw_pack_name, str) or not raw_pack_name.strip():
                    raise ConfigError(
                        f"{source_path}: run_sets.{current}.patch_packs keys must be non-empty strings"
                    )
                pack_name = raw_pack_name.strip()
                replacement = _validate_mapping(
                    raw_replacement,
                    label=f"{source_path}: run_sets.{current}.patch_packs['{pack_name}']",
                )

                replace_index: int | None = None
                for index, candidate_raw in enumerate(patched_packs):
                    candidate = _validate_mapping(
                        candidate_raw,
                        label=(
                            f"{source_path}: run_sets.{current}.patch_packs "
                            f"target select.packs[{index}]"
                        ),
                    )
                    candidate_name = candidate.get("name")
                    if (
                        isinstance(candidate_name, str)
                        and candidate_name.strip() == pack_name
                    ):
                        replace_index = index
                        break

                if replace_index is None:
                    raise ConfigError(
                        f"{source_path}: run_sets.{current}.patch_packs['{pack_name}'] "
                        "does not match any pack name in select.packs"
                    )

                replacement_doc = deepcopy(dict(replacement))
                replacement_name = replacement_doc.get("name")
                if replacement_name is None:
                    replacement_doc["name"] = pack_name
                elif (
                    not isinstance(replacement_name, str)
                    or not replacement_name.strip()
                ):
                    raise ConfigError(
                        f"{source_path}: run_sets.{current}.patch_packs['{pack_name}'].name "
                        "must be a non-empty string when provided"
                    )
                elif replacement_name.strip() != pack_name:
                    raise ConfigError(
                        f"{source_path}: run_sets.{current}.patch_packs['{pack_name}'].name "
                        f"must match '{pack_name}' when provided"
                    )
                else:
                    replacement_doc["name"] = replacement_name.strip()

                patched_packs[replace_index] = replacement_doc

            select_map["packs"] = patched_packs
            merged["select"] = select_map
            applied_ops.append(
                {
                    "run_set": current,
                    "operation": "patch_packs",
                    "payload": deepcopy(dict(patch_map)),
                }
            )

    merged.pop("run_sets", None)
    run_set_diag = {
        "selected": run_set_name,
        "available": sorted(run_sets.keys()),
        "resolved_order": order,
        "applied_operations": applied_ops,
    }
    return merged, run_set_diag


def list_run_set_names(experiment_path: str | Path) -> list[str]:
    root_path = Path(experiment_path).resolve()
    diagnostics: dict[str, Any] = {"files": [], "warnings": []}
    registry = _Registry()
    loaded: set[_FileLoadKey] = set()
    root_raw = _load_file_into_registry(
        source_path=root_path,
        package=None,
        registry=registry,
        diagnostics=diagnostics,
        loaded=loaded,
        stack=[],
    )
    return sorted(_normalized_run_sets(root_raw, source_path=root_path).keys())


def _normalize_option_filter(
    value: Any,
    *,
    source_path: Path,
    label: str,
) -> dict[str, set[str]]:
    if value is None:
        return {"include_ids": set(), "exclude_ids": set()}
    mapping = _validate_mapping(value, label=f"{source_path}: {label}")
    unknown = sorted(
        set(str(key) for key in mapping.keys()) - {"include_ids", "exclude_ids"}
    )
    if unknown:
        raise ConfigError(f"{source_path}: {label} has unknown keys: {unknown}")

    include_ids = {
        item
        for item in _coerce_str_list(
            mapping.get("include_ids"),
            source_path=source_path,
            label=f"{label}.include_ids",
        )
    }
    exclude_ids = {
        item
        for item in _coerce_str_list(
            mapping.get("exclude_ids"),
            source_path=source_path,
            label=f"{label}.exclude_ids",
        )
    }
    return {"include_ids": include_ids, "exclude_ids": exclude_ids}


def _apply_option_filter(
    options: list[dict[str, Any]],
    *,
    include_ids: set[str],
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for option in options:
        option_id = option.get("id")
        option_id_s = str(option_id) if option_id is not None else None
        if include_ids and (option_id_s is None or option_id_s not in include_ids):
            continue
        if option_id_s is not None and option_id_s in exclude_ids:
            continue
        filtered.append(option)
    return filtered


def _resolve_option_source(
    source: Any,
    *,
    package: str | None,
    source_path: Path,
    registry: _Registry,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    source_map = _validate_mapping(source, label=f"{source_path}: options_from entry")
    unknown = sorted(
        set(str(key) for key in source_map.keys())
        - {"ref", "include_ids", "exclude_ids"}
    )
    if unknown:
        raise ConfigError(
            f"{source_path}: options_from entry has unknown keys: {unknown}"
        )
    if "ref" not in source_map:
        raise ConfigError(f"{source_path}: options_from entry must include 'ref'")

    resolved_ref = _resolve_reference_name(
        source_map["ref"],
        package=package,
        source_path=source_path,
        label="options_from ref",
    )
    if resolved_ref not in registry.option_sets:
        known = sorted(registry.option_sets.keys())
        preview = ", ".join(known[:8])
        suffix = "..." if len(known) > 8 else ""
        raise ConfigError(
            f"{source_path}: unknown option_set ref '{resolved_ref}'. Known option_sets: {preview}{suffix}"
        )

    include_ids = {
        item
        for item in _coerce_str_list(
            source_map.get("include_ids"), source_path=source_path, label="include_ids"
        )
    }
    exclude_ids = {
        item
        for item in _coerce_str_list(
            source_map.get("exclude_ids"), source_path=source_path, label="exclude_ids"
        )
    }

    option_set_record = registry.option_sets[resolved_ref]
    options = deepcopy(list(option_set_record["value"]))
    filtered = _apply_option_filter(
        options, include_ids=include_ids, exclude_ids=exclude_ids
    )

    diagnostics["expansion"]["option_sources"].append(
        {
            "ref": resolved_ref,
            "requested_from": str(source_path),
            "before": len(options),
            "after": len(filtered),
            "filter": {
                "include_ids": sorted(include_ids),
                "exclude_ids": sorted(exclude_ids),
            },
        }
    )
    return filtered


def _normalize_pack_spec(
    pack_spec: Mapping[str, Any],
    *,
    package: str | None,
    source_path: Path,
    registry: _Registry,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    unknown = sorted(
        set(str(key) for key in pack_spec.keys())
        - {"name", "options_from", "options", "priority"}
    )
    if unknown:
        raise ConfigError(f"{source_path}: pack spec has unknown keys: {unknown}")

    name = pack_spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{source_path}: pack spec must include non-empty 'name'")

    options: list[dict[str, Any]] = []
    options_from = pack_spec.get("options_from", [])
    if options_from is None:
        options_from = []
    if not isinstance(options_from, list):
        raise ConfigError(f"{source_path}: options_from must be a list")
    for source in options_from:
        options.extend(
            _resolve_option_source(
                source,
                package=package,
                source_path=source_path,
                registry=registry,
                diagnostics=diagnostics,
            )
        )

    inline_options = pack_spec.get("options", [])
    if inline_options is None:
        inline_options = []
    if not isinstance(inline_options, list):
        raise ConfigError(f"{source_path}: options must be a list")
    for index, option in enumerate(inline_options):
        if not isinstance(option, Mapping):
            raise ConfigError(f"{source_path}: options[{index}] must be a mapping")
        options.append(deepcopy(dict(option)))

    if not options:
        raise ConfigError(f"{source_path}: pack '{name}' resolved to zero options")

    priority_raw = pack_spec.get("priority", 0)
    if not isinstance(priority_raw, int):
        raise ConfigError(f"{source_path}: priority must be an integer")

    expanded: dict[str, Any] = {"name": name.strip(), "options": options}
    if priority_raw != 0:
        expanded["priority"] = priority_raw
    return expanded


def _expand_pack_reference(
    *,
    selector: Mapping[str, Any],
    package: str | None,
    source_path: Path,
    registry: _Registry,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    unknown = sorted(
        set(str(key) for key in selector.keys())
        - {
            "ref",
            "name",
            "priority",
            "replace_options",
            "options_from",
            "options",
            "filter",
        }
    )
    if unknown:
        raise ConfigError(f"{source_path}: pack selector has unknown keys: {unknown}")

    ref = selector.get("ref")
    resolved_ref = _resolve_reference_name(
        ref, package=package, source_path=source_path, label="pack ref"
    )
    if resolved_ref not in registry.packs:
        known = sorted(registry.packs.keys())
        preview = ", ".join(known[:8])
        suffix = "..." if len(known) > 8 else ""
        raise ConfigError(
            f"{source_path}: unknown pack ref '{resolved_ref}'. Known packs: {preview}{suffix}"
        )

    pack_record = registry.packs[resolved_ref]
    base_pack = _normalize_pack_spec(
        _validate_mapping(
            pack_record["value"], label=f"{source_path}: registry pack value"
        ),
        package=pack_record.get("package"),
        source_path=Path(pack_record["source_path"]),
        registry=registry,
        diagnostics=diagnostics,
    )

    diagnostics["expansion"]["pack_refs"].append(
        {"ref": resolved_ref, "requested_from": str(source_path)}
    )

    merged = deepcopy(base_pack)

    if "name" in selector:
        alias = selector.get("name")
        if not isinstance(alias, str) or not alias.strip():
            raise ConfigError(
                f"{source_path}: selector.name must be a non-empty string"
            )
        merged["name"] = alias.strip()

    replace_options = selector.get("replace_options", False)
    if not isinstance(replace_options, bool):
        raise ConfigError(f"{source_path}: selector.replace_options must be a boolean")
    if replace_options:
        merged["options"] = []

    if "priority" in selector:
        priority_raw = selector.get("priority")
        if not isinstance(priority_raw, int):
            raise ConfigError(f"{source_path}: selector.priority must be an integer")
        merged["priority"] = priority_raw

    source_entries = selector.get("options_from", [])
    if source_entries is None:
        source_entries = []
    if not isinstance(source_entries, list):
        raise ConfigError(f"{source_path}: selector.options_from must be a list")
    for source in source_entries:
        merged["options"].extend(
            _resolve_option_source(
                source,
                package=package,
                source_path=source_path,
                registry=registry,
                diagnostics=diagnostics,
            )
        )

    inline_options = selector.get("options", [])
    if inline_options is None:
        inline_options = []
    if not isinstance(inline_options, list):
        raise ConfigError(f"{source_path}: selector.options must be a list")
    for index, option in enumerate(inline_options):
        if not isinstance(option, Mapping):
            raise ConfigError(
                f"{source_path}: selector.options[{index}] must be a mapping"
            )
        merged["options"].append(deepcopy(dict(option)))

    option_filter = _normalize_option_filter(
        selector.get("filter"), source_path=source_path, label="selector.filter"
    )
    merged["options"] = _apply_option_filter(
        merged["options"],
        include_ids=option_filter["include_ids"],
        exclude_ids=option_filter["exclude_ids"],
    )
    if not merged["options"]:
        raise ConfigError(
            f"{source_path}: selector for '{resolved_ref}' removed all options"
        )

    return merged


def _expand_group(
    *,
    ref: Any,
    package: str | None,
    source_path: Path,
    registry: _Registry,
    diagnostics: dict[str, Any],
    stack: list[str],
) -> list[dict[str, Any]]:
    resolved_ref = _resolve_reference_name(
        ref, package=package, source_path=source_path, label="group ref"
    )
    if resolved_ref not in registry.groups:
        known = sorted(registry.groups.keys())
        preview = ", ".join(known[:8])
        suffix = "..." if len(known) > 8 else ""
        raise ConfigError(
            f"{source_path}: unknown group ref '{resolved_ref}'. Known groups: {preview}{suffix}"
        )

    if resolved_ref in stack:
        chain = [*stack, resolved_ref]
        raise ConfigError(
            "Group include cycle detected.\n"
            f"{_render_cycle_chain(chain)}\n"
            "  fix: remove one nested group reference to break the cycle."
        )

    diagnostics["expansion"]["group_refs"].append(
        {"ref": resolved_ref, "requested_from": str(source_path)}
    )

    group_record = registry.groups[resolved_ref]
    group_entries = list(group_record["value"])
    group_package = group_record.get("package")
    group_source = Path(group_record["source_path"])

    expanded: list[dict[str, Any]] = []
    for entry in group_entries:
        expanded.extend(
            _expand_pack_selector(
                _validate_mapping(entry, label=f"{group_source}: group selector"),
                package=group_package,
                source_path=group_source,
                registry=registry,
                diagnostics=diagnostics,
                group_stack=[*stack, resolved_ref],
            )
        )
    return expanded


def _expand_pack_selector(
    selector: Mapping[str, Any],
    *,
    package: str | None,
    source_path: Path,
    registry: _Registry,
    diagnostics: dict[str, Any],
    group_stack: list[str],
) -> list[dict[str, Any]]:
    if "group" in selector:
        unknown = sorted(set(str(key) for key in selector.keys()) - {"group"})
        if unknown:
            raise ConfigError(
                f"{source_path}: group selector has unknown keys: {unknown}"
            )
        return _expand_group(
            ref=selector["group"],
            package=package,
            source_path=source_path,
            registry=registry,
            diagnostics=diagnostics,
            stack=group_stack,
        )

    if "ref" in selector:
        return [
            _expand_pack_reference(
                selector=selector,
                package=package,
                source_path=source_path,
                registry=registry,
                diagnostics=diagnostics,
            )
        ]

    diagnostics["expansion"]["inline_packs"] += 1
    return [
        _normalize_pack_spec(
            selector,
            package=package,
            source_path=source_path,
            registry=registry,
            diagnostics=diagnostics,
        )
    ]


def compose_experiment_data(
    experiment_path: str | Path,
    *,
    run_set: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_path = Path(experiment_path).resolve()

    diagnostics: dict[str, Any] = {
        "files": [],
        "warnings": [],
        "registry": {},
        "expansion": {
            "group_refs": [],
            "pack_refs": [],
            "option_sources": [],
            "inline_packs": 0,
            "final_pack_count": 0,
        },
    }
    registry = _Registry()
    loaded: set[_FileLoadKey] = set()

    root_raw = _load_file_into_registry(
        source_path=root_path,
        package=None,
        registry=registry,
        diagnostics=diagnostics,
        loaded=loaded,
        stack=[],
    )

    run_set_diag: dict[str, Any] = {
        "selected": None,
        "available": list_run_set_names(root_path),
        "resolved_order": [],
        "applied_operations": [],
    }
    if run_set is not None:
        trimmed = run_set.strip()
        if not trimmed:
            raise ConfigError(f"{root_path}: --run-set cannot be empty")
        root_raw, run_set_diag = _apply_run_set(
            root_raw=root_raw,
            run_set_name=trimmed,
            source_path=root_path,
        )

    behavior = _schema_unknown_key_behavior(root_raw, source_path=root_path)
    _validate_file_top_level(
        root_raw, source_path=root_path, behavior=behavior, diagnostics=diagnostics
    )

    select_raw = root_raw.get("select", {})
    if select_raw is None:
        select_raw = {}
    select_map = _validate_mapping(select_raw, label=f"{root_path}: select")
    unknown_select = sorted(
        set(str(key) for key in select_map.keys()) - {"groups", "packs"}
    )
    _handle_unknown_keys(
        unknown_select,
        label=f"{root_path}: select",
        behavior=behavior,
        diagnostics=diagnostics,
    )

    selected_groups = select_map.get("groups", [])
    if selected_groups is None:
        selected_groups = []
    if not isinstance(selected_groups, list):
        raise ConfigError(f"{root_path}: select.groups must be a list")

    selected_pack_selectors = select_map.get("packs", [])
    if selected_pack_selectors is None:
        selected_pack_selectors = []
    if not isinstance(selected_pack_selectors, list):
        raise ConfigError(f"{root_path}: select.packs must be a list")

    selector_entries: list[Mapping[str, Any]] = []
    for index, group_ref in enumerate(selected_groups):
        if not isinstance(group_ref, str) or not group_ref.strip():
            raise ConfigError(
                f"{root_path}: select.groups[{index}] must be a non-empty string"
            )
        selector_entries.append({"group": group_ref.strip()})

    for index, selector in enumerate(selected_pack_selectors):
        if not isinstance(selector, Mapping):
            raise ConfigError(f"{root_path}: select.packs[{index}] must be a mapping")
        selector_entries.append(dict(selector))

    expanded_packs: list[dict[str, Any]] = []
    for selector in selector_entries:
        expanded_packs.extend(
            _expand_pack_selector(
                selector,
                package=None,
                source_path=root_path,
                registry=registry,
                diagnostics=diagnostics,
                group_stack=[],
            )
        )

    diagnostics["expansion"]["final_pack_count"] = len(expanded_packs)
    diagnostics["registry"] = {
        "option_sets": sorted(registry.option_sets.keys()),
        "packs": sorted(registry.packs.keys()),
        "groups": sorted(registry.groups.keys()),
    }
    diagnostics["run_set"] = run_set_diag

    composed = dict(root_raw)
    schema_map = dict(
        _validate_mapping(composed.get("schema", {}), label=f"{root_path}: schema")
    )
    if "version" not in schema_map:
        schema_map["version"] = 4
    if "unknown_key_behavior" not in schema_map:
        schema_map["unknown_key_behavior"] = "error"
    composed["schema"] = schema_map
    composed["packs"] = expanded_packs
    composed.pop("imports", None)
    composed.pop("option_sets", None)
    composed.pop("groups", None)
    composed.pop("run_sets", None)
    composed.pop("select", None)

    return composed, diagnostics
