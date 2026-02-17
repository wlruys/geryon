from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from geryon.models import ConfigError


def _ensure_non_empty_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_str_list(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{label} must be a list of strings")
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f"{label}[{index}] must be a string")
        item_s = item.strip()
        if not item_s or item_s in seen:
            continue
        seen.add(item_s)
        out.append(item_s)
    return tuple(out)


def assert_no_dotted_keys(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            if "." in key:
                raise ConfigError(
                    f"{label} key '{key}' is invalid: dotted keys are not allowed"
                )
            assert_no_dotted_keys(nested, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_dotted_keys(item, label=f"{label}[{index}]")


def _deep_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return deepcopy(dict(value))


@dataclass(frozen=True)
class OptionSource:
    ref: str
    include_ids: tuple[str, ...] = ()
    exclude_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ref", _ensure_non_empty_str(self.ref, label="OptionSource.ref")
        )
        object.__setattr__(
            self,
            "include_ids",
            _normalize_str_list(self.include_ids, label="OptionSource.include_ids"),
        )
        object.__setattr__(
            self,
            "exclude_ids",
            _normalize_str_list(self.exclude_ids, label="OptionSource.exclude_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ref": self.ref}
        if self.include_ids:
            out["include_ids"] = list(self.include_ids)
        if self.exclude_ids:
            out["exclude_ids"] = list(self.exclude_ids)
        return out


@dataclass(frozen=True, init=False)
class Option:
    option_id: str
    params: dict[str, Any]
    tag: str | None = None

    def __init__(
        self,
        id: str | None = None,
        params: Mapping[str, Any] | None = None,
        tag: str | None = None,
        *,
        option_id: str | None = None,
    ) -> None:
        if id is not None and option_id is not None:
            raise ConfigError("Option accepts exactly one of 'id' or 'option_id'")
        resolved_id = id if id is not None else option_id
        if resolved_id is None:
            raise ConfigError("Option requires 'id'")
        if params is None:
            raise ConfigError("Option.params is required")
        object.__setattr__(self, "option_id", resolved_id)
        object.__setattr__(self, "params", dict(params))
        object.__setattr__(self, "tag", tag)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "option_id", _ensure_non_empty_str(self.option_id, label="Option.id")
        )
        params = _deep_mapping(self.params, label=f"Option('{self.option_id}').params")
        assert_no_dotted_keys(params, label=f"Option('{self.option_id}').params")
        object.__setattr__(self, "params", params)

        if self.tag is not None:
            object.__setattr__(
                self, "tag", _ensure_non_empty_str(self.tag, label="Option.tag")
            )

    @property
    def id(self) -> str:
        return self.option_id

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.option_id, "params": deepcopy(self.params)}
        if self.tag:
            out["tag"] = self.tag
        return out


@dataclass(frozen=True)
class Pack:
    name: str
    options: tuple[Option, ...] = ()
    options_from: tuple[OptionSource, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _ensure_non_empty_str(self.name, label="Pack.name")
        )
        if not isinstance(self.priority, int):
            raise ConfigError("Pack.priority must be an integer")
        if not isinstance(self.options, tuple):
            object.__setattr__(self, "options", tuple(self.options))
        if not isinstance(self.options_from, tuple):
            object.__setattr__(self, "options_from", tuple(self.options_from))

        for idx, option in enumerate(self.options):
            if not isinstance(option, Option):
                raise ConfigError(
                    f"Pack('{self.name}').options[{idx}] must be an Option"
                )
        for idx, source in enumerate(self.options_from):
            if not isinstance(source, OptionSource):
                raise ConfigError(
                    f"Pack('{self.name}').options_from[{idx}] must be an OptionSource"
                )

        if not self.options and not self.options_from:
            raise ConfigError(
                f"Pack('{self.name}') must define at least one of options or options_from"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.options_from:
            out["options_from"] = [source.to_dict() for source in self.options_from]
        if self.options:
            out["options"] = [option.to_dict() for option in self.options]
        if self.priority != 0:
            out["priority"] = self.priority
        return out


@dataclass(frozen=True)
class PackSelector:
    ref: str
    name: str | None = None
    priority: int | None = None
    replace_options: bool | None = None
    options_from: tuple[OptionSource, ...] = ()
    options: tuple[Option, ...] = ()
    include_ids: tuple[str, ...] = ()
    exclude_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ref", _ensure_non_empty_str(self.ref, label="PackSelector.ref")
        )
        if self.name is not None:
            object.__setattr__(
                self,
                "name",
                _ensure_non_empty_str(self.name, label="PackSelector.name"),
            )
        if self.priority is not None and not isinstance(self.priority, int):
            raise ConfigError("PackSelector.priority must be an integer when provided")
        if self.replace_options is not None and not isinstance(
            self.replace_options, bool
        ):
            raise ConfigError(
                "PackSelector.replace_options must be a boolean when provided"
            )

        if not isinstance(self.options_from, tuple):
            object.__setattr__(self, "options_from", tuple(self.options_from))
        if not isinstance(self.options, tuple):
            object.__setattr__(self, "options", tuple(self.options))

        for idx, source in enumerate(self.options_from):
            if not isinstance(source, OptionSource):
                raise ConfigError(
                    f"PackSelector('{self.ref}').options_from[{idx}] must be OptionSource"
                )
        for idx, option in enumerate(self.options):
            if not isinstance(option, Option):
                raise ConfigError(
                    f"PackSelector('{self.ref}').options[{idx}] must be Option"
                )

        object.__setattr__(
            self,
            "include_ids",
            _normalize_str_list(
                self.include_ids, label=f"PackSelector('{self.ref}').include_ids"
            ),
        )
        object.__setattr__(
            self,
            "exclude_ids",
            _normalize_str_list(
                self.exclude_ids, label=f"PackSelector('{self.ref}').exclude_ids"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ref": self.ref}
        if self.name:
            out["name"] = self.name
        if self.priority is not None:
            out["priority"] = self.priority
        if self.replace_options is not None:
            out["replace_options"] = self.replace_options
        if self.options_from:
            out["options_from"] = [source.to_dict() for source in self.options_from]
        if self.options:
            out["options"] = [option.to_dict() for option in self.options]
        if self.include_ids or self.exclude_ids:
            out["filter"] = {}
            if self.include_ids:
                out["filter"]["include_ids"] = list(self.include_ids)
            if self.exclude_ids:
                out["filter"]["exclude_ids"] = list(self.exclude_ids)
        return out


@dataclass(frozen=True)
class PredicateArg:
    source_kind: str
    source_key: str
    has_default: bool = False
    default_value: Any = None

    def __post_init__(self) -> None:
        if self.source_kind not in {"pack_id", "param"}:
            raise ConfigError("PredicateArg.source_kind must be 'pack_id' or 'param'")
        object.__setattr__(
            self,
            "source_key",
            _ensure_non_empty_str(self.source_key, label="PredicateArg.source_key"),
        )

    @staticmethod
    def pack_id(
        pack_name: str, *, default: Any = None, has_default: bool = False
    ) -> "PredicateArg":
        if has_default:
            return PredicateArg(
                "pack_id", pack_name, has_default=True, default_value=deepcopy(default)
            )
        return PredicateArg("pack_id", pack_name)

    @staticmethod
    def param(
        path: str, *, default: Any = None, has_default: bool = False
    ) -> "PredicateArg":
        if has_default:
            return PredicateArg(
                "param", path, has_default=True, default_value=deepcopy(default)
            )
        return PredicateArg("param", path)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {self.source_kind: self.source_key}
        if self.has_default:
            out["default"] = deepcopy(self.default_value)
        return out


@dataclass(frozen=True, init=False)
class Predicate:
    predicate_id: str
    args: dict[str, PredicateArg]
    expr: Any
    on_error: str = "error"

    def __init__(
        self,
        id: str | None = None,
        args: Mapping[str, PredicateArg] | None = None,
        expr: Any = None,
        on_error: str = "error",
        *,
        predicate_id: str | None = None,
    ) -> None:
        if id is not None and predicate_id is not None:
            raise ConfigError("Predicate accepts exactly one of 'id' or 'predicate_id'")
        resolved_id = id if id is not None else predicate_id
        if resolved_id is None:
            raise ConfigError("Predicate requires 'id'")
        if args is None:
            raise ConfigError("Predicate.args is required")
        if expr is None:
            raise ConfigError("Predicate.expr is required")
        object.__setattr__(self, "predicate_id", resolved_id)
        object.__setattr__(self, "args", dict(args))
        object.__setattr__(self, "expr", deepcopy(expr))
        object.__setattr__(self, "on_error", on_error)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicate_id",
            _ensure_non_empty_str(self.predicate_id, label="Predicate.id"),
        )
        if not isinstance(self.args, Mapping) or not self.args:
            raise ConfigError(
                f"Predicate('{self.predicate_id}').args must be a non-empty mapping"
            )
        normalized: dict[str, PredicateArg] = {}
        for raw_name, arg in self.args.items():
            name = _ensure_non_empty_str(
                raw_name, label=f"Predicate('{self.predicate_id}').args key"
            )
            if name in normalized:
                raise ConfigError(
                    f"Predicate('{self.predicate_id}') has duplicate arg '{name}'"
                )
            if not isinstance(arg, PredicateArg):
                raise ConfigError(
                    f"Predicate('{self.predicate_id}').args['{name}'] must be PredicateArg"
                )
            normalized[name] = arg
        object.__setattr__(self, "args", normalized)

        if self.on_error not in {"error", "false"}:
            raise ConfigError(
                f"Predicate('{self.predicate_id}').on_error must be 'error' or 'false'"
            )

    @property
    def id(self) -> str:
        return self.predicate_id

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.predicate_id,
            "args": {name: arg.to_dict() for name, arg in self.args.items()},
            "expr": deepcopy(self.expr),
        }
        if self.on_error != "error":
            out["on_error"] = self.on_error
        return out


@dataclass(frozen=True)
class ConstraintRule:
    when: dict[str, tuple[str, ...]]
    mode: str = "exclude"

    def __post_init__(self) -> None:
        if self.mode not in {"include", "exclude"}:
            raise ConfigError("ConstraintRule.mode must be include or exclude")
        if not isinstance(self.when, Mapping) or not self.when:
            raise ConfigError("ConstraintRule.when must be a non-empty mapping")
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_pack, raw_ids in self.when.items():
            pack = _ensure_non_empty_str(raw_pack, label="ConstraintRule.when pack")
            if isinstance(raw_ids, str):
                ids = (raw_ids,)
            else:
                ids = _normalize_str_list(
                    raw_ids, label=f"ConstraintRule.when['{pack}']"
                )
            if not ids:
                raise ConfigError(
                    f"ConstraintRule.when['{pack}'] must include at least one id"
                )
            normalized[pack] = ids
        object.__setattr__(self, "when", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"when": {pack: list(ids) for pack, ids in self.when.items()}}


@dataclass(frozen=True)
class DefaultsSpec:
    params: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        params = _deep_mapping(self.params, label="DefaultsSpec.params")
        assert_no_dotted_keys(params, label="defaults.params")
        object.__setattr__(self, "params", params)
        object.__setattr__(
            self, "tags", _normalize_str_list(self.tags, label="DefaultsSpec.tags")
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.params:
            out["params"] = deepcopy(self.params)
        if self.tags:
            out["tags"] = list(self.tags)
        return out


@dataclass(frozen=True)
class MergeSpec:
    mode: str | None = None
    strategies: dict[str, str] = field(default_factory=dict)
    delete_sentinel: str | None = None

    def __post_init__(self) -> None:
        if self.mode is not None:
            mode_s = _ensure_non_empty_str(self.mode, label="MergeSpec.mode")
            if mode_s not in {"none", "merge"}:
                raise ConfigError("MergeSpec.mode must be one of ['merge', 'none']")
            object.__setattr__(self, "mode", mode_s)

        if not isinstance(self.strategies, Mapping):
            raise ConfigError("MergeSpec.strategies must be a mapping")
        normalized_strategies: dict[str, str] = {}
        allowed_strategies = {
            "error",
            "replace",
            "deep_merge",
            "append_unique",
            "set_union",
        }
        for raw_path, raw_strategy in self.strategies.items():
            path = _ensure_non_empty_str(raw_path, label="MergeSpec.strategies key")
            strategy = _ensure_non_empty_str(
                raw_strategy, label=f"MergeSpec.strategies['{path}']"
            )
            if strategy not in allowed_strategies:
                raise ConfigError(
                    f"MergeSpec.strategies['{path}'] must be one of {sorted(allowed_strategies)}"
                )
            normalized_strategies[path] = strategy
        object.__setattr__(self, "strategies", normalized_strategies)

        if self.delete_sentinel is not None:
            object.__setattr__(
                self,
                "delete_sentinel",
                _ensure_non_empty_str(
                    self.delete_sentinel, label="MergeSpec.delete_sentinel"
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.mode is not None:
            out["mode"] = self.mode
        if self.strategies:
            out["strategies"] = dict(self.strategies)
        if self.delete_sentinel is not None:
            out["delete_sentinel"] = self.delete_sentinel
        return out


@dataclass(frozen=True)
class RunSetSpec:
    extends: tuple[str, ...] = ()
    append: dict[str, Any] = field(default_factory=dict)
    replace: dict[str, Any] = field(default_factory=dict)
    patch_packs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extends",
            _normalize_str_list(self.extends, label="RunSetSpec.extends"),
        )
        append = _deep_mapping(self.append, label="RunSetSpec.append")
        replace = _deep_mapping(self.replace, label="RunSetSpec.replace")
        patch_packs = _deep_mapping(self.patch_packs, label="RunSetSpec.patch_packs")

        allowed_roots = {"select", "defaults", "constraints", "merge"}
        append_unknown = sorted(set(append.keys()) - allowed_roots)
        replace_unknown = sorted(set(replace.keys()) - allowed_roots)
        if append_unknown:
            raise ConfigError(f"RunSetSpec.append has unknown roots: {append_unknown}")
        if replace_unknown:
            raise ConfigError(
                f"RunSetSpec.replace has unknown roots: {replace_unknown}"
            )
        for raw_name, raw_patch in patch_packs.items():
            pack_name = _ensure_non_empty_str(
                raw_name, label="RunSetSpec.patch_packs key"
            )
            if not isinstance(raw_patch, Mapping):
                raise ConfigError(
                    f"RunSetSpec.patch_packs['{pack_name}'] must be a mapping"
                )

        object.__setattr__(self, "append", append)
        object.__setattr__(self, "replace", replace)
        object.__setattr__(self, "patch_packs", patch_packs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.extends:
            out["extends"] = list(self.extends)
        if self.append:
            out["append"] = deepcopy(self.append)
        if self.replace:
            out["replace"] = deepcopy(self.replace)
        if self.patch_packs:
            out["patch_packs"] = deepcopy(self.patch_packs)
        return out
