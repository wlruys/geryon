from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from geryon.models import ConfigError

from geryon.dsl.specs import Option, Pack, Predicate, PredicateArg


def _sanitize_token(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "v"


def _build_arg_name(prefix: str, source: str, *, used: set[str]) -> str:
    base = f"{prefix}_{re.sub(r'[^A-Za-z0-9]+', '_', source).strip('_') or 'v'}"
    name = base
    suffix = 1
    while name in used:
        suffix += 1
        name = f"{base}_{suffix}"
    used.add(name)
    return name


def _nested_param(path: str, value: Any) -> dict[str, Any]:
    if not isinstance(path, str) or not path.strip():
        raise ConfigError("param path must be a non-empty dotted path")
    parts = [part for part in path.strip().split(".") if part]
    if not parts:
        raise ConfigError("param path must be a non-empty dotted path")

    leaf: Any = value
    for part in reversed(parts):
        leaf = {part: leaf}
    return leaf


def _default_value_token(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _build_constraint_expr(arg_ref: str, constraint: Any) -> dict[str, list[Any]]:
    if isinstance(constraint, tuple):
        if not constraint:
            raise ConfigError("constraint tuple must not be empty")
        op = str(constraint[0]).strip()
        if not op:
            raise ConfigError("constraint operator must be non-empty")
        if op == "between":
            if len(constraint) != 3:
                raise ConfigError("between constraint must be ('between', lo, hi)")
            return {"between": [arg_ref, constraint[1], constraint[2]]}
        if op in {"in", "not_in"}:
            if len(constraint) != 2:
                raise ConfigError(f"{op} constraint must be ('{op}', values)")
            values = constraint[1]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ConfigError(f"{op} constraint values must be a sequence")
            return {op: [arg_ref, list(values)]}
        if len(constraint) != 2:
            raise ConfigError(f"{op} constraint must be ('{op}', value)")
        return {op: [arg_ref, constraint[1]]}
    return {"eq": [arg_ref, constraint]}


def require(
    *,
    rules: list[tuple[dict[str, Any], dict[str, Any]]],
    predicate_id: str = "require",
    on_error: str = "error",
) -> Predicate:
    if not isinstance(rules, list) or not rules:
        raise ConfigError("rules must be a non-empty list of (when, then) tuples")

    used_arg_names: set[str] = set()
    args: dict[str, PredicateArg] = {}
    pack_args: dict[str, str] = {}
    param_args: dict[str, str] = {}
    clauses: list[dict[str, Any]] = []

    def get_pack_arg(pack_name: str) -> str:
        if pack_name not in pack_args:
            arg_name = _build_arg_name("pack", pack_name, used=used_arg_names)
            pack_args[pack_name] = arg_name
            args[arg_name] = PredicateArg.pack_id(pack_name)
        return pack_args[pack_name]

    def get_param_arg(param_path: str) -> str:
        if param_path not in param_args:
            arg_name = _build_arg_name("param", param_path, used=used_arg_names)
            param_args[param_path] = arg_name
            args[arg_name] = PredicateArg.param(param_path)
        return param_args[param_path]

    for index, rule in enumerate(rules):
        if not isinstance(rule, tuple) or len(rule) != 2:
            raise ConfigError(f"rules[{index}] must be a (when, then) tuple")
        when, then = rule
        if not isinstance(when, Mapping) or not isinstance(then, Mapping):
            raise ConfigError(f"rules[{index}] must contain two mappings: (when, then)")
        if not when:
            raise ConfigError(f"rules[{index}].when must not be empty")
        if not then:
            raise ConfigError(f"rules[{index}].then must not be empty")

        conjuncts: list[dict[str, Any]] = []
        for raw_pack_name, raw_option_id in when.items():
            pack_name = str(raw_pack_name).strip()
            option_id = str(raw_option_id).strip()
            if not pack_name:
                raise ConfigError(f"rules[{index}].when has an empty pack name")
            if not option_id:
                raise ConfigError(
                    f"rules[{index}].when['{pack_name}'] must be a non-empty option id"
                )
            pack_arg = get_pack_arg(pack_name)
            conjuncts.append({"eq": [f"${pack_arg}", option_id]})

        for raw_param_path, constraint in then.items():
            param_path = str(raw_param_path).strip()
            if not param_path:
                raise ConfigError(f"rules[{index}].then has an empty param path")
            param_arg = get_param_arg(param_path)
            conjuncts.append(_build_constraint_expr(f"${param_arg}", constraint))

        clauses.append({"and": conjuncts})

    return Predicate(
        id=predicate_id,
        args=args,
        expr={"or": clauses},
        on_error=on_error,
    )


def pack_param_values(
    *,
    pack_name: str,
    param_path: str,
    values: Iterable[Any],
    id_prefix: str | None = None,
    tag_prefix: str | None = None,
    value_to_id: Callable[[Any], str] | None = None,
) -> Pack:
    value_list = list(values)
    if not value_list:
        raise ConfigError("values must be non-empty")

    prefix = (
        id_prefix.strip() if isinstance(id_prefix, str) and id_prefix.strip() else None
    )
    options: list[Option] = []
    for index, value in enumerate(value_list):
        token_raw = value_to_id(value) if value_to_id else _default_value_token(value)
        token = _sanitize_token(token_raw)
        option_id = f"{prefix}_{token}" if prefix else token
        if option_id in {opt.option_id for opt in options}:
            option_id = f"{option_id}_{index:02d}"
        tag = f"{tag_prefix}:{token}" if tag_prefix else None
        options.append(
            Option(
                option_id=option_id,
                params=_nested_param(param_path, value),
                tag=tag,
            )
        )

    return Pack(name=pack_name, options=tuple(options))


def pack_map(
    *,
    pack_name: str,
    options: Mapping[str, Mapping[str, Any]],
    tag_prefix: str | None = None,
) -> Pack:
    """Create a Pack from an ordered mapping of option_id -> params.

    Entry formats:
    - {"option_id": {"train": {"lr": 1e-3}}}
    - {"option_id": {"params": {...}, "tag": "..."}}
    """

    if not isinstance(options, Mapping) or not options:
        raise ConfigError("options must be a non-empty mapping")

    built_options: list[Option] = []
    for raw_id, raw_value in options.items():
        option_id = str(raw_id).strip()
        if not option_id:
            raise ConfigError("pack_map option ids must be non-empty strings")
        if not isinstance(raw_value, Mapping):
            raise ConfigError(f"pack_map option '{option_id}' must be a mapping")

        value_map = dict(raw_value)
        if "params" in value_map:
            unknown = sorted(set(value_map.keys()) - {"params", "tag"})
            if unknown:
                raise ConfigError(
                    f"pack_map option '{option_id}' has unknown keys: {unknown}"
                )
            params = value_map.get("params")
            tag = value_map.get("tag")
        else:
            params = value_map
            tag = f"{tag_prefix}:{_sanitize_token(option_id)}" if tag_prefix else None

        if not isinstance(params, Mapping):
            raise ConfigError(f"pack_map option '{option_id}'.params must be a mapping")

        built_options.append(
            Option(
                option_id=option_id,
                params=dict(params),  # Option validates deep mapping structure.
                tag=tag,
            )
        )

    return Pack(name=pack_name, options=tuple(built_options))


def match_ids(
    *,
    left_pack: str,
    right_pack: str,
    pairs: Iterable[tuple[str, str]],
    predicate_id: str | None = None,
    left_arg: str = "left",
    right_arg: str = "right",
    on_error: str = "error",
) -> Predicate:
    """Build a compact ID matching predicate over two pack IDs."""

    normalized_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_pair in enumerate(pairs):
        if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
            raise ConfigError(f"pairs[{index}] must be a tuple(str, str)")
        left_id = str(raw_pair[0]).strip()
        right_id = str(raw_pair[1]).strip()
        if not left_id or not right_id:
            raise ConfigError(f"pairs[{index}] must contain non-empty ids")
        pair = (left_id, right_id)
        if pair in seen:
            continue
        seen.add(pair)
        normalized_pairs.append(pair)

    if not normalized_pairs:
        raise ConfigError("pairs must include at least one (left_id, right_id) entry")

    pred_id = (
        predicate_id
        or f"match_{_sanitize_token(left_pack)}_{_sanitize_token(right_pack)}"
    )
    expr = {
        "or": [
            {
                "and": [
                    {"eq": [f"${left_arg}", left_id]},
                    {"eq": [f"${right_arg}", right_id]},
                ]
            }
            for left_id, right_id in normalized_pairs
        ]
    }
    return Predicate(
        predicate_id=pred_id,
        args={
            left_arg: PredicateArg.pack_id(left_pack),
            right_arg: PredicateArg.pack_id(right_pack),
        },
        expr=expr,
        on_error=on_error,
    )


def pack_param_linspace(
    *,
    pack_name: str,
    param_path: str,
    start: float,
    stop: float,
    num: int,
    id_prefix: str | None = None,
    tag_prefix: str | None = None,
) -> Pack:
    if num <= 0:
        raise ConfigError("num must be positive")
    if num == 1:
        values = [float(start)]
    else:
        step = (float(stop) - float(start)) / float(num - 1)
        values = [float(start) + step * idx for idx in range(num)]
    return pack_param_values(
        pack_name=pack_name,
        param_path=param_path,
        values=values,
        id_prefix=id_prefix,
        tag_prefix=tag_prefix,
    )


def pack_param_logspace(
    *,
    pack_name: str,
    param_path: str,
    start_exp: float,
    stop_exp: float,
    num: int,
    base: float = 10.0,
    id_prefix: str | None = None,
    tag_prefix: str | None = None,
) -> Pack:
    if num <= 0:
        raise ConfigError("num must be positive")
    if base <= 0.0 or math.isclose(base, 1.0):
        raise ConfigError("base must be positive and not equal to 1")

    if num == 1:
        exps = [float(start_exp)]
    else:
        step = (float(stop_exp) - float(start_exp)) / float(num - 1)
        exps = [float(start_exp) + step * idx for idx in range(num)]

    values = [base**exp for exp in exps]
    return pack_param_values(
        pack_name=pack_name,
        param_path=param_path,
        values=values,
        id_prefix=id_prefix,
        tag_prefix=tag_prefix,
    )
