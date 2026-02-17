#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import sys
import time
from typing import Any


def _assign_dotted_key(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""

    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None

    if value[0] in {"[", "{", '"'}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        pass

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def main(argv: list[str]) -> int:
    config: dict[str, Any] = {}
    passthrough_args: list[str] = []

    for token in argv:
        if token.startswith("-") or "=" not in token:
            passthrough_args.append(token)
            continue
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            passthrough_args.append(token)
            continue
        _assign_dotted_key(config, key, _parse_scalar(raw_value))

    time.sleep(10)

    print(
        "RESOLVED_CONFIG_JSON="
        + json.dumps(config, sort_keys=True, separators=(",", ":"))
    )
    if passthrough_args:
        print("UNPARSED_ARGS_JSON=" + json.dumps(passthrough_args, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
