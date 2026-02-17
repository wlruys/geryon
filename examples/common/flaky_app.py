#!/usr/bin/env python3
"""Test application that can simulate failures, timeouts, and flaky behavior.

Used by resilience/recovery examples (04_execution_resilience,
05_workflow_and_recovery). Behaves like dummy_hydra_app.py for config
resolution, but
adds controllable failure modes via special parameters:

    fail.mode       - "none" (default), "always", "random", "slow"
    fail.exit_code  - exit code on failure (default 1)
    fail.sleep_sec  - seconds to sleep (for "slow" mode; default 10)
    fail.rate       - probability of failure for "random" mode (0-1; default 0.5)
"""

from __future__ import annotations

import json
import random
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
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def main(argv: list[str]) -> int:
    config: dict[str, Any] = {}
    for token in argv:
        if token.startswith("-") or "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            continue
        _assign_dotted_key(config, key, _parse_scalar(raw_value))

    print(
        "RESOLVED_CONFIG_JSON="
        + json.dumps(config, sort_keys=True, separators=(",", ":"))
    )

    fail = config.get("fail", {})
    if not isinstance(fail, dict):
        fail = {}

    mode = str(fail.get("mode", "none"))
    exit_code = int(fail.get("exit_code", 1))
    sleep_sec = float(fail.get("sleep_sec", 10))
    rate = float(fail.get("rate", 0.5))

    if mode == "always":
        print(f"SIMULATED_FAILURE mode={mode}", file=sys.stderr)
        return exit_code

    if mode == "random":
        if random.random() < rate:
            print(f"SIMULATED_FAILURE mode={mode} rate={rate}", file=sys.stderr)
            return exit_code
        print(f"SIMULATED_SUCCESS mode={mode} rate={rate}")
        return 0

    if mode == "slow":
        print(f"SIMULATED_SLOW sleep_sec={sleep_sec}")
        time.sleep(sleep_sec)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
