from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

_log = logging.getLogger("geryon.utils")


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha1_hex(value: Any) -> str:
    payload = value if isinstance(value, str) else stable_json(value)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def flatten_dict(
    data: Mapping[str, Any], prefix: str = "", sep: str = "."
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_dict(value, prefix=full_key, sep=sep))
        else:
            out[full_key] = value
    return out


def serialize_hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return stable_json(value)
    return str(value)


def render_hydra_override(key: str, value: Any) -> str:
    return shlex.quote(f"{key}={serialize_hydra_value(value)}")


def parse_cpulist(cpulist: str) -> list[int]:
    values: list[int] = []
    for token in cpulist.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            lo = int(left)
            hi = int(right)
            if hi < lo:
                raise ValueError(f"Invalid cpu range '{token}'")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(token))
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content)
        tmp.replace(path)
    except Exception:
        # Clean up the temp file so we don't leave partial writes on disk.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())

    def _iter() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    _log.warning(
                        "Skipping corrupt JSONL line %d in %s",
                        line_number,
                        path,
                    )

    return _iter()


def read_jsonl_with_stats(
    path: Path, *, strict: bool = False
) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    records: list[dict[str, Any]] = []
    corrupt_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                corrupt_lines += 1
                message = f"Corrupt JSONL line {line_number} in {path}"
                if strict:
                    raise ValueError(message) from exc
                _log.warning("%s", message)
    return records, corrupt_lines


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(stable_json(dict(record)) + "\n")
    except OSError as exc:
        _log.error("Failed to append JSONL record to %s: %s", path, exc)
        raise


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json(dict(record)) + "\n")


def sanitize_for_path(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("-")
    out = "".join(safe).strip("-")
    return out or "x"


def read_text_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    return [line for line in lines if line.strip() and not line.strip().startswith("#")]


def env_default(key: str, fallback: str) -> str:
    value = os.environ.get(key)
    return value if value else fallback
