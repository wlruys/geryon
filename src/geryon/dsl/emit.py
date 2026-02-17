from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def dump_yaml(data: dict[str, Any], *, sort_keys: bool = False) -> str:
    text = yaml.safe_dump(
        data,
        sort_keys=sort_keys,
        default_flow_style=False,
        allow_unicode=False,
        indent=2,
    )
    if not text.endswith("\n"):
        text += "\n"
    return text


def write_yaml(
    path: str | Path, data: dict[str, Any], *, sort_keys: bool = False
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_yaml(data, sort_keys=sort_keys), encoding="utf-8")
    return out_path
