from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast

from rich import box
from rich.live import Live
from rich.console import Console, Group
from rich.table import Table
import yaml

from geryon._logging import setup_logging
from geryon.config_compose import compose_experiment_data
from geryon.collect import collect_run
from geryon.execution import (
    build_task_payloads,
    execute_task_payload,
    select_batch_indices,
)
from geryon.models import ConfigError, ResolvedSlurmConfig, SlurmParameterValue
from geryon.planner import (
    get_experiment_run_sets,
    parse_experiment_yaml_with_diagnostics,
    plan_experiment,
)
from geryon.profiles import (
    RunProfile,
    load_profiles,
    merge_profile_env,
    resolve_profile,
)
from geryon.status import (
    RERUN_STATUS_FILTERS,
    build_run_report_payload,
    build_run_status_index,
    render_report_markdown,
    select_rerun_config_ids,
    summarize_run_status,
    summarize_status_groups,
)
from geryon.store import ArtifactStore
from geryon.submitit_backend import submit_payloads
from geryon.utils import append_jsonl, sanitize_for_path, utc_now_iso

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SLURM_SCALAR_FIELDS = (
    "partition",
    "time_min",
    "cpus_per_task",
    "mem_gb",
    "gpus_per_node",
    "job_name",
    "mail_user",
    "mail_type",
    "query_status",
)
_SLURM_SCALAR_BUILTINS: dict[str, Any] = {
    "partition": None,
    "time_min": None,
    "cpus_per_task": 1,
    "mem_gb": 16,
    "gpus_per_node": 0,
    "job_name": "geryon",
    "mail_user": None,
    "mail_type": None,
    "query_status": False,
}
_SBATCH_RESERVED_MANAGED_KEYS = {
    "partition",
    "time",
    "time_min",
    "cpus_per_task",
    "mem_gb",
    "gpus_per_node",
    "name",
    "job_name",
    "mail_user",
    "mail_type",
}


class _LocalConcurrency(TypedDict):
    max_concurrent_tasks: int
    cores_per_task: int
    max_total_cores: int | None


@dataclass(frozen=True)
class _RunExecutionContext:
    store: ArtifactStore
    batches: list[int]
    profile: RunProfile | None
    profiles_path: Path
    env_script: str | None
    env_setup_cmds: list[str]
    env_vars: dict[str, str]
    retry_on_status: list[str]
    fail_fast_threshold: float | None
    selected_config_ids: list[str]
    local_concurrency: _LocalConcurrency
    resolved_defaults_sources: dict[str, str]


_RUN_LOCAL_BUILTIN_DEFAULTS: dict[str, Any] = {
    "executor": "process",
    "tmux_prefix": "geryon",
    "max_concurrent_tasks": 1,
    "cores_per_task": 1,
    "max_total_cores": None,
    "cores": None,
    "batches_per_task": 1,
    "command_timeout_sec": None,
    "max_retries": 0,
    "retry_on_status": None,
    "max_failures": None,
    "fail_fast_threshold": None,
    "resume": False,
    "progress": False,
}
_RUN_SLURM_BUILTIN_DEFAULTS: dict[str, Any] = {
    "executor": "tmux",
    "tmux_prefix": "geryon",
    "max_concurrent_tasks": 1,
    "cores_per_task": 1,
    "max_total_cores": None,
    "cores": None,
    "batches_per_task": 1,
    "command_timeout_sec": None,
    "max_retries": 0,
    "retry_on_status": None,
    "max_failures": None,
    "fail_fast_threshold": None,
    "resume": False,
}


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    if isinstance(value, float):
        return int(value)
    return 0


def _short_config_id(config_id: str, *, width: int = 10) -> str:
    text = str(config_id)
    return text if len(text) <= width else text[:width]


def _short_text(value: object, *, width: int = 100) -> str:
    text = str(value)
    return text if len(text) <= width else f"{text[: max(width - 3, 1)]}..."


def _format_name_list(values: list[str], *, max_items: int = 3) -> str:
    items = [item for item in values if item]
    if not items:
        return "-"
    head = items[:max_items]
    text = ", ".join(head)
    remaining = len(items) - len(head)
    if remaining > 0:
        text = f"{text} (+{remaining})"
    return text


def _format_cores(value: object) -> str:
    if isinstance(value, list):
        if not value:
            return "-"
        return ",".join(str(item) for item in value)
    if isinstance(value, tuple):
        if not value:
            return "-"
        return ",".join(str(item) for item in value)
    if value is None:
        return "-"
    text = str(value).strip()
    return text or "-"


def _format_concurrency(
    *,
    max_concurrent_tasks: int,
    cores_per_task: int,
    max_total_cores: int | None,
) -> str:
    total_text = "pool" if max_total_cores is None else str(max_total_cores)
    return f"{max_concurrent_tasks}x{cores_per_task} <= {total_text}"


def _render_local_live_dashboard(state: dict[str, Any]) -> Group:
    summary = Table(title="Local Run Live", show_header=False, box=box.SIMPLE)
    summary.add_column("Field", style="bold cyan")
    summary.add_column("Value")
    summary.add_row("Run ID", str(state.get("run_id", "")))
    summary.add_row("Executor", str(state.get("executor", "")))
    summary.add_row("Conc", str(state.get("concurrency", "-")))
    summary.add_row(
        "Payloads",
        f"{_int_value(state.get('payload_done'))}/{_int_value(state.get('payload_total'))}",
    )
    summary.add_row("Current", str(state.get("current_payload", "-")))
    summary.add_row("Active", str(len(_mapping(state.get("active")))))
    summary.add_row(
        "Done",
        str(
            _int_value(state.get("success"))
            + _int_value(state.get("failed"))
            + _int_value(state.get("terminated"))
        ),
    )
    summary.add_row(
        "Status",
        f"ok={_int_value(state.get('success'))} "
        f"f={_int_value(state.get('failed'))} "
        f"t={_int_value(state.get('terminated'))}",
    )
    summary.add_row("Retry", str(_int_value(state.get("retry_scheduled"))))

    active_entries = list(_mapping(state.get("active")).values())
    active_entries.sort(key=lambda item: str(_mapping(item).get("time", "")))
    active_tmux_sessions = sorted(
        {
            str(_mapping(item).get("tmux_session", "")).strip()
            for item in active_entries
            if str(_mapping(item).get("tmux_session", "")).strip()
        }
    )
    summary.add_row("Tmux Active", _format_name_list(active_tmux_sessions))

    running = Table(title="Running", box=box.SIMPLE)
    running.add_column("Exec")
    running.add_column("Task")
    running.add_column("B:L", justify="right")
    running.add_column("Config")
    running.add_column("Cores")
    running.add_column("PID", justify="right")
    running.add_column("Sess")
    for raw in active_entries:
        item = _mapping(raw)
        running.add_row(
            str(item.get("executor", "")),
            f"{item.get('job_id', '')}/{item.get('array_task_id', '')}",
            f"{item.get('batch_index', '-')}/{item.get('line_index', '-')}",
            _short_config_id(str(item.get("config_id", ""))),
            _format_cores(item.get("assigned_cores")),
            str(item.get("pid", "") or "-"),
            str(item.get("tmux_session", "") or "-"),
        )
    if not active_entries:
        running.add_row("<none>", "-", "-", "-", "-", "-", "-")

    started_rows = list(state.get("recent_starts", []))
    started = Table(title="Started (latest)", box=box.SIMPLE)
    started.add_column("Exec")
    started.add_column("Task")
    started.add_column("B:L", justify="right")
    started.add_column("Config")
    started.add_column("Cores")
    started.add_column("PID", justify="right")
    started.add_column("Sess")
    for raw in started_rows:
        item = _mapping(raw)
        started.add_row(
            str(item.get("executor", "")),
            f"{item.get('job_id', '')}/{item.get('array_task_id', '')}",
            f"{item.get('batch_index', '-')}/{item.get('line_index', '-')}",
            _short_config_id(str(item.get("config_id", ""))),
            _format_cores(item.get("assigned_cores")),
            str(item.get("pid", "") or "-"),
            str(item.get("tmux_session", "") or "-"),
        )
    if not started_rows:
        started.add_row("-", "-", "-", "-", "-", "-", "-")

    recent_rows = list(state.get("recent_completed", []))
    recent = Table(title="Completed (latest)", box=box.SIMPLE)
    recent.add_column("St")
    recent.add_column("Exec")
    recent.add_column("B:L", justify="right")
    recent.add_column("Config")
    recent.add_column("Dur", justify="right")
    recent.add_column("Cores")
    recent.add_column("PID", justify="right")
    for raw in recent_rows:
        item = _mapping(raw)
        status = str(item.get("status", ""))
        status_style = status
        if status == "success":
            status_style = "[green]ok[/green]"
        elif status == "failed":
            status_style = "[red]fail[/red]"
        elif status == "terminated":
            status_style = "[yellow]term[/yellow]"
        recent.add_row(
            status_style,
            str(item.get("executor", "")),
            f"{item.get('batch_index', '-')}/{item.get('line_index', '-')}",
            _short_config_id(str(item.get("config_id", ""))),
            str(item.get("duration_sec", "-")),
            _format_cores(item.get("assigned_cores")),
            str(item.get("pid", "") or "-"),
        )
    if not recent_rows:
        recent.add_row("-", "-", "-", "-", "-", "-", "-")
    return Group(summary, running, started, recent)


def _query_slurm_queue_status(job_ids: list[str]) -> dict[str, Any]:
    cleaned_ids = [str(job_id).strip() for job_id in job_ids if str(job_id).strip()]
    cleaned_ids = list(dict.fromkeys(cleaned_ids))
    base: dict[str, Any] = {
        "source": "squeue",
        "job_ids": cleaned_ids,
        "available": False,
        "queried": False,
        "rows": [],
        "by_state": {},
        "num_rows": 0,
        "error": None,
    }
    if not cleaned_ids:
        base["error"] = "No submitted job IDs to query."
        return base

    cmd = [
        "squeue",
        "-h",
        "-j",
        ",".join(cleaned_ids),
        "-o",
        "%i|%t|%M|%D|%P|%j|%u|%R",
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        base["error"] = "squeue was not found in PATH."
        return base
    except subprocess.CalledProcessError as exc:
        msg = (exc.stderr or exc.stdout or str(exc)).strip()
        base["error"] = msg or f"squeue failed with exit code {exc.returncode}."
        return base

    rows: list[dict[str, str]] = []
    by_state: dict[str, int] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 8:
            parts.extend([""] * (8 - len(parts)))
        parts = parts[:8]
        row = {
            "job_id": parts[0].strip(),
            "state": parts[1].strip(),
            "time": parts[2].strip(),
            "nodes": parts[3].strip(),
            "partition": parts[4].strip(),
            "name": parts[5].strip(),
            "user": parts[6].strip(),
            "reason": parts[7].strip(),
        }
        rows.append(row)
        state = row["state"] or "unknown"
        by_state[state] = by_state.get(state, 0) + 1

    base["available"] = True
    base["queried"] = True
    base["rows"] = rows
    base["by_state"] = by_state
    base["num_rows"] = len(rows)
    return base


def _validate_env_var_name(name: str, *, source: str) -> str:
    key = str(name).strip()
    if not key:
        raise ConfigError(f"{source}: environment variable name cannot be empty.")
    if not _ENV_NAME_RE.match(key):
        raise ConfigError(f"{source}: invalid environment variable name '{key}'.")
    return key


def _parse_env_args(raw_items: list[str] | None) -> dict[str, str]:
    if not raw_items:
        return {}
    parsed: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ConfigError(f"Invalid --env '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = _validate_env_var_name(key, source=f"--env '{item}'")
        parsed[key] = value
    return parsed


def _parse_retry_on_status(raw_items: list[str] | None) -> list[str]:
    if not raw_items:
        return ["failed", "terminated"]
    allowed = {"failed", "terminated"}
    out: list[str] = []
    for item in raw_items:
        for token in str(item).split(","):
            value = token.strip()
            if not value:
                continue
            if value not in allowed:
                raise ConfigError(
                    f"Invalid retry status '{value}'. Expected one of {sorted(allowed)}."
                )
            if value not in out:
                out.append(value)
    return out or ["failed", "terminated"]


def _parse_fail_fast_threshold(raw_value: str | float | int | None) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        value = float(raw_value)
        if value > 1.0:
            value = value / 100.0
    else:
        text = str(raw_value).strip()
        if not text:
            return None
        try:
            if text.endswith("%"):
                value = float(text[:-1]) / 100.0
            else:
                value = float(text)
                if value > 1.0:
                    value = value / 100.0
        except ValueError as exc:
            raise ConfigError(
                f"Invalid fail_fast_threshold '{raw_value}'. Use decimal (0-1) or percent (e.g. 20%)."
            ) from exc
    if not (0.0 < value <= 1.0):
        raise ConfigError("fail_fast_threshold must be in (0, 1] or (0%, 100%].")
    return value


def _parse_by_pack_args(raw_items: list[str] | None) -> list[str]:
    if not raw_items:
        return []
    parsed: list[str] = []
    for item in raw_items:
        for token in str(item).split(","):
            pack = token.strip()
            if not pack:
                continue
            if pack not in parsed:
                parsed.append(pack)
    return parsed


def _normalize_sbatch_option_key(raw_key: str, *, source: str) -> str:
    key = str(raw_key).strip()
    if not key:
        raise ConfigError(f"Invalid {source}: option name cannot be empty.")
    return key.replace("-", "_").lower()


def _normalize_sbatch_option_value(
    value: SlurmParameterValue,
    *,
    source: str,
    key: str,
) -> SlurmParameterValue:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ConfigError(
                f"Invalid {source}: sbatch option '{key}' cannot be an empty string."
            )
        return cleaned
    return value


def _parse_sbatch_option_args(
    raw_items: list[str] | None,
) -> dict[str, SlurmParameterValue]:
    if not raw_items:
        return {}
    parsed: dict[str, SlurmParameterValue] = {}
    for item in raw_items:
        if "=" in item:
            raw_key, raw_value = item.split("=", 1)
            key = _normalize_sbatch_option_key(
                raw_key, source=f"sbatch_option '{item}'"
            )
            parsed[key] = _normalize_sbatch_option_value(
                raw_value, source=f"sbatch_option '{item}'", key=key
            )
            continue
        key = _normalize_sbatch_option_key(item, source=f"sbatch_option '{item}'")
        parsed[key] = True
    return parsed


def _merge_sbatch_option_layers(
    layers: list[tuple[str, dict[str, SlurmParameterValue]]],
) -> tuple[dict[str, SlurmParameterValue], dict[str, str], list[str]]:
    resolved: dict[str, SlurmParameterValue] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []
    for source, raw_values in layers:
        for raw_key, raw_value in raw_values.items():
            key = _normalize_sbatch_option_key(
                raw_key, source=f"{source}.sbatch_option"
            )
            if key in _SBATCH_RESERVED_MANAGED_KEYS:
                raise ConfigError(
                    "sbatch option "
                    f"'{raw_key}' is managed by dedicated fields "
                    "(partition/time/cpus_per_task/mem_gb/gpus_per_node/job_name/mail_*)."
                )
            value = _normalize_sbatch_option_value(
                raw_value,
                source=f"{source}.sbatch_option.{raw_key}",
                key=key,
            )
            if key in resolved and resolved[key] != value:
                previous_source = sources.get(key, "unknown")
                warnings.append(
                    f"sbatch option '{key}' overridden by {source} (was {previous_source})."
                )
            resolved[key] = value
            sources[key] = source
    return resolved, sources, warnings


def _merge_slurm_setup_layers(
    layers: list[tuple[str, list[str]]],
) -> tuple[list[str], dict[str, str], list[str]]:
    merged: list[str] = []
    sources: dict[str, str] = {}
    warnings: list[str] = []
    for source, commands in layers:
        for raw_cmd in commands:
            cmd = str(raw_cmd).strip()
            if not cmd:
                continue
            if cmd in sources:
                warnings.append(
                    f"duplicate slurm setup command ignored from {source}: {cmd!r}"
                )
                continue
            merged.append(cmd)
            sources[cmd] = source
    return merged, sources, warnings


def _resolve_local_concurrency_payload(
    *,
    max_concurrent_tasks: int,
    cores_per_task: int,
    max_total_cores: int | None,
) -> _LocalConcurrency:
    if isinstance(max_concurrent_tasks, bool) or not isinstance(
        max_concurrent_tasks, int
    ):
        raise ConfigError("max_concurrent_tasks must be an integer >= 1")
    if isinstance(cores_per_task, bool) or not isinstance(cores_per_task, int):
        raise ConfigError("cores_per_task must be an integer >= 1")
    if max_total_cores is not None and (
        isinstance(max_total_cores, bool) or not isinstance(max_total_cores, int)
    ):
        raise ConfigError("max_total_cores must be an integer >= 1 when provided")

    if max_concurrent_tasks <= 0:
        raise ConfigError("max_concurrent_tasks must be >= 1")
    if cores_per_task <= 0:
        raise ConfigError("cores_per_task must be >= 1")
    if max_total_cores is not None:
        if max_total_cores <= 0:
            raise ConfigError("max_total_cores must be >= 1")
        allowed = max_total_cores // cores_per_task
        if allowed < 1:
            raise ConfigError("max_total_cores must be >= cores_per_task")
        max_concurrent_tasks = min(max_concurrent_tasks, allowed)
    return cast(
        _LocalConcurrency,
        {
            "max_concurrent_tasks": max_concurrent_tasks,
            "cores_per_task": cores_per_task,
            "max_total_cores": max_total_cores,
        },
    )


def _console() -> Console:
    return Console(highlight=False)


def _resolve_run_id(store: ArtifactStore) -> str:
    if not store.run_meta_path.exists():
        return store.run_id
    run_meta = store.read_run_meta()
    run_id = str(run_meta.get("run_id", "")).strip()
    return run_id or store.run_id


def _append_launcher_event(store: ArtifactStore, event: dict[str, Any]) -> None:
    payload = {"time": utc_now_iso(), **event}
    try:
        append_jsonl(store.launcher_log_path, payload)
    except OSError as exc:
        _cli_log.warning(
            "Failed to write launcher event to %s: %s", store.launcher_log_path, exc
        )


def _preflight_env_script(env_script: str | None) -> str | None:
    if not env_script:
        return None
    script_path = Path(env_script).expanduser().resolve()
    if not script_path.exists():
        raise ConfigError(f"Environment setup script not found: {script_path}")
    if not script_path.is_file():
        raise ConfigError(f"Environment setup script is not a file: {script_path}")
    return str(script_path)


def _load_retry_metadata(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    retry_path = Path(path).expanduser().resolve()
    if not retry_path.exists():
        raise ConfigError(f"Retry metadata file not found: {retry_path}")

    try:
        payload = json.loads(retry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid retry metadata JSON: {retry_path}") from exc
    raw_ids = payload.get("target_config_ids", [])
    if not isinstance(raw_ids, list):
        raise ConfigError(
            f"Retry metadata 'target_config_ids' must be a list: {retry_path}"
        )
    return {
        "retry_path": str(retry_path),
        "run_id": payload.get("run_id"),
        "run_root": payload.get("run_root"),
        "target_config_ids": [str(item) for item in raw_ids if str(item)],
    }


def _merge_selected_config_ids(
    direct_ids: list[str] | None,
    retry_file: str | None,
    *,
    expected_run_root: Path,
    expected_run_id: str,
) -> list[str]:
    direct = [str(item) for item in (direct_ids or [])]
    retry = _load_retry_metadata(retry_file)
    retry_ids: list[str] = []
    if retry is not None:
        retry_ids = list(retry.get("target_config_ids", []))
        retry_run_root_raw = retry.get("run_root")
        if retry_run_root_raw:
            retry_run_root = Path(str(retry_run_root_raw)).expanduser().resolve()
            if retry_run_root != expected_run_root.resolve():
                raise ConfigError(
                    "Retry metadata run root mismatch: "
                    f"retry_file={retry_run_root}, --run={expected_run_root.resolve()}"
                )
        retry_run_id_raw = retry.get("run_id")
        if retry_run_id_raw:
            retry_run_id = str(retry_run_id_raw).strip()
            if retry_run_id and retry_run_id != expected_run_id:
                raise ConfigError(
                    "Retry metadata run id mismatch: "
                    f"retry_file={retry_run_id}, run={expected_run_id}"
                )
    return list(dict.fromkeys([*direct, *retry_ids]))


def _validate_selected_config_ids(
    store: ArtifactStore, selected_config_ids: list[str]
) -> None:
    if not selected_config_ids:
        return
    planned_ids = {cfg.config_id for cfg in store.read_planned_configs()}
    unknown = sorted(
        config_id for config_id in selected_config_ids if config_id not in planned_ids
    )
    if unknown:
        raise ConfigError(f"Unknown config IDs selected: {unknown}")


def _resolve_run_defaults(
    args: argparse.Namespace,
    *,
    command: str,
    profile: RunProfile | None,
) -> dict[str, str]:
    if command not in {"run_local", "run_slurm"}:
        raise ConfigError(f"Unknown run command for defaults resolution: {command}")

    def _describe_sources(items: list[str]) -> str:
        if not items:
            return "builtin"
        if len(items) == 1:
            return items[0]
        return f"merged({'+'.join(items)})"

    builtins = (
        _RUN_LOCAL_BUILTIN_DEFAULTS
        if command == "run_local"
        else _RUN_SLURM_BUILTIN_DEFAULTS
    )
    profile_defaults = (
        dict(profile.run_local_defaults)
        if profile and command == "run_local"
        else dict(profile.run_slurm_defaults)
        if profile
        else {}
    )

    sources: dict[str, str] = {}
    for key, builtin_value in builtins.items():
        resolved = builtin_value
        used_sources: list[str] = []

        if key in profile_defaults and profile_defaults[key] is not None:
            resolved = profile_defaults[key]
            used_sources.append("profile.defaults")

        cli_value = getattr(args, key, None)
        if cli_value is not None:
            resolved = cli_value
            used_sources.append("cli")

        if not used_sources:
            used_sources.append("builtin")
        setattr(args, key, resolved)
        sources[key] = _describe_sources(used_sources)

    return sources


def _resolve_slurm_config(
    *,
    args: argparse.Namespace,
    profile: RunProfile | None,
) -> ResolvedSlurmConfig:
    layers: list[tuple[str, dict[str, Any]]] = [
        ("builtin", dict(_SLURM_SCALAR_BUILTINS))
    ]
    profile_defaults = dict(profile.run_slurm_defaults) if profile else {}
    profile_default_scalars = {
        key: profile_defaults.get(key)
        for key in _SLURM_SCALAR_FIELDS
        if profile_defaults.get(key) is not None
    }
    if profile_default_scalars:
        layers.append(("profile.defaults", profile_default_scalars))
    if profile:
        profile_scalars = {
            "partition": profile.partition,
            "time_min": profile.time_min,
            "cpus_per_task": profile.cpus_per_task,
            "mem_gb": profile.mem_gb,
            "gpus_per_node": profile.gpus_per_node,
            "job_name": profile.job_name,
            "mail_user": profile.mail_user,
            "mail_type": profile.mail_type,
        }
        profile_scalars = {
            key: value for key, value in profile_scalars.items() if value is not None
        }
        if profile_scalars:
            layers.append(("profile", profile_scalars))
    cli_scalars = {
        "partition": getattr(args, "partition", None),
        "time_min": getattr(args, "time_min", None),
        "cpus_per_task": getattr(args, "cpus_per_task", None),
        "mem_gb": getattr(args, "mem_gb", None),
        "gpus_per_node": getattr(args, "gpus_per_node", None),
        "job_name": getattr(args, "job_name", None),
        "mail_user": getattr(args, "mail_user", None),
        "mail_type": getattr(args, "mail_type", None),
        "query_status": getattr(args, "query_status", None),
    }
    cli_scalars = {
        key: value for key, value in cli_scalars.items() if value is not None
    }
    if cli_scalars:
        layers.append(("cli", cli_scalars))

    resolved_scalars = dict(_SLURM_SCALAR_BUILTINS)
    scalar_sources = {key: "builtin" for key in _SLURM_SCALAR_FIELDS}
    warnings: list[str] = []
    for source, values in layers[1:]:
        for key, value in values.items():
            if key not in _SLURM_SCALAR_FIELDS:
                continue
            previous_value = resolved_scalars.get(key)
            previous_source = scalar_sources.get(key, "builtin")
            if previous_source != source and previous_value != value:
                if previous_source != "builtin" or source == "cli":
                    warnings.append(
                        f"{key} overridden by {source} (was {previous_source})."
                    )
            resolved_scalars[key] = value
            scalar_sources[key] = source

    setup_layers: list[tuple[str, list[str]]] = []
    defaults_setup = list(profile_defaults.get("slurm_setup_cmds", []))
    if defaults_setup:
        setup_layers.append(("profile.defaults", defaults_setup))
    if profile and profile.slurm_setup_cmds:
        setup_layers.append(("profile", list(profile.slurm_setup_cmds)))
    cli_setup = list(getattr(args, "slurm_setup_cmd", None) or [])
    if cli_setup:
        setup_layers.append(("cli", cli_setup))
    slurm_setup_cmds, slurm_setup_sources, setup_warnings = _merge_slurm_setup_layers(
        setup_layers
    )
    warnings.extend(setup_warnings)

    sbatch_layers: list[tuple[str, dict[str, SlurmParameterValue]]] = []
    defaults_sbatch = dict(profile_defaults.get("sbatch_option", {}))
    if defaults_sbatch:
        sbatch_layers.append(("profile.defaults", defaults_sbatch))
    if profile and profile.slurm_additional_parameters:
        sbatch_layers.append(("profile", dict(profile.slurm_additional_parameters)))
    cli_sbatch = _parse_sbatch_option_args(getattr(args, "sbatch_option", None))
    if cli_sbatch:
        sbatch_layers.append(("cli", cli_sbatch))
    sbatch_opts, sbatch_sources, sbatch_warnings = _merge_sbatch_option_layers(
        sbatch_layers
    )
    warnings.extend(sbatch_warnings)

    partition_raw = str(resolved_scalars.get("partition") or "").strip()
    if not partition_raw:
        raise ConfigError(
            "Missing partition (set profiles.<name>.partition or profiles.<name>.defaults.run_slurm.partition)"
        )
    time_min_raw = resolved_scalars.get("time_min")
    if time_min_raw is None:
        raise ConfigError(
            "Missing time_min (set profiles.<name>.time_min or profiles.<name>.defaults.run_slurm.time_min)"
        )
    try:
        time_min = int(time_min_raw)
        cpus_per_task = int(resolved_scalars.get("cpus_per_task", 1))
        mem_gb = int(resolved_scalars.get("mem_gb", 16))
        gpus_per_node = int(resolved_scalars.get("gpus_per_node", 0))
    except (TypeError, ValueError) as exc:
        raise ConfigError("Slurm numeric settings must be integers.") from exc
    if time_min <= 0:
        raise ConfigError("time_min must be > 0")
    if cpus_per_task <= 0:
        raise ConfigError("cpus_per_task must be > 0")
    if mem_gb <= 0:
        raise ConfigError("mem_gb must be > 0")
    if gpus_per_node < 0:
        raise ConfigError("gpus_per_node must be >= 0")

    job_name = str(resolved_scalars.get("job_name") or "").strip() or "geryon"
    mail_user = (
        None
        if resolved_scalars.get("mail_user") is None
        else str(resolved_scalars.get("mail_user")).strip() or None
    )
    mail_type = (
        None
        if resolved_scalars.get("mail_type") is None
        else str(resolved_scalars.get("mail_type")).strip() or None
    )
    query_status = bool(resolved_scalars.get("query_status", False))

    return ResolvedSlurmConfig(
        partition=partition_raw,
        time_min=time_min,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        gpus_per_node=gpus_per_node,
        job_name=job_name,
        mail_user=mail_user,
        mail_type=mail_type,
        query_status=query_status,
        slurm_setup_cmds=tuple(slurm_setup_cmds),
        slurm_additional_parameters=sbatch_opts,
        sources=scalar_sources,
        sbatch_option_sources=sbatch_sources,
        slurm_setup_sources=slurm_setup_sources,
        warnings=tuple(warnings),
    )


def _prepare_run_execution_context(
    args: argparse.Namespace,
    *,
    command: str,
) -> _RunExecutionContext:
    store = ArtifactStore.from_run_dir(args.run)
    batches = select_batch_indices(store, getattr(args, "batch_index", None))
    profiles_path, profile = resolve_profile(
        profile_name=getattr(args, "profile", None),
        profiles_file=getattr(args, "profiles_file", None),
    )
    resolved_defaults_sources = _resolve_run_defaults(
        args,
        command=command,
        profile=profile,
    )
    cli_env_vars = _parse_env_args(getattr(args, "env", None))
    env_script, env_setup_cmds, env_vars = merge_profile_env(
        profile,
        cli_env_script=getattr(args, "env_script", None),
        cli_env_setup_cmds=list(getattr(args, "env_setup_cmd", None) or []),
        cli_env_vars=cli_env_vars,
    )
    env_script = _preflight_env_script(env_script)
    retry_on_status = _parse_retry_on_status(args.retry_on_status)
    fail_fast_threshold = _parse_fail_fast_threshold(args.fail_fast_threshold)
    selected_config_ids = _merge_selected_config_ids(
        getattr(args, "config_id", None),
        getattr(args, "retry_file", None),
        expected_run_root=store.run_root,
        expected_run_id=_resolve_run_id(store),
    )
    if getattr(args, "retry_file", None) and not bool(args.resume):
        args.resume = True
        resolved_defaults_sources["resume"] = "retry_file"
    _validate_selected_config_ids(store, selected_config_ids)
    local_concurrency = _resolve_local_concurrency_payload(
        max_concurrent_tasks=args.max_concurrent_tasks,
        cores_per_task=args.cores_per_task,
        max_total_cores=args.max_total_cores,
    )
    return _RunExecutionContext(
        store=store,
        batches=batches,
        profile=profile,
        profiles_path=profiles_path,
        env_script=env_script,
        env_setup_cmds=env_setup_cmds,
        env_vars=env_vars,
        retry_on_status=retry_on_status,
        fail_fast_threshold=fail_fast_threshold,
        selected_config_ids=selected_config_ids,
        local_concurrency=local_concurrency,
        resolved_defaults_sources=resolved_defaults_sources,
    )


def _render_status_table(
    summary: dict[str, object],
    *,
    grouped_by_packs: dict[str, Any] | None = None,
) -> None:
    console = _console()
    configs = _mapping(summary.get("configs"))
    attempts = _mapping(summary.get("attempts"))
    batches = _mapping(summary.get("batches"))

    overview = Table(title="Run Status", show_header=False)
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value", style="white")
    overview.add_row("Run ID", str(summary.get("run_id", "")))
    overview.add_row("Run Root", str(summary.get("run_root", "")))
    overview.add_row("Generated", str(summary.get("generated_at", "")))
    overview.add_row("Configs", str(configs.get("total", 0)))
    overview.add_row("Attempts", str(attempts.get("total", 0)))
    overview.add_row("Corrupt JSONL Lines", str(summary.get("corrupt_result_lines", 0)))
    overview.add_row(
        "Unfinished Batches",
        ", ".join(str(item) for item in batches.get("unfinished_batch_indices", []))
        or "none",
    )
    console.print(overview)

    status_table = Table(title="Config Summary", show_lines=False)
    status_table.add_column("Category", style="bold")
    status_table.add_column("Count", justify="right")
    latest_by_status = _mapping(configs.get("latest_by_status"))
    completion = _mapping(configs.get("completion"))
    status_table.add_row("Latest success", str(latest_by_status.get("success", 0)))
    status_table.add_row("Latest failed", str(latest_by_status.get("failed", 0)))
    status_table.add_row(
        "Latest terminated", str(latest_by_status.get("terminated", 0))
    )
    status_table.add_row("Latest missing", str(latest_by_status.get("missing", 0)))
    status_table.add_row("Resume complete", str(completion.get("complete", 0)))
    status_table.add_row("Resume pending", str(completion.get("pending", 0)))
    console.print(status_table)

    pending = Table(title="Unfinished Config IDs", show_lines=False)
    pending.add_column("Status", style="bold")
    pending.add_column("Count", justify="right")
    pending.add_row("failed", str(len(configs.get("failed_config_ids", []))))
    pending.add_row("terminated", str(len(configs.get("terminated_config_ids", []))))
    pending.add_row("missing", str(len(configs.get("missing_config_ids", []))))
    pending.add_row(
        "total unfinished", str(len(configs.get("unfinished_config_ids", [])))
    )
    console.print(pending)

    grouped = grouped_by_packs or {}
    packs = [str(pack) for pack in _list(grouped.get("packs"))]
    groups = _list(grouped.get("groups"))
    if packs:
        title = f"Grouped By: {', '.join(packs)}"
        by_pack = Table(title=title, show_lines=False)
        by_pack.add_column("Group", style="bold")
        by_pack.add_column("Total", justify="right")
        by_pack.add_column("Complete", justify="right")
        by_pack.add_column("Pending", justify="right")
        by_pack.add_column("Latest By Status")
        for item in groups:
            key = _mapping(item.get("key")) if isinstance(item, dict) else {}
            label = ", ".join(f"{pack}={key.get(pack, '<missing>')}" for pack in packs)
            by_pack.add_row(
                label,
                str(item.get("total", 0) if isinstance(item, dict) else 0),
                str(item.get("complete", 0) if isinstance(item, dict) else 0),
                str(item.get("pending", 0) if isinstance(item, dict) else 0),
                json.dumps(
                    item.get("latest_by_status", {}) if isinstance(item, dict) else {},
                    sort_keys=True,
                ),
            )
        if not groups:
            by_pack.add_row("<none>", "0", "0", "0", "{}")
        console.print(by_pack)


def _render_rerun_table(payload: dict[str, object]) -> None:
    console = _console()
    overview = Table(title="Retry Plan", show_header=False)
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value")
    overview.add_row("Run ID", str(payload.get("run_id", "")))
    overview.add_row("Status Filter", str(payload.get("status_filter", "")))
    overview.add_row("Selected Configs", str(payload.get("num_target_configs", 0)))
    overview.add_row("Selected Batches", str(payload.get("num_target_batches", 0)))
    overview.add_row("Retry File", str(payload.get("retry_file", "<dry-run>")))
    console.print(overview)

    targets = _list(payload.get("target_config_ids"))
    preview = Table(title="Target Config Preview")
    preview.add_column("Index", justify="right")
    preview.add_column("Config ID")
    for idx, config_id in enumerate(targets[:20], start=1):
        preview.add_row(str(idx), str(config_id))
    if len(targets) > 20:
        preview.add_row("...", f"{len(targets) - 20} more")
    console.print(preview)


def _render_profiles_table(
    profiles_file: Path, profiles: dict[str, RunProfile]
) -> None:
    console = _console()
    overview = Table(title="Profiles File", show_header=False)
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value")
    overview.add_row("Path", str(profiles_file))
    overview.add_row("Count", str(len(profiles)))
    console.print(overview)

    table = Table(title="Available Profiles")
    table.add_column("Name", style="bold")
    table.add_column("Partition")
    table.add_column("Time (min)", justify="right")
    table.add_column("CPUs", justify="right")
    table.add_column("Mem GB", justify="right")
    table.add_column("GPUs", justify="right")
    table.add_column("Mail User")
    table.add_column("Mail Type")
    table.add_column("Env Script")
    table.add_column("Env Vars", justify="right")
    table.add_column("Slurm Setup Cmds", justify="right")
    table.add_column("SBatch Opts", justify="right")
    for name in sorted(profiles.keys()):
        profile = profiles[name]
        table.add_row(
            name,
            profile.partition or "",
            "" if profile.time_min is None else str(profile.time_min),
            "" if profile.cpus_per_task is None else str(profile.cpus_per_task),
            "" if profile.mem_gb is None else str(profile.mem_gb),
            "" if profile.gpus_per_node is None else str(profile.gpus_per_node),
            profile.mail_user or "",
            profile.mail_type or "",
            profile.env_script or "",
            str(len(profile.env)),
            str(len(profile.slurm_setup_cmds)),
            str(len(profile.slurm_additional_parameters)),
        )
    console.print(table)


def _render_report_table(report: dict[str, Any]) -> None:
    summary = dict(report.get("summary", {}))
    grouped = dict(report.get("grouped_by_packs", {}))
    _render_status_table(summary, grouped_by_packs=grouped)


def _render_plan_table(payload: dict[str, Any]) -> None:
    console = _console()
    if bool(payload.get("all_run_sets")):
        items = list(payload.get("items", []))
        overview = Table(title="Plan Summary", show_header=False)
        overview.add_column("Field", style="bold cyan")
        overview.add_column("Value")
        overview.add_row("All Run-Sets", "true")
        overview.add_row("Run-Set Count", str(payload.get("num_run_sets", 0)))
        overview.add_row("Dry Run", str(payload.get("dry_run", False)))
        console.print(overview)

        table = Table(title="Planned Run-Sets")
        table.add_column("Run-Set", style="bold")
        table.add_column("Run ID")
        table.add_column("Configs", justify="right")
        table.add_column("Batches", justify="right")
        table.add_column("Run Root")
        for item in items:
            row = dict(item)
            table.add_row(
                str(row.get("run_set", "")),
                str(row.get("run_id", "")),
                str(row.get("total_configs", 0)),
                str(row.get("total_batches", 0)),
                str(row.get("run_root", "")),
            )
        if not items:
            table.add_row("<none>", "", "0", "0", "")
        console.print(table)
        return

    overview = Table(title="Plan Summary", show_header=False)
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value")
    overview.add_row("Run ID", str(payload.get("run_id", "")))
    overview.add_row("Run Set", str(payload.get("run_set", "") or "<default>"))
    overview.add_row("Run Root", str(payload.get("run_root", "")))
    overview.add_row("Configs", str(payload.get("total_configs", 0)))
    overview.add_row("Batches", str(payload.get("total_batches", 0)))
    overview.add_row("Dry Run", str(payload.get("dry_run", False)))
    console.print(overview)

    preview_configs = [dict(item) for item in _list(payload.get("preview_configs"))]
    if preview_configs:
        preview_table = Table(title="Preview Configs")
        preview_table.add_column("Config ID", style="bold")
        preview_table.add_column("Batch:Line", justify="right")
        preview_table.add_column("Selected Options")
        preview_table.add_column("Command")
        for cfg in preview_configs:
            selected = _mapping(cfg.get("selected_options"))
            selected_text = ", ".join(
                f"{key}={value}" for key, value in sorted(selected.items())
            )
            preview_table.add_row(
                _short_config_id(str(cfg.get("config_id", "")), width=12),
                f"{cfg.get('batch_index', 0)}:{cfg.get('line_index', 0)}",
                _short_text(selected_text, width=80),
                _short_text(cfg.get("command", ""), width=120),
            )
        console.print(preview_table)


def _render_run_local_table(payload: dict[str, Any]) -> None:
    console = _console()
    summaries = [dict(item) for item in list(payload.get("payloads", []))]

    executors = sorted(
        {
            str(item.get("executor", "")).strip()
            for item in summaries
            if str(item.get("executor", "")).strip()
        }
    )
    totals = {
        "total": sum(_int_value(item.get("total")) for item in summaries),
        "success": sum(_int_value(item.get("success")) for item in summaries),
        "failed": sum(_int_value(item.get("failed")) for item in summaries),
        "terminated": sum(_int_value(item.get("terminated")) for item in summaries),
        "retry_scheduled": sum(
            _int_value(item.get("retry_scheduled")) for item in summaries
        ),
        "skipped_resume": sum(
            _int_value(item.get("skipped_resume")) for item in summaries
        ),
        "skipped_not_selected": sum(
            _int_value(item.get("skipped_not_selected")) for item in summaries
        ),
        "skipped_fail_fast": sum(
            _int_value(item.get("skipped_fail_fast")) for item in summaries
        ),
        "fail_fast_tasks": sum(
            1 if bool(item.get("fail_fast_triggered", False)) else 0
            for item in summaries
        ),
    }
    first = summaries[0] if summaries else {}
    first_concurrency = _mapping(first.get("concurrency"))
    max_concurrent_tasks = _int_value(first_concurrency.get("max_concurrent_tasks"))
    if max_concurrent_tasks <= 0:
        max_concurrent_tasks = 1
    cores_per_task = _int_value(first_concurrency.get("cores_per_task"))
    if cores_per_task <= 0:
        cores_per_task = 1
    max_total_cores: int | None = None
    raw_max_total = first_concurrency.get("max_total_cores")
    if raw_max_total is not None:
        parsed_total = _int_value(raw_max_total)
        if parsed_total > 0:
            max_total_cores = parsed_total

    overview = Table(title="Local Execution", show_header=False, box=box.SIMPLE)
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value")
    overview.add_row("Run ID", str(first.get("run_id", "")))
    overview.add_row("Executor", ", ".join(executors) if executors else "<none>")
    overview.add_row("Payloads", str(len(summaries)))
    overview.add_row("Dry Run", str(bool(first.get("dry_run", False))))
    overview.add_row("Resume", str(bool(first.get("resume", False))))
    overview.add_row(
        "Concurrency",
        _format_concurrency(
            max_concurrent_tasks=max_concurrent_tasks,
            cores_per_task=cores_per_task,
            max_total_cores=max_total_cores,
        ),
    )
    console.print(overview)

    totals_table = Table(title="Attempt Totals", box=box.SIMPLE)
    totals_table.add_column("Metric", style="bold")
    totals_table.add_column("Count", justify="right")
    totals_table.add_row("Total", str(totals["total"]))
    totals_table.add_row("Success", str(totals["success"]))
    totals_table.add_row("Failed", str(totals["failed"]))
    totals_table.add_row("Terminated", str(totals["terminated"]))
    totals_table.add_row("Retries Scheduled", str(totals["retry_scheduled"]))
    totals_table.add_row("Skipped (Resume)", str(totals["skipped_resume"]))
    totals_table.add_row("Skipped (Not Selected)", str(totals["skipped_not_selected"]))
    totals_table.add_row("Skipped (Fail-Fast)", str(totals["skipped_fail_fast"]))
    totals_table.add_row("Fail-Fast Triggered Tasks", str(totals["fail_fast_tasks"]))
    console.print(totals_table)

    table = Table(title="Payload Results", box=box.SIMPLE)
    table.add_column("Task")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for item in summaries:
        task_id = f"{item.get('job_id', '')}/{item.get('array_task_id', '')}"
        rows = [
            (
                "Batches",
                ",".join(str(batch) for batch in list(item.get("batches", []))) or "-",
            ),
            ("Total", str(_int_value(item.get("total")))),
            ("Success", str(_int_value(item.get("success")))),
            ("Failed", str(_int_value(item.get("failed")))),
            ("Terminated", str(_int_value(item.get("terminated")))),
            ("Retry", str(_int_value(item.get("retry_scheduled")))),
            ("Skip Resume", str(_int_value(item.get("skipped_resume")))),
            ("Skip Select", str(_int_value(item.get("skipped_not_selected")))),
            ("Skip Fail-Fast", str(_int_value(item.get("skipped_fail_fast")))),
        ]
        for row_index, (field, value) in enumerate(rows):
            table.add_row(task_id if row_index == 0 else "", field, value)
    if not summaries:
        table.add_row("<none>", "-", "-")
    console.print(table)

    artifact_table = Table(title="Artifacts", box=box.SIMPLE)
    artifact_table.add_column("Task")
    artifact_table.add_column("Field", style="bold")
    artifact_table.add_column("Path")
    for item in summaries:
        task_id = f"{item.get('job_id', '')}/{item.get('array_task_id', '')}"
        rows = [
            ("Result", str(item.get("result_path", ""))),
            ("Events", str(item.get("task_events_path", ""))),
            ("Stdout", str(item.get("task_stdout_log", ""))),
            ("Stderr", str(item.get("task_stderr_log", ""))),
        ]
        for row_index, (field, value) in enumerate(rows):
            artifact_table.add_row(task_id if row_index == 0 else "", field, value)
    if not summaries:
        artifact_table.add_row("<none>", "-", "")
    console.print(artifact_table)


def _render_queue_status_table(payload: dict[str, Any]) -> None:
    console = _console()
    query = Table(title="Queue Query", show_header=False, box=box.SIMPLE)
    query.add_column("Field", style="bold cyan")
    query.add_column("Value")
    query.add_row("Run ID", str(payload.get("run_id", "")))
    query.add_row("Submission ID", str(payload.get("submission_id", "") or "<none>"))
    query.add_row("Source", str(payload.get("source", "")))
    query.add_row("Queried", str(payload.get("queried", False)))
    query.add_row("Available", str(payload.get("available", False)))
    query.add_row("Rows", str(payload.get("num_rows", 0)))
    query.add_row("Error", str(payload.get("error", "") or "<none>"))
    query.add_row("Refreshed At", str(payload.get("refreshed_at", "")))
    console.print(query)

    rows = _list(payload.get("rows"))
    queue_table = Table(title="Queue Status", box=box.SIMPLE)
    queue_table.add_column("Job ID")
    queue_table.add_column("State", style="bold")
    queue_table.add_column("Time")
    queue_table.add_column("Nodes", justify="right")
    queue_table.add_column("Partition")
    queue_table.add_column("Reason")
    for item in rows:
        row = _mapping(item)
        queue_table.add_row(
            str(row.get("job_id", "")),
            str(row.get("state", "")),
            str(row.get("time", "")),
            str(row.get("nodes", "")),
            str(row.get("partition", "")),
            str(row.get("reason", "")),
        )
    if not rows:
        queue_table.add_row("<none>", "", "", "", "", "")
    console.print(queue_table)

    by_state = _mapping(payload.get("by_state"))
    state_table = Table(title="Queue State Counts", box=box.SIMPLE)
    state_table.add_column("State", style="bold")
    state_table.add_column("Count", justify="right")
    for state in sorted(by_state.keys()):
        state_table.add_row(str(state), str(by_state[state]))
    if not by_state:
        state_table.add_row("<none>", "0")
    console.print(state_table)


def _render_run_slurm_table(payload: dict[str, Any]) -> None:
    console = _console()
    overview = Table(title="Slurm Submission", show_header=False, box=box.SIMPLE)
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value")
    overview.add_row("Submitted", str(payload.get("submitted", False)))
    overview.add_row("Dry Run", str(payload.get("dry_run", False)))
    overview.add_row("Payloads", str(payload.get("num_payloads", 0)))
    overview.add_row("Partition", str(payload.get("partition", "")))
    overview.add_row("Time (min)", str(payload.get("time_min", "")))
    overview.add_row("CPUs", str(payload.get("cpus_per_task", "")))
    overview.add_row("Mem GB", str(payload.get("mem_gb", "")))
    overview.add_row("GPUs", str(payload.get("gpus_per_node", "")))
    overview.add_row("Mail User", str(payload.get("mail_user", "")))
    overview.add_row("Mail Type", str(payload.get("mail_type", "")))
    overview.add_row("Job Name", str(payload.get("job_name", "")))
    local_concurrency = _mapping(payload.get("local_concurrency"))
    if local_concurrency:
        overview.add_row(
            "Task Concurrency",
            _format_concurrency(
                max_concurrent_tasks=max(
                    1, _int_value(local_concurrency.get("max_concurrent_tasks"))
                ),
                cores_per_task=max(
                    1, _int_value(local_concurrency.get("cores_per_task"))
                ),
                max_total_cores=(
                    None
                    if local_concurrency.get("max_total_cores") is None
                    else _int_value(local_concurrency.get("max_total_cores"))
                ),
            ),
        )
    overview.add_row(
        "SBatch Opts",
        str(len(dict(payload.get("slurm_additional_parameters", {})))),
    )
    overview.add_row("Profile", str(payload.get("profile_name", "")))
    overview.add_row("Profiles File", str(payload.get("profiles_file", "")))
    overview.add_row("Payloads File", str(payload.get("payloads_path", "")))
    overview.add_row("Payload Snapshot", str(payload.get("payloads_snapshot_path", "")))
    overview.add_row("Submission ID", str(payload.get("submission_id", "")))
    overview.add_row("Submitted At", str(payload.get("submitted_at", "")))
    console.print(overview)

    effective = _mapping(payload.get("effective_config"))
    if effective:
        effective_table = Table(title="Effective Config", box=box.SIMPLE)
        effective_table.add_column("Field", style="bold")
        effective_table.add_column("Value")
        for field in (
            "partition",
            "time_min",
            "cpus_per_task",
            "mem_gb",
            "gpus_per_node",
            "job_name",
            "mail_user",
            "mail_type",
            "query_status",
        ):
            effective_table.add_row(field, str(effective.get(field, "")))
        effective_table.add_row(
            "slurm_setup_cmds", str(len(_list(effective.get("slurm_setup_cmds"))))
        )
        effective_table.add_row(
            "sbatch_option",
            str(len(_mapping(effective.get("slurm_additional_parameters")))),
        )
        console.print(effective_table)

    source_of_truth = _mapping(payload.get("source_of_truth"))
    if source_of_truth:
        src_table = Table(title="Source Of Truth", box=box.SIMPLE)
        src_table.add_column("Setting", style="bold")
        src_table.add_column("Source")
        scalar_sources = _mapping(source_of_truth.get("scalars"))
        for key in sorted(scalar_sources.keys()):
            src_table.add_row(str(key), str(scalar_sources[key]))
        sbatch_sources = _mapping(source_of_truth.get("sbatch_option"))
        for key in sorted(sbatch_sources.keys()):
            src_table.add_row(f"sbatch.{key}", str(sbatch_sources[key]))
        setup_sources = _mapping(source_of_truth.get("slurm_setup_cmds"))
        for cmd, source in setup_sources.items():
            src_table.add_row(f"setup:{cmd}", str(source))
        if not scalar_sources and not sbatch_sources and not setup_sources:
            src_table.add_row("<none>", "<none>")
        console.print(src_table)

    warnings = [str(item) for item in _list(payload.get("resolution_warnings"))]
    if warnings:
        warn_table = Table(title="Resolution Warnings", box=box.SIMPLE)
        warn_table.add_column("Warning")
        for warning in warnings:
            warn_table.add_row(warning)
        console.print(warn_table)

    job_ids = list(payload.get("job_ids", []))
    jobs = Table(title="Job IDs", box=box.SIMPLE)
    jobs.add_column("Index", justify="right")
    jobs.add_column("Job ID")
    for idx, job_id in enumerate(job_ids, start=1):
        jobs.add_row(str(idx), str(job_id))
    if not job_ids:
        jobs.add_row("1", "<none>")
    console.print(jobs)

    slurm_setup = list(payload.get("slurm_setup", []))
    setup = Table(title="Slurm Setup Commands", box=box.SIMPLE)
    setup.add_column("Index", justify="right")
    setup.add_column("Command")
    for idx, cmd in enumerate(slurm_setup, start=1):
        setup.add_row(str(idx), str(cmd))
    if not slurm_setup:
        setup.add_row("1", "<none>")
    console.print(setup)

    metadata_path = str(payload.get("submitit_jobs_path", ""))
    metadata_snapshot = str(payload.get("submitit_jobs_snapshot_path", ""))
    artifacts = Table(title="Submission Artifacts", box=box.SIMPLE)
    artifacts.add_column("Field", style="bold")
    artifacts.add_column("Value")
    artifacts.add_row("Latest Metadata", metadata_path or "<none>")
    artifacts.add_row("Metadata Snapshot", metadata_snapshot or "<none>")
    artifacts.add_row("Launcher Log", str(payload.get("launcher_log_path", "")))
    console.print(artifacts)

    submitit_states = list(payload.get("submitit_states", []))
    if submitit_states:
        submitit_table = Table(
            title="Submitit Job States (Submission Snapshot)", box=box.SIMPLE
        )
        submitit_table.add_column("Job ID")
        submitit_table.add_column("State")
        for entry in submitit_states:
            row = dict(entry) if isinstance(entry, dict) else {}
            submitit_table.add_row(
                str(row.get("job_id", "")), str(row.get("state", ""))
            )
        console.print(submitit_table)

    queue_status = _mapping(payload.get("queue_status"))
    if queue_status:
        _render_queue_status_table(queue_status)

    if job_ids:
        monitor_cmd = f"squeue -j {','.join(str(job_id) for job_id in job_ids)}"
        console.print(f"[bold]Monitor:[/bold] {monitor_cmd}")
    elif bool(payload.get("dry_run", False)):
        console.print("[bold]Next:[/bold] rerun without --dry-run to submit jobs.")


def _render_collect_table(summary: dict[str, Any]) -> None:
    console = _console()
    attempts = dict(summary.get("attempts", {}))
    configs = dict(summary.get("configs", {}))
    batches = dict(summary.get("batches", {}))

    overview = Table(title="Collected Summary", show_header=False)
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value")
    overview.add_row("Run ID", str(summary.get("run_id", "")))
    overview.add_row("Run Root", str(summary.get("run_root", "")))
    overview.add_row("Result Files", str(summary.get("num_result_files", 0)))
    overview.add_row("Attempts", str(attempts.get("total", 0)))
    overview.add_row("Corrupt JSONL Lines", str(summary.get("corrupt_result_lines", 0)))
    overview.add_row("Configs", str(configs.get("total", 0)))
    overview.add_row(
        "Batches", str(batches.get("total", len(dict(batches.get("by_batch", {})))))
    )
    console.print(overview)

    attempt_table = Table(title="Attempt Status Counts")
    attempt_table.add_column("Status", style="bold")
    attempt_table.add_column("Count", justify="right")
    for key, value in sorted(dict(attempts.get("by_status", {})).items()):
        attempt_table.add_row(str(key), str(value))
    if not dict(attempts.get("by_status", {})):
        attempt_table.add_row("<none>", "0")
    console.print(attempt_table)

    final_table = Table(title="Final Config Status Counts")
    final_table.add_column("Status", style="bold")
    final_table.add_column("Count", justify="right")
    for key, value in sorted(dict(configs.get("final_by_status", {})).items()):
        final_table.add_row(str(key), str(value))
    if not dict(configs.get("final_by_status", {})):
        final_table.add_row("<none>", "0")
    console.print(final_table)


def _load_submitit_submission_metadata(store: ArtifactStore) -> dict[str, Any]:
    if not store.submitit_jobs_path.exists():
        raise ConfigError(
            f"Submission metadata not found: {store.submitit_jobs_path}. "
            "Run `run-slurm` first."
        )
    try:
        raw = json.loads(store.submitit_jobs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid submission metadata JSON: {store.submitit_jobs_path}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Submission metadata must be a JSON object: {store.submitit_jobs_path}"
        )
    return {str(key): value for key, value in raw.items()}


def _resolve_queue_job_ids(
    *,
    store: ArtifactStore,
    explicit_job_ids: list[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    cleaned = [str(job_id).strip() for job_id in (explicit_job_ids or [])]
    cleaned = [job_id for job_id in cleaned if job_id]
    if cleaned:
        return list(dict.fromkeys(cleaned)), {}

    metadata = _load_submitit_submission_metadata(store)
    metadata_job_ids = [
        str(job_id).strip()
        for job_id in _list(metadata.get("job_ids"))
        if str(job_id).strip()
    ]
    return list(dict.fromkeys(metadata_job_ids)), metadata


def _refresh_queue_status(
    *,
    store: ArtifactStore,
    job_ids: list[str],
    metadata: dict[str, Any],
    persist: bool,
) -> dict[str, Any]:
    queue_status = _query_slurm_queue_status(job_ids)
    payload = {
        **queue_status,
        "run_id": _resolve_run_id(store),
        "run_root": str(store.run_root),
        "submission_id": metadata.get("submission_id"),
        "submitted_at": metadata.get("submitted_at"),
        "refreshed_at": utc_now_iso(),
    }
    if persist:
        store.ensure_exec_layout()
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        store.queue_status_path.write_text(text, encoding="utf-8")
        snapshot_name = sanitize_for_path(
            payload["refreshed_at"].replace(":", "").replace(".", "")
        )
        snapshot_path = (
            store.queue_status_snapshots_dir / f"queue_status_{snapshot_name}.json"
        )
        snapshot_path.write_text(text, encoding="utf-8")
        payload["queue_status_path"] = str(store.queue_status_path)
        payload["queue_status_snapshot_path"] = str(snapshot_path)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geryon", description="Geryon experiment launcher"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_run_selection_args(
        target: argparse._ArgumentGroup | argparse.ArgumentParser,
        *,
        include_retry_file: bool = True,
    ) -> None:
        target.add_argument(
            "--batch-index",
            action="append",
            type=int,
            default=None,
            help="Run only selected batch indices (repeatable)",
        )
        target.add_argument(
            "--config-id",
            action="append",
            default=None,
            help="Run only selected config IDs (repeatable)",
        )
        if include_retry_file:
            target.add_argument(
                "--retry-file",
                default=None,
                help="Retry metadata from `geryon rerun`",
            )

    def _add_profile_args(
        target: argparse._ArgumentGroup | argparse.ArgumentParser,
    ) -> None:
        target.add_argument("--profile", default=None, help="Profile name")
        target.add_argument(
            "--profiles-file",
            default=None,
            help="Profiles YAML path (default: ./profiles.yaml)",
        )

    plan = sub.add_parser(
        "plan", aliases=["build"], help="Plan configs and batch files"
    )
    plan.add_argument("--experiment", required=True, help="Experiment YAML")
    plan.add_argument("--out", required=True, help="Output directory root")
    plan.add_argument("--batch-size", required=True, type=int)
    plan.add_argument("--run-id", default=None)
    run_set_group = plan.add_mutually_exclusive_group()
    run_set_group.add_argument(
        "--run-set", default=None, help="Run-set name to materialize before planning"
    )
    run_set_group.add_argument(
        "--all-run-sets",
        action="store_true",
        help="Plan all run-sets defined in experiment.run_sets",
    )
    plan.add_argument("--dry-run", action="store_true")
    plan.add_argument(
        "--preview-configs",
        type=int,
        default=0,
        help="Include first N planned configs in output (works with --dry-run).",
    )
    plan.add_argument("--format", choices=["json", "table"], default="table")
    plan.set_defaults(handler=_cmd_plan)

    validate = sub.add_parser(
        "validate-config", help="Validate experiment schema and composed config"
    )
    validate.add_argument("--experiment", required=True, help="Experiment YAML")
    validate.add_argument(
        "--run-set", default=None, help="Optional run-set name to validate"
    )
    validate.add_argument(
        "--show-diagnostics",
        action="store_true",
        help="Include planner diagnostics in output",
    )
    validate.add_argument("--format", choices=["json", "table"], default="table")
    validate.set_defaults(handler=_cmd_validate_config)

    inspect = sub.add_parser(
        "inspect-config",
        help="Render composed config after imports/defs/select expansion",
    )
    inspect.add_argument("--experiment", required=True, help="Experiment YAML")
    inspect.add_argument(
        "--run-set", default=None, help="Optional run-set name to inspect"
    )
    inspect.add_argument("--format", choices=["yaml", "json"], default="yaml")
    inspect.add_argument(
        "--show-diagnostics",
        action="store_true",
        help="Include composition diagnostics in output",
    )
    inspect.add_argument(
        "--out",
        default=None,
        help="Optional file path for rendered output (prints to stdout when omitted)",
    )
    inspect.set_defaults(handler=_cmd_inspect_config)

    show_config = sub.add_parser(
        "show-config",
        help="Show a planned config by full or prefix config hash from a run directory",
    )
    show_config.add_argument("--run", required=True, help="Run directory path")
    show_config.add_argument("config_id", help="Full config ID or unique prefix")
    show_config.add_argument(
        "--format", choices=["table", "json", "yaml"], default="table"
    )
    show_config.set_defaults(handler=_cmd_show_config)

    run_local = sub.add_parser(
        "run-local",
        aliases=["local"],
        help="Execute planned work locally (profile-configured)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  geryon run-local --run ./outputs/runs/demo --profile local_fast\n"
            "  geryon run-slurm --run ./outputs/runs/demo --profile a100_short\n"
            "  geryon recover --run ./outputs/runs/demo --backend local\n"
        ),
    )
    run_local.add_argument("--run", required=True, help="Run directory path")
    local_selection = run_local.add_argument_group("Selection")
    _add_run_selection_args(local_selection)
    local_profile = run_local.add_argument_group("Profile")
    _add_profile_args(local_profile)
    local_output = run_local.add_argument_group("Output")
    local_output.add_argument("--dry-run", action="store_true")
    local_output.add_argument("--format", choices=["json", "table"], default="table")
    run_local.set_defaults(handler=_cmd_run_local)

    run_slurm = sub.add_parser(
        "run-slurm",
        aliases=["slurm"],
        help="Submit planned work to Slurm (profile-configured)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  geryon run-slurm --run ./outputs/runs/demo --profile a100_short\n"
            "  geryon recover --run ./outputs/runs/demo --backend slurm\n"
        ),
    )
    run_slurm.add_argument("--run", required=True, help="Run directory path")
    slurm_selection = run_slurm.add_argument_group("Selection")
    _add_run_selection_args(slurm_selection)
    slurm_profile = run_slurm.add_argument_group("Profile")
    _add_profile_args(slurm_profile)
    slurm_runtime = run_slurm.add_argument_group("Slurm")
    slurm_runtime.add_argument("--partition", default=None, help="SLURM partition")
    slurm_runtime.add_argument(
        "--time-min", type=int, default=None, help="SLURM time limit in minutes"
    )
    slurm_runtime.add_argument(
        "--cpus-per-task", type=int, default=None, help="CPUs per task"
    )
    slurm_runtime.add_argument(
        "--mem-gb", type=int, default=None, help="Memory per task in GB"
    )
    slurm_runtime.add_argument(
        "--gpus-per-node", type=int, default=None, help="GPUs per node"
    )
    slurm_runtime.add_argument("--job-name", default=None, help="SLURM job name")
    slurm_runtime.add_argument("--mail-user", default=None, help="SLURM mail user")
    slurm_runtime.add_argument("--mail-type", default=None, help="SLURM mail type")
    query_toggle = slurm_runtime.add_mutually_exclusive_group()
    query_toggle.add_argument(
        "--query-status",
        dest="query_status",
        action="store_true",
        default=None,
        help="Query squeue after submission",
    )
    query_toggle.add_argument(
        "--no-query-status",
        dest="query_status",
        action="store_false",
        default=None,
        help="Skip queue query even if profile enables it",
    )
    slurm_runtime.add_argument(
        "--slurm-setup-cmd",
        action="append",
        default=None,
        help="Command to run in SLURM setup context (repeatable)",
    )
    slurm_runtime.add_argument(
        "--sbatch-option",
        action="append",
        default=None,
        help="Additional sbatch option (KEY=VALUE or KEY for boolean true, repeatable)",
    )
    slurm_output = run_slurm.add_argument_group("Output")
    slurm_output.add_argument("--dry-run", action="store_true")
    slurm_output.add_argument("--format", choices=["json", "table"], default="table")
    run_slurm.set_defaults(handler=_cmd_run_slurm)

    launch = sub.add_parser(
        "launch",
        help="Validate + plan + execute in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    launch.add_argument("--experiment", required=True, help="Experiment YAML")
    launch.add_argument("--out", required=True, help="Output directory root")
    launch.add_argument("--batch-size", required=True, type=int)
    launch.add_argument("--run-id", default=None)
    launch.add_argument("--run-set", default=None)
    launch.add_argument(
        "--backend",
        choices=["local", "slurm"],
        default="local",
        help="Execution backend",
    )
    launch.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validate-config stage",
    )
    launch_selection = launch.add_argument_group("Selection")
    launch_selection.add_argument(
        "--batch-index",
        action="append",
        type=int,
        default=None,
        help="Run only selected batch indices (repeatable)",
    )
    launch_selection.add_argument(
        "--config-id",
        action="append",
        default=None,
        help="Run only selected config IDs (repeatable)",
    )
    launch_profile = launch.add_argument_group("Profile")
    _add_profile_args(launch_profile)
    launch_output = launch.add_argument_group("Output")
    launch_output.add_argument("--dry-run", action="store_true")
    launch_output.add_argument("--format", choices=["table", "json"], default="table")
    launch.set_defaults(handler=_cmd_launch)

    recover = sub.add_parser(
        "recover",
        help="Retry failed/terminated/missing configs and execute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    recover.add_argument("--run", required=True, help="Run directory path")
    recover.add_argument(
        "--status",
        choices=sorted(RERUN_STATUS_FILTERS),
        default="failed",
        help="Status class to recover",
    )
    recover.add_argument(
        "--backend",
        choices=["local", "slurm"],
        default="local",
        help="Execution backend",
    )
    recover_selection = recover.add_argument_group("Selection")
    _add_run_selection_args(
        recover_selection,
        include_retry_file=False,
    )
    recover_profile = recover.add_argument_group("Profile")
    _add_profile_args(recover_profile)
    recover_output = recover.add_argument_group("Output")
    recover_output.add_argument("--dry-run", action="store_true")
    recover_output.add_argument("--format", choices=["table", "json"], default="table")
    recover.set_defaults(handler=_cmd_recover)

    list_profiles = sub.add_parser(
        "list-profiles", help="Show profiles available for run-local/run-slurm"
    )
    list_profiles.add_argument(
        "--profiles-file",
        default=None,
        help="Profiles YAML path (default: ./profiles.yaml)",
    )
    list_profiles.add_argument("--format", choices=["table", "json"], default="table")
    list_profiles.set_defaults(handler=_cmd_list_profiles)

    queue = sub.add_parser(
        "queue", help="Query live SLURM queue status for a submitted run"
    )
    queue.add_argument("--run", required=True, help="Run directory path")
    queue.add_argument(
        "--job-id",
        action="append",
        default=None,
        help="Explicit SLURM job id (repeatable). Defaults to latest submission metadata.",
    )
    queue.add_argument("--format", choices=["table", "json"], default="table")
    queue.set_defaults(handler=_cmd_queue)

    queue_refresh = sub.add_parser(
        "queue-refresh",
        help="Query and persist SLURM queue status snapshot for a run",
    )
    queue_refresh.add_argument("--run", required=True, help="Run directory path")
    queue_refresh.add_argument(
        "--job-id",
        action="append",
        default=None,
        help="Explicit SLURM job id (repeatable). Defaults to latest submission metadata.",
    )
    queue_refresh.add_argument("--format", choices=["table", "json"], default="table")
    queue_refresh.set_defaults(handler=_cmd_queue_refresh)

    status = sub.add_parser(
        "status", help="Summarize current run progress from plan + result files"
    )
    status.add_argument("--run", required=True)
    status.add_argument(
        "--by-pack",
        action="append",
        default=[],
        help="Group status by pack(s), repeatable or comma-separated",
    )
    status.add_argument(
        "--strict-jsonl",
        action="store_true",
        help="Fail if corrupt JSONL lines are detected in result files",
    )
    status.add_argument("--format", choices=["table", "json"], default="table")
    status.set_defaults(handler=_cmd_status)

    report = sub.add_parser(
        "report", help="Generate a run report from plan + result files"
    )
    report.add_argument("--run", required=True)
    report.add_argument(
        "--by-pack",
        action="append",
        default=[],
        help="Group report by pack(s), repeatable or comma-separated",
    )
    report.add_argument(
        "--strict-jsonl",
        action="store_true",
        help="Fail if corrupt JSONL lines are detected in result files",
    )
    report.add_argument(
        "--format", choices=["table", "markdown", "json"], default="table"
    )
    report.add_argument(
        "--out", default=None, help="Optional output file for markdown/json report"
    )
    report.set_defaults(handler=_cmd_report)

    rerun = sub.add_parser(
        "rerun", help="Create retry metadata for failed/terminated/missing configs"
    )
    rerun.add_argument("--run", required=True)
    rerun.add_argument(
        "--status", choices=sorted(RERUN_STATUS_FILTERS), default="failed"
    )
    rerun.add_argument(
        "--config-id",
        action="append",
        default=None,
        help="Always include selected config IDs (repeatable)",
    )
    rerun.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
    )
    rerun.add_argument("--dry-run", action="store_true")
    rerun.set_defaults(handler=_cmd_rerun)

    collect = sub.add_parser("collect", help="Merge task result files into summary")
    collect.add_argument("--run", required=True)
    collect.add_argument(
        "--strict-jsonl",
        action="store_true",
        help="Fail if corrupt JSONL lines are detected in result files",
    )
    collect.add_argument("--dry-run", action="store_true")
    collect.add_argument("--format", choices=["json", "table"], default="table")
    collect.set_defaults(handler=_cmd_collect)

    clean = sub.add_parser(
        "clean", help="Clean run artifacts (requires explicit scope)"
    )
    clean.add_argument("--run", required=True)
    clean.add_argument(
        "--all", action="store_true", help="Delete the entire run directory"
    )
    clean.add_argument("--plan", action="store_true", help="Delete plan directory")
    clean.add_argument("--exec", action="store_true", help="Delete exec directory")
    clean.set_defaults(handler=_cmd_clean)

    return parser


_cli_log = logging.getLogger("geryon.cli")


def _maybe_print_alias_notice(command: str) -> None:
    mapping = {
        "build": "plan",
        "local": "run-local",
        "slurm": "run-slurm",
    }
    if command in mapping:
        _cli_log.warning(
            "'%s' is a deprecated alias for '%s'.", command, mapping[command]
        )


def _cmd_plan(args: argparse.Namespace) -> int:
    if args.all_run_sets:
        run_sets = get_experiment_run_sets(args.experiment)
        if not run_sets:
            raise ConfigError(
                "No run_sets defined in experiment. "
                "Use --run-set for a named run_set or remove --all-run-sets."
            )

        items: list[dict[str, object]] = []
        for run_set in run_sets:
            derived_run_id = args.run_id
            if derived_run_id:
                derived_run_id = f"{derived_run_id}_{sanitize_for_path(run_set)}"

            summary = plan_experiment(
                experiment_path=args.experiment,
                out_dir=args.out,
                batch_size=args.batch_size,
                run_id=derived_run_id,
                run_set=run_set,
                dry_run=args.dry_run,
                preview_count=args.preview_configs,
            )
            items.append(
                {
                    "run_set": run_set,
                    "run_id": summary.run_id,
                    "run_root": str(summary.run_root),
                    "total_configs": summary.total_configs,
                    "total_batches": summary.total_batches,
                    "diagnostics_path": (
                        str(summary.run_root / "plan" / "diagnostics.json")
                        if not args.dry_run
                        else None
                    ),
                    "diagnostics_summary_path": (
                        str(summary.run_root / "plan" / "diagnostics.summary.txt")
                        if not args.dry_run
                        else None
                    ),
                    "run_set_path": (
                        str(summary.run_root / "plan" / "run_set.json")
                        if not args.dry_run
                        else None
                    ),
                    "preview_configs": list(summary.preview_configs),
                }
            )

        payload = {
            "all_run_sets": True,
            "num_run_sets": len(items),
            "items": items,
            "dry_run": args.dry_run,
        }
        if args.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _render_plan_table(payload)
        return 0

    summary = plan_experiment(
        experiment_path=args.experiment,
        out_dir=args.out,
        batch_size=args.batch_size,
        run_id=args.run_id,
        run_set=args.run_set,
        dry_run=args.dry_run,
        preview_count=args.preview_configs,
    )
    payload = {
        "run_set": args.run_set,
        "run_id": summary.run_id,
        "run_root": str(summary.run_root),
        "total_configs": summary.total_configs,
        "total_batches": summary.total_batches,
        "diagnostics_path": (
            str(summary.run_root / "plan" / "diagnostics.json")
            if not args.dry_run
            else None
        ),
        "diagnostics_summary_path": (
            str(summary.run_root / "plan" / "diagnostics.summary.txt")
            if not args.dry_run
            else None
        ),
        "run_set_path": (
            str(summary.run_root / "plan" / "run_set.json")
            if not args.dry_run
            else None
        ),
        "dry_run": args.dry_run,
        "preview_configs": list(summary.preview_configs),
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_plan_table(payload)
    return 0


def _cmd_inspect_config(args: argparse.Namespace) -> int:
    composed, diagnostics = compose_experiment_data(
        args.experiment, run_set=args.run_set
    )
    payload: dict[str, object]
    if args.show_diagnostics:
        payload = {"experiment": composed, "diagnostics": diagnostics}
    else:
        payload = composed

    if args.format == "json":
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    else:
        text = yaml.safe_dump(payload, sort_keys=False)

    if args.out:
        output_path = Path(args.out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(json.dumps({"output_path": str(output_path)}, indent=2, sort_keys=True))
    else:
        print(text, end="")
    return 0


def _cmd_show_config(args: argparse.Namespace) -> int:
    store = ArtifactStore.from_run_dir(args.run)
    planned = store.read_planned_configs()
    query = str(args.config_id).strip()
    if not query:
        raise ConfigError("config_id must be a non-empty string")

    matches = [cfg for cfg in planned if cfg.config_id.startswith(query)]
    exact_matches = [cfg for cfg in matches if cfg.config_id == query]
    if exact_matches:
        matches = exact_matches

    if not matches:
        raise ConfigError(
            f"No planned config matches '{query}' in {store.plan_configs_path}."
        )
    if len(matches) > 1:
        candidates = ", ".join(
            _short_config_id(cfg.config_id, width=12) for cfg in matches[:8]
        )
        suffix = "..." if len(matches) > 8 else ""
        raise ConfigError(
            f"Config id prefix '{query}' is ambiguous ({len(matches)} matches): "
            f"{candidates}{suffix}. Use a longer prefix."
        )

    config = matches[0]
    payload = config.to_json()
    payload["batch_file"] = str(store.batch_file(config.batch_index))

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.format == "yaml":
        print(yaml.safe_dump(payload, sort_keys=False), end="")
        return 0

    console = _console()
    overview = Table(title="Planned Config", show_header=False)
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value")
    overview.add_row("Run ID", str(payload.get("run_id", "")))
    overview.add_row("Config ID", str(payload.get("config_id", "")))
    overview.add_row("Batch", str(payload.get("batch_index", "")))
    overview.add_row("Line", str(payload.get("line_index", "")))
    overview.add_row("Batch File", str(payload.get("batch_file", "")))
    overview.add_row("W&B Name", str(payload.get("wandb_name", "")))
    overview.add_row("Tags", ", ".join(str(x) for x in _list(payload.get("tags"))))
    selected = _mapping(payload.get("selected_options"))
    selected_text = ", ".join(
        f"{pack}={option}" for pack, option in sorted(selected.items())
    )
    overview.add_row("Selected Options", selected_text or "-")
    overview.add_row("Command", str(payload.get("command", "")))
    console.print(overview)
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    spec, diagnostics = parse_experiment_yaml_with_diagnostics(
        args.experiment, run_set=args.run_set
    )
    payload: dict[str, Any] = {
        "valid": True,
        "run_set": args.run_set,
        "base_command": spec.base_command,
        "num_packs": len(spec.packs),
        "num_constraints": len(spec.constraints),
        "num_predicates": len(spec.predicates),
    }
    if args.show_diagnostics:
        payload["diagnostics"] = diagnostics

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    console = _console()
    table = Table(title="Validation Summary", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Valid", "true")
    table.add_row("Run Set", str(args.run_set or "<default>"))
    table.add_row("Base Command", str(spec.base_command))
    table.add_row("Packs", str(len(spec.packs)))
    table.add_row("Constraints", str(len(spec.constraints)))
    table.add_row("Predicates", str(len(spec.predicates)))
    console.print(table)
    if args.show_diagnostics:
        console.print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


def _execute_run_local(args: argparse.Namespace) -> dict[str, Any]:
    context = _prepare_run_execution_context(args, command="run_local")
    progress_enabled = bool(getattr(args, "progress", False))

    payloads = build_task_payloads(
        run_root=context.store.run_root,
        batch_indices=context.batches,
        batches_per_task=args.batches_per_task,
        executor=args.executor,
        max_workers=int(context.local_concurrency["max_concurrent_tasks"]),
        k_per_session=int(context.local_concurrency["cores_per_task"]),
        max_total_cores=(
            None
            if context.local_concurrency["max_total_cores"] is None
            else int(context.local_concurrency["max_total_cores"])
        ),
        cores=args.cores,
        tmux_prefix=args.tmux_prefix,
        env_setup_script=context.env_script,
        env_setup_commands=context.env_setup_cmds,
        env_vars=context.env_vars,
        dry_run=args.dry_run,
        include_config_ids=context.selected_config_ids,
        resume=args.resume,
        command_timeout_sec=args.command_timeout_sec,
        max_retries=args.max_retries,
        retry_on_status=context.retry_on_status,
        max_failures=args.max_failures,
        max_failure_rate=context.fail_fast_threshold,
    )

    summaries: list[dict[str, Any]] = []
    if progress_enabled and args.format == "json":
        _cli_log.warning("Ignoring --progress because --format json was requested.")

    if progress_enabled and args.format == "table":
        live_state: dict[str, Any] = {
            "run_id": context.store.run_id,
            "executor": args.executor,
            "concurrency": _format_concurrency(
                max_concurrent_tasks=int(
                    context.local_concurrency["max_concurrent_tasks"]
                ),
                cores_per_task=int(context.local_concurrency["cores_per_task"]),
                max_total_cores=(
                    None
                    if context.local_concurrency["max_total_cores"] is None
                    else int(context.local_concurrency["max_total_cores"])
                ),
            ),
            "payload_total": len(payloads),
            "payload_done": 0,
            "current_payload": "-",
            "active": {},
            "recent_starts": deque(maxlen=12),
            "recent_completed": deque(maxlen=12),
            "success": 0,
            "failed": 0,
            "terminated": 0,
            "retry_scheduled": 0,
        }
        live_ref: dict[str, Live | None] = {"handle": None}

        def _refresh_live() -> None:
            handle = live_ref["handle"]
            if handle is None:
                return
            handle.update(_render_local_live_dashboard(live_state), refresh=True)

        def _on_progress_event(event: dict[str, Any]) -> None:
            event_name = str(event.get("event", ""))
            if event_name == "task_start":
                live_state["run_id"] = str(event.get("run_id", live_state["run_id"]))
                live_state["executor"] = str(
                    event.get("executor", live_state["executor"])
                )
                _refresh_live()
                return
            if event_name == "command_start":
                attempt_id = str(event.get("attempt_id", ""))
                if attempt_id:
                    started_item = {
                        "time": event.get("time"),
                        "job_id": event.get("job_id"),
                        "array_task_id": event.get("array_task_id"),
                        "batch_index": event.get("batch_index"),
                        "line_index": event.get("line_index"),
                        "config_id": event.get("config_id"),
                        "attempt_id": attempt_id,
                        "pid": event.get("pid"),
                        "tmux_session": event.get("tmux_session"),
                        "assigned_cores": list(event.get("assigned_cores", [])),
                        "executor": event.get("executor"),
                    }
                    live_state["active"][attempt_id] = dict(started_item)
                    live_state["recent_starts"].appendleft(started_item)
                _refresh_live()
                return
            if event_name == "command_complete":
                attempt_id = str(event.get("attempt_id", ""))
                live_state["active"].pop(attempt_id, None)
                status = str(event.get("status", ""))
                if status in {"success", "failed", "terminated"}:
                    live_state[status] = _int_value(live_state.get(status)) + 1
                live_state["recent_completed"].appendleft(
                    {
                        "status": status,
                        "executor": event.get("executor", live_state.get("executor")),
                        "batch_index": event.get("batch_index"),
                        "line_index": event.get("line_index"),
                        "config_id": event.get("config_id"),
                        "duration_sec": event.get("duration_sec"),
                        "assigned_cores": list(event.get("assigned_cores", [])),
                        "pid": event.get("pid"),
                    }
                )
                _refresh_live()
                return
            if event_name == "retry_scheduled":
                live_state["retry_scheduled"] = (
                    _int_value(live_state.get("retry_scheduled")) + 1
                )
                _refresh_live()

        with Live(
            _render_local_live_dashboard(live_state),
            console=_console(),
            refresh_per_second=6,
            transient=False,
        ) as live:
            live_ref["handle"] = live
            _refresh_live()
            if not payloads:
                live_state["current_payload"] = "none"
                _refresh_live()

            for payload_index, payload in enumerate(payloads, start=1):
                live_state["current_payload"] = (
                    f"{payload_index}/{len(payloads)} "
                    f"(batches={','.join(str(x) for x in payload.batch_indices)})"
                )
                _refresh_live()
                local_task_id = str(payload_index - 1)
                result = execute_task_payload(
                    payload,
                    job_id="local",
                    array_task_id=local_task_id,
                    progress_callback=_on_progress_event,
                )
                summaries.append(result)
                live_state["payload_done"] = payload_index
                live_state["success"] = sum(
                    _int_value(item.get("success")) for item in summaries
                )
                live_state["failed"] = sum(
                    _int_value(item.get("failed")) for item in summaries
                )
                live_state["terminated"] = sum(
                    _int_value(item.get("terminated")) for item in summaries
                )
                live_state["retry_scheduled"] = sum(
                    _int_value(item.get("retry_scheduled")) for item in summaries
                )
                _refresh_live()
                # Keep the dashboard responsive after each payload boundary.
                time.sleep(0.01)

            live_state["current_payload"] = "done"
            _refresh_live()
    else:
        for payload_index, payload in enumerate(payloads):
            local_task_id = str(payload_index)
            result = execute_task_payload(
                payload, job_id="local", array_task_id=local_task_id
            )
            summaries.append(result)
    return {
        "payloads": summaries,
        "resolved_defaults_sources": context.resolved_defaults_sources,
    }


def _cmd_run_local(args: argparse.Namespace) -> int:
    payload = _execute_run_local(args)
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_run_local_table(payload)
    return 0


def _execute_run_slurm(args: argparse.Namespace) -> dict[str, Any]:
    context = _prepare_run_execution_context(args, command="run_slurm")
    slurm_config = _resolve_slurm_config(args=args, profile=context.profile)
    run_id = _resolve_run_id(context.store)
    for warning in slurm_config.warnings:
        _cli_log.warning("slurm config: %s", warning)

    payloads = build_task_payloads(
        run_root=context.store.run_root,
        batch_indices=context.batches,
        batches_per_task=args.batches_per_task,
        executor=args.executor,
        max_workers=int(context.local_concurrency["max_concurrent_tasks"]),
        k_per_session=int(context.local_concurrency["cores_per_task"]),
        max_total_cores=(
            None
            if context.local_concurrency["max_total_cores"] is None
            else int(context.local_concurrency["max_total_cores"])
        ),
        cores=args.cores,
        tmux_prefix=args.tmux_prefix,
        env_setup_script=context.env_script,
        env_setup_commands=context.env_setup_cmds,
        env_vars=context.env_vars,
        dry_run=args.dry_run,
        include_config_ids=context.selected_config_ids,
        resume=args.resume,
        command_timeout_sec=args.command_timeout_sec,
        max_retries=args.max_retries,
        retry_on_status=context.retry_on_status,
        max_failures=args.max_failures,
        max_failure_rate=context.fail_fast_threshold,
    )

    _append_launcher_event(
        context.store,
        {
            "event": "slurm_submit_start",
            "run_id": run_id,
            "profile_name": args.profile,
            "effective_config": slurm_config.to_json(),
            "num_payloads": len(payloads),
        },
    )

    summary = submit_payloads(
        store=context.store,
        payloads=payloads,
        slurm_config=slurm_config,
        dry_run=args.dry_run,
        profile_name=args.profile,
        profiles_file=str(context.profiles_path) if args.profile else None,
        run_id=run_id,
    )

    _append_launcher_event(
        context.store,
        {
            "event": "slurm_submit_end",
            "run_id": run_id,
            "submission_id": summary.get("submission_id"),
            "submitted": bool(summary.get("submitted", False)),
            "num_payloads": summary.get("num_payloads", 0),
            "job_ids": list(summary.get("job_ids", [])),
        },
    )

    if slurm_config.query_status and not bool(summary.get("dry_run", False)):
        queue_status = _refresh_queue_status(
            store=context.store,
            job_ids=list(summary.get("job_ids", [])),
            metadata=summary,
            persist=True,
        )
        _append_launcher_event(
            context.store,
            {
                "event": "queue_query",
                "run_id": run_id,
                "submission_id": summary.get("submission_id"),
                "job_ids": list(summary.get("job_ids", [])),
                "available": queue_status.get("available"),
                "num_rows": queue_status.get("num_rows"),
                "error": queue_status.get("error"),
            },
        )
        summary = {**summary, "queue_status": queue_status}

    source_of_truth = {
        "scalars": dict(slurm_config.sources),
        "sbatch_option": dict(slurm_config.sbatch_option_sources),
        "slurm_setup_cmds": dict(slurm_config.slurm_setup_sources),
    }
    summary = {
        **summary,
        "local_concurrency": context.local_concurrency,
        "resolved_defaults_sources": context.resolved_defaults_sources,
        "effective_config": slurm_config.to_json(),
        "source_of_truth": source_of_truth,
        "resolution_warnings": list(slurm_config.warnings),
        "launcher_log_path": str(context.store.launcher_log_path),
    }
    return summary


def _cmd_run_slurm(args: argparse.Namespace) -> int:
    summary = _execute_run_slurm(args)

    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _render_run_slurm_table(summary)
    return 0


def _cmd_list_profiles(args: argparse.Namespace) -> int:
    profiles_file, profiles = load_profiles(args.profiles_file, required=True)
    payload = {
        "profiles_file": str(profiles_file),
        "count": len(profiles),
        "profiles": {
            name: profile.to_json() for name, profile in sorted(profiles.items())
        },
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_profiles_table(profiles_file, profiles)
    return 0


def _cmd_queue(args: argparse.Namespace) -> int:
    store = ArtifactStore.from_run_dir(args.run)
    job_ids, metadata = _resolve_queue_job_ids(
        store=store,
        explicit_job_ids=getattr(args, "job_id", None),
    )
    payload = _refresh_queue_status(
        store=store,
        job_ids=job_ids,
        metadata=metadata,
        persist=False,
    )
    _append_launcher_event(
        store,
        {
            "event": "queue_query",
            "run_id": payload.get("run_id"),
            "submission_id": payload.get("submission_id"),
            "job_ids": list(payload.get("job_ids", [])),
            "available": payload.get("available"),
            "num_rows": payload.get("num_rows"),
            "error": payload.get("error"),
        },
    )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_queue_status_table(payload)
    return 0


def _cmd_queue_refresh(args: argparse.Namespace) -> int:
    store = ArtifactStore.from_run_dir(args.run)
    job_ids, metadata = _resolve_queue_job_ids(
        store=store,
        explicit_job_ids=getattr(args, "job_id", None),
    )
    payload = _refresh_queue_status(
        store=store,
        job_ids=job_ids,
        metadata=metadata,
        persist=True,
    )
    _append_launcher_event(
        store,
        {
            "event": "queue_refresh",
            "run_id": payload.get("run_id"),
            "submission_id": payload.get("submission_id"),
            "job_ids": list(payload.get("job_ids", [])),
            "available": payload.get("available"),
            "num_rows": payload.get("num_rows"),
            "error": payload.get("error"),
        },
    )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_queue_status_table(payload)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    index = build_run_status_index(args.run, strict_jsonl=bool(args.strict_jsonl))
    summary = summarize_run_status(index)
    by_packs = _parse_by_pack_args(args.by_pack)
    grouped = summarize_status_groups(index, by_packs=by_packs)
    payload = {**summary, "grouped_by_packs": grouped}
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_status_table(summary, grouped_by_packs=grouped)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    index = build_run_status_index(args.run, strict_jsonl=bool(args.strict_jsonl))
    by_packs = _parse_by_pack_args(args.by_pack)
    report_payload = build_run_report_payload(index, by_packs=by_packs)

    if args.format == "json":
        text = json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
        if args.out:
            output_path = Path(args.out).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            print(
                json.dumps({"output_path": str(output_path)}, indent=2, sort_keys=True)
            )
        else:
            print(text, end="")
        return 0

    if args.format == "markdown":
        text = render_report_markdown(report_payload)
        if args.out:
            output_path = Path(args.out).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            print(
                json.dumps({"output_path": str(output_path)}, indent=2, sort_keys=True)
            )
        else:
            print(text, end="")
        return 0

    _render_report_table(report_payload)
    if args.out:
        output_path = Path(args.out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_report_markdown(report_payload), encoding="utf-8")
        print(json.dumps({"output_path": str(output_path)}, indent=2, sort_keys=True))
    return 0


def _build_retry_payload(
    *,
    run: str,
    status_filter: str,
    explicit_config_ids: list[str] | None,
    dry_run: bool,
) -> tuple[dict[str, Any], Any]:
    index = build_run_status_index(run)
    explicit_ids = list(dict.fromkeys(explicit_config_ids or []))
    target_config_ids = select_rerun_config_ids(
        index,
        status_filter=status_filter,
        explicit_config_ids=explicit_ids,
    )
    if not target_config_ids:
        raise ConfigError(
            f"No rerun targets found for --status {status_filter!r}. "
            "Use --config-id to force specific configs."
        )

    target_batch_indices = sorted(
        {
            batch_index
            for config_id in target_config_ids
            for batch_index in index.config_status_by_id[config_id].batch_indices
        }
    )
    created_at = utc_now_iso()
    retry_stamp = created_at.replace("-", "").replace(":", "").replace("T", "_")

    payload = {
        "retry_id": f"retry_{retry_stamp}",
        "created_at": created_at,
        "run_id": index.run_id,
        "run_root": str(index.run_root),
        "status_filter": status_filter,
        "explicit_config_ids": explicit_ids,
        "target_config_ids": target_config_ids,
        "target_batch_indices": target_batch_indices,
        "num_target_configs": len(target_config_ids),
        "num_target_batches": len(target_batch_indices),
        "resume_required": True,
        "notes": [
            "Use `--retry-file` with run-local/run-slurm to execute this retry selection.",
            "Retry execution appends attempts; existing result records are not modified.",
        ],
    }

    retry_file: str | None = None
    if not dry_run:
        store = ArtifactStore.from_run_dir(run)
        store.ensure_exec_layout()
        retry_path = store.retries_dir / f"retry_{retry_stamp}.json"
        retry_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        retry_file = str(retry_path)

    payload_with_path = {
        **payload,
        "retry_file": retry_file,
        "dry_run": bool(dry_run),
    }
    return payload_with_path, index


def _cmd_launch(args: argparse.Namespace) -> int:
    validation_payload: dict[str, Any] | None = None
    if not args.skip_validate:
        spec, diagnostics = parse_experiment_yaml_with_diagnostics(
            args.experiment, run_set=args.run_set
        )
        validation_payload = {
            "valid": True,
            "run_set": args.run_set,
            "base_command": spec.base_command,
            "num_packs": len(spec.packs),
            "num_constraints": len(spec.constraints),
            "num_predicates": len(spec.predicates),
            "diagnostics": diagnostics,
        }

    plan_summary = plan_experiment(
        experiment_path=args.experiment,
        out_dir=args.out,
        batch_size=args.batch_size,
        run_id=args.run_id,
        run_set=args.run_set,
        dry_run=bool(args.dry_run),
    )
    plan_payload = {
        "run_set": args.run_set,
        "run_id": plan_summary.run_id,
        "run_root": str(plan_summary.run_root),
        "total_configs": plan_summary.total_configs,
        "total_batches": plan_summary.total_batches,
        "dry_run": bool(args.dry_run),
    }

    execution_payload: dict[str, Any]
    if args.dry_run:
        execution_payload = {
            "skipped": True,
            "reason": "dry_run",
            "backend": args.backend,
        }
    else:
        args.run = str(plan_summary.run_root)
        execution_payload = (
            _execute_run_local(args)
            if args.backend == "local"
            else _execute_run_slurm(args)
        )
    payload = {
        "backend": args.backend,
        "dry_run": bool(args.dry_run),
        "validation": validation_payload,
        "plan": plan_payload,
        "execution": execution_payload,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_plan_table(plan_payload)
        if args.backend == "local" and not bool(execution_payload.get("skipped")):
            _render_run_local_table(execution_payload)
        elif not bool(execution_payload.get("skipped")):
            _render_run_slurm_table(execution_payload)
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    retry_payload, index = _build_retry_payload(
        run=args.run,
        status_filter=args.status,
        explicit_config_ids=args.config_id,
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        execution_payload: dict[str, Any] = {
            "skipped": True,
            "reason": "dry_run",
            "backend": args.backend,
        }
    else:
        args.resume = True
        args.run = str(index.run_root)
        args.retry_file = retry_payload.get("retry_file")
        execution_payload = (
            _execute_run_local(args)
            if args.backend == "local"
            else _execute_run_slurm(args)
        )
    payload = {
        "backend": args.backend,
        "dry_run": bool(args.dry_run),
        "retry": retry_payload,
        "execution": execution_payload,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_rerun_table(retry_payload)
        if args.backend == "local" and not bool(execution_payload.get("skipped")):
            _render_run_local_table(execution_payload)
        elif not bool(execution_payload.get("skipped")):
            _render_run_slurm_table(execution_payload)
    return 0


def _cmd_rerun(args: argparse.Namespace) -> int:
    payload_with_path, index = _build_retry_payload(
        run=args.run,
        status_filter=args.status,
        explicit_config_ids=args.config_id,
        dry_run=bool(args.dry_run),
    )
    retry_file = payload_with_path.get("retry_file")

    if args.format == "json":
        print(json.dumps(payload_with_path, indent=2, sort_keys=True))
    else:
        _render_rerun_table(payload_with_path)
        if retry_file:
            console = _console()
            console.print(
                f"[bold]Next:[/bold] geryon run-local --run {index.run_root} "
                f"--retry-file {retry_file}"
            )
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    summary = collect_run(
        args.run,
        dry_run=args.dry_run,
        strict_jsonl=bool(args.strict_jsonl),
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _render_collect_table(summary)
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    store = ArtifactStore.from_run_dir(args.run)
    removed: list[str] = []

    if not args.all and not args.plan and not args.exec:
        raise ConfigError("Specify one of --plan, --exec, or --all for clean.")

    if args.all:
        shutil.rmtree(store.run_root)
        removed.append(str(store.run_root))
    else:
        remove_plan = args.plan
        remove_exec = args.exec

        if remove_plan:
            shutil.rmtree(store.plan_dir)
            removed.append(str(store.plan_dir))
        if remove_exec:
            shutil.rmtree(store.exec_dir)
            removed.append(str(store.exec_dir))

    print(json.dumps({"removed": removed}, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    invoked = raw_argv[0] if raw_argv else ""
    _maybe_print_alias_notice(invoked)
    command = str(getattr(args, "command", invoked or "unknown"))
    argv_text = " ".join(raw_argv)
    started = time.perf_counter()
    _cli_log.info("cli_command_start command=%s argv=%s", command, argv_text)

    exit_code = 1
    try:
        exit_code = int(args.handler(args))
    except ConfigError as exc:
        _cli_log.error(
            "cli_command_error command=%s kind=config error=%s", command, exc
        )
        print(f"[config error] {exc}", file=sys.stderr)
        exit_code = 2
    except KeyboardInterrupt:
        _cli_log.error("cli_command_error command=%s kind=interrupted", command)
        print("\n[interrupted]", file=sys.stderr)
        exit_code = 130
    except RuntimeError as exc:
        _cli_log.error(
            "cli_command_error command=%s kind=runtime error=%s", command, exc
        )
        print(f"[runtime error] {exc}", file=sys.stderr)
        exit_code = 1
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        _cli_log.error(
            "cli_command_error command=%s kind=%s error=%s",
            command,
            type(exc).__name__,
            exc,
        )
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 1
    except subprocess.CalledProcessError as exc:
        stderr_msg = exc.stderr.strip() if exc.stderr else ""
        _cli_log.error(
            "cli_command_error command=%s kind=called_process exit=%s cmd=%s stderr=%s",
            command,
            exc.returncode,
            " ".join(exc.cmd),
            stderr_msg,
        )
        print(
            f"[error] Command failed (exit {exc.returncode}): "
            f"{' '.join(exc.cmd)}"
            f"{': ' + stderr_msg if stderr_msg else ''}",
            file=sys.stderr,
        )
        exit_code = 1
    finally:
        duration_sec = time.perf_counter() - started
        _cli_log.info(
            "cli_command_end command=%s exit_code=%s duration_sec=%.3f",
            command,
            exit_code,
            duration_sec,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
