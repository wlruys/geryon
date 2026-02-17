from __future__ import annotations

import logging
import re
import shlex
import socket
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from geryon.executors import (
    ProcessBatchExecutor,
    PyLauncherBatchExecutor,
    TmuxBatchExecutor,
)
from geryon.executors.base import CommandLaunch
from geryon.models import (
    AttemptRecord,
    ConfigError,
    PlannedConfig,
    TaskContext,
    TaskPayload,
)
from geryon.status import successful_config_ids
from geryon.store import ArtifactStore
from geryon.task_event_schema import (
    BATCH_EVENT_SCHEMA_VERSION,
    TASK_EVENT_SCHEMA_VERSION,
    validate_batch_event,
    validate_task_event,
)
from geryon.utils import append_jsonl, read_jsonl, render_hydra_override, utc_now_iso

_log = logging.getLogger("geryon.execution")
ProgressEventCallback = Callable[[dict[str, Any]], None]


def _notify_progress_event(
    callback: ProgressEventCallback | None, event: dict[str, Any]
) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:  # pragma: no cover - defensive callback isolation
        _log.warning("Ignoring progress callback error: %s", exc)


def _event_with_correlation(
    event: dict[str, Any],
    *,
    run_id: str,
    job_id: str,
    array_task_id: str,
    submission_id: str | None,
) -> dict[str, Any]:
    payload = dict(event)
    payload["schema_version"] = TASK_EVENT_SCHEMA_VERSION
    payload["run_id"] = run_id
    payload["job_id"] = job_id
    payload["array_task_id"] = array_task_id
    payload["submission_id"] = submission_id
    return payload


def _append_task_event(
    *,
    task_events_path: Path,
    event: dict[str, Any],
    context: TaskContext,
    submission_id: str | None,
    progress_callback: ProgressEventCallback | None,
) -> None:
    payload = _event_with_correlation(
        event,
        run_id=context.run_id,
        job_id=context.job_id,
        array_task_id=context.array_task_id,
        submission_id=submission_id,
    )
    validated = validate_task_event(payload)
    append_jsonl(task_events_path, validated)
    _notify_progress_event(progress_callback, validated)


def _event_with_batch_correlation(
    event: dict[str, Any],
    *,
    run_id: str,
    job_id: str,
    array_task_id: str,
    submission_id: str | None,
) -> dict[str, Any]:
    payload = dict(event)
    payload["schema_version"] = BATCH_EVENT_SCHEMA_VERSION
    payload["run_id"] = run_id
    payload["job_id"] = job_id
    payload["array_task_id"] = array_task_id
    payload["submission_id"] = submission_id
    return payload


def _append_batch_event(
    *,
    batch_log_path: Path,
    event: dict[str, Any],
    context: TaskContext,
    submission_id: str | None,
) -> None:
    payload = _event_with_batch_correlation(
        event,
        run_id=context.run_id,
        job_id=context.job_id,
        array_task_id=context.array_task_id,
        submission_id=submission_id,
    )
    validated = validate_batch_event(payload)
    append_jsonl(batch_log_path, validated)


def _resolve_context(
    store: ArtifactStore, *, job_id: str | None, array_task_id: str | None
) -> TaskContext:
    meta = (
        store.read_run_meta()
        if store.run_meta_path.exists()
        else {"run_id": store.run_id}
    )
    resolved_job_id = job_id or "local"
    resolved_array_task = array_task_id or "0"
    return TaskContext(
        run_id=str(meta.get("run_id", store.run_id)),
        run_root=store.run_root,
        job_id=str(resolved_job_id),
        array_task_id=str(resolved_array_task),
        hostname=socket.gethostname(),
    )


def _group_configs_by_batch(
    configs: Iterable[PlannedConfig],
) -> dict[int, list[PlannedConfig]]:
    grouped: dict[int, list[PlannedConfig]] = defaultdict(list)
    for cfg in configs:
        grouped[cfg.batch_index].append(cfg)
    for values in grouped.values():
        values.sort(key=lambda item: item.line_index)
    return grouped


def _attempt_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("end_time", "")),
        str(record.get("start_time", "")),
        str(record.get("attempt_id", "")),
    )


def _load_attempt_history(store: ArtifactStore) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(store.results_dir.glob("task_*.jsonl")):
        for record in read_jsonl(path):
            config_id = str(record.get("config_id", ""))
            if not config_id:
                continue
            history[config_id].append(record)
    for items in history.values():
        items.sort(key=_attempt_sort_key)
    return history


def available_batch_indices(store: ArtifactStore) -> list[int]:
    batches = store.read_planned_batches()
    return [item.batch_index for item in batches]


def select_batch_indices(
    store: ArtifactStore, requested: list[int] | None
) -> list[int]:
    available = set(available_batch_indices(store))
    if not requested:
        return sorted(available)

    selected = sorted(set(requested))
    missing = [idx for idx in selected if idx not in available]
    if missing:
        raise ConfigError(f"Unknown batch indices: {missing}")
    return selected


def build_task_payloads(
    *,
    run_root: str | Path,
    batch_indices: list[int],
    batches_per_task: int,
    executor: str,
    max_workers: int,
    k_per_session: int,
    max_total_cores: int | None,
    cores: str | None,
    tmux_prefix: str,
    env_setup_script: str | None,
    env_setup_commands: list[str],
    env_vars: dict[str, str],
    dry_run: bool,
    include_config_ids: list[str] | None = None,
    resume: bool = False,
    command_timeout_sec: int | None = None,
    max_retries: int = 0,
    retry_on_status: list[str] | None = None,
    max_failures: int | None = None,
    max_failure_rate: float | None = None,
) -> list[TaskPayload]:
    if batches_per_task <= 0:
        raise ConfigError("batches_per_task must be positive")
    if max_workers <= 0:
        raise ConfigError("max_concurrent_tasks/max_workers must be positive")
    if k_per_session <= 0:
        raise ConfigError("cores_per_task/k_per_session must be positive")
    if max_total_cores is not None:
        if max_total_cores <= 0:
            raise ConfigError("max_total_cores must be positive when provided")
        allowed_workers = max_total_cores // k_per_session
        if allowed_workers < 1:
            raise ConfigError("max_total_cores must be >= cores_per_task/k_per_session")
        max_workers = min(max_workers, allowed_workers)

    script_abs: str | None = None
    if env_setup_script:
        script_abs = str(Path(env_setup_script).expanduser().resolve())

    selected_config_ids: tuple[str, ...] = tuple(
        dict.fromkeys(include_config_ids or [])
    )

    chunks: list[TaskPayload] = []
    for i in range(0, len(batch_indices), batches_per_task):
        chunk = tuple(batch_indices[i : i + batches_per_task])
        chunks.append(
            TaskPayload(
                run_root=str(Path(run_root).resolve()),
                batch_indices=chunk,
                executor=executor,
                max_workers=max_workers,
                k_per_session=k_per_session,
                max_total_cores=max_total_cores,
                cores=cores,
                tmux_prefix=tmux_prefix,
                env_setup_script=script_abs,
                env_setup_commands=tuple(env_setup_commands),
                env_vars=dict(env_vars),
                include_config_ids=selected_config_ids,
                resume=resume,
                command_timeout_sec=command_timeout_sec,
                max_retries=max_retries,
                retry_on_status=tuple(retry_on_status or ["failed", "terminated"]),
                max_failures=max_failures,
                max_failure_rate=max_failure_rate,
                dry_run=dry_run,
            )
        )
    return chunks


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TASK_WORK_DIR_KEY = "hydra.run.dir"


def _wrap_command_with_env_setup(command: str, payload: TaskPayload) -> str:
    prefix_parts: list[str] = []

    for name, value in sorted(payload.env_vars.items()):
        if not _ENV_NAME_RE.match(name):
            raise ConfigError(f"Invalid environment variable name '{name}'")
        prefix_parts.append(f"export {name}={shlex.quote(value)}")

    if payload.env_setup_script:
        if not Path(payload.env_setup_script).exists():
            raise ConfigError(
                f"Environment setup script not found: {payload.env_setup_script}"
            )
        prefix_parts.append(f"source {shlex.quote(payload.env_setup_script)}")

    for cmd in payload.env_setup_commands:
        if cmd.strip():
            prefix_parts.append(cmd.strip())

    if not prefix_parts:
        return command

    return " && ".join([*prefix_parts, command])


def _append_task_work_dir_override(command: str, work_dir: Path) -> str:
    override = render_hydra_override(_TASK_WORK_DIR_KEY, str(work_dir))
    return f"{command} {override}"


def _validate_policy(payload: TaskPayload) -> None:
    if payload.max_retries < 0:
        raise ConfigError("max_retries must be >= 0")
    if payload.command_timeout_sec is not None and payload.command_timeout_sec <= 0:
        raise ConfigError("command_timeout_sec must be > 0")
    if payload.max_failures is not None and payload.max_failures <= 0:
        raise ConfigError("max_failures must be > 0")
    if payload.max_failure_rate is not None and not (0 < payload.max_failure_rate <= 1):
        raise ConfigError("max_failure_rate must be in (0, 1]")

    allowed = {"failed", "terminated"}
    unknown = sorted(
        {status for status in payload.retry_on_status if status not in allowed}
    )
    if unknown:
        raise ConfigError(
            f"retry_on_status contains unsupported values: {unknown}. "
            f"Allowed values: {sorted(allowed)}"
        )


def _check_fail_fast(
    *,
    terminal_failed: int,
    terminal_total: int,
    max_failures: int | None,
    max_failure_rate: float | None,
) -> str | None:
    if max_failures is not None and terminal_failed >= max_failures:
        return "max_failures"

    if max_failure_rate is not None and terminal_total > 0:
        ratio = terminal_failed / terminal_total
        if ratio >= max_failure_rate:
            return "max_failure_rate"
    return None


def execute_task_payload(
    payload: TaskPayload,
    *,
    job_id: str | None = None,
    array_task_id: str | None = None,
    submission_id: str | None = None,
    progress_callback: ProgressEventCallback | None = None,
) -> dict[str, Any]:
    _validate_policy(payload)
    store = ArtifactStore.from_run_dir(payload.run_root)
    store.ensure_exec_layout()
    context = _resolve_context(store, job_id=job_id, array_task_id=array_task_id)
    result_path = store.task_result_file(context.job_id, context.array_task_id)
    task_events_path = store.task_events_file(context.job_id, context.array_task_id)
    task_stdout_path = store.task_stdout_log_file(context.job_id, context.array_task_id)
    task_stderr_path = store.task_stderr_log_file(context.job_id, context.array_task_id)
    task_stdout_path.parent.mkdir(parents=True, exist_ok=True)
    task_stdout_path.touch(exist_ok=True)
    task_stderr_path.touch(exist_ok=True)

    all_configs = store.read_planned_configs()
    grouped = _group_configs_by_batch(all_configs)
    planned_config_ids = {cfg.config_id for cfg in all_configs}

    selected_config_ids = set(payload.include_config_ids)
    if selected_config_ids:
        unknown = sorted(
            config_id
            for config_id in selected_config_ids
            if config_id not in planned_config_ids
        )
        if unknown:
            raise ConfigError(
                f"Unknown config IDs in payload.include_config_ids: {unknown}"
            )

    previous_successes = successful_config_ids(store) if payload.resume else set()
    attempt_history = _load_attempt_history(store)

    if payload.executor not in {"process", "tmux", "pylauncher"}:
        raise ConfigError(f"Unsupported executor '{payload.executor}'")

    if payload.executor == "process":
        executor = ProcessBatchExecutor(
            max_workers=payload.max_workers,
            k_per_worker=payload.k_per_session,
            cores=payload.cores,
            command_timeout_sec=payload.command_timeout_sec,
        )
    elif payload.executor == "pylauncher":
        state_dir = (
            store.batch_logs_dir
            / f"task_{context.job_id}_{context.array_task_id}"
            / "pylauncher_state"
        )
        executor = PyLauncherBatchExecutor(
            state_dir=state_dir,
            k_per_launch=payload.k_per_session,
            max_parallel_tasks=payload.max_workers,
            cores=payload.cores,
            command_timeout_sec=payload.command_timeout_sec,
        )
    else:
        state_dir = (
            store.batch_logs_dir
            / f"task_{context.job_id}_{context.array_task_id}"
            / "state"
        )
        executor = TmuxBatchExecutor(
            state_dir=state_dir,
            k_per_session=payload.k_per_session,
            max_parallel_tasks=payload.max_workers,
            cores=payload.cores,
            prefix=payload.tmux_prefix,
            command_timeout_sec=payload.command_timeout_sec,
        )

    summary = {
        "run_id": context.run_id,
        "submission_id": submission_id,
        "job_id": context.job_id,
        "array_task_id": context.array_task_id,
        "executor": payload.executor,
        "batches": list(payload.batch_indices),
        "result_path": str(result_path),
        "total": 0,
        "success": 0,
        "failed": 0,
        "terminated": 0,
        "skipped_resume": 0,
        "skipped_not_selected": 0,
        "skipped_fail_fast": 0,
        "retry_scheduled": 0,
        "dry_run": payload.dry_run,
        "resume": payload.resume,
        "selected_config_ids": list(payload.include_config_ids),
        "concurrency": {
            "cores_per_task": payload.k_per_session,
            "max_concurrent_tasks": payload.max_workers,
            "max_total_cores": payload.max_total_cores,
        },
        "policy": {
            "command_timeout_sec": payload.command_timeout_sec,
            "max_retries": payload.max_retries,
            "retry_on_status": list(payload.retry_on_status),
            "max_failures": payload.max_failures,
            "max_failure_rate": payload.max_failure_rate,
        },
        "fail_fast_triggered": False,
        "fail_fast_reason": None,
        "task_events_path": str(task_events_path),
        "task_stdout_log": str(task_stdout_path),
        "task_stderr_log": str(task_stderr_path),
    }

    batch_log_dir = store.task_log_dir(context.job_id, context.array_task_id)
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    task_start_event = {
        "event": "task_start",
        "time": utc_now_iso(),
        "run_id": context.run_id,
        "job_id": context.job_id,
        "array_task_id": context.array_task_id,
        "executor": payload.executor,
        "batches": list(payload.batch_indices),
        "resume": payload.resume,
        "selected_config_ids": list(payload.include_config_ids),
        "policy": dict(summary["policy"]),
    }
    _append_task_event(
        task_events_path=task_events_path,
        event=task_start_event,
        context=context,
        submission_id=submission_id,
        progress_callback=progress_callback,
    )

    retry_on_status = set(payload.retry_on_status)
    retries_used_by_config: dict[str, int] = defaultdict(int)
    terminal_status_by_config: dict[str, str] = {}
    terminal_total = 0
    terminal_failed = 0
    fail_fast_reason: str | None = None

    def _build_launch(cfg: PlannedConfig) -> CommandLaunch:
        attempt_id = uuid.uuid4().hex
        cmd_dir = (
            store.cmd_logs_dir
            / f"task_{context.job_id}_{context.array_task_id}"
            / f"batch_{cfg.batch_index:03d}"
        )
        stdout_path = (
            cmd_dir
            / f"line_{cfg.line_index:04d}_{cfg.config_id[:8]}_{attempt_id[:8]}.stdout.log"
        )
        stderr_path = (
            cmd_dir
            / f"line_{cfg.line_index:04d}_{cfg.config_id[:8]}_{attempt_id[:8]}.stderr.log"
        )
        work_dir = store.command_work_dir(
            context.job_id,
            context.array_task_id,
            batch_index=cfg.batch_index,
            line_index=cfg.line_index,
            config_id=cfg.config_id,
            attempt_id=attempt_id,
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        command_with_work_dir = _append_task_work_dir_override(cfg.command, work_dir)
        return CommandLaunch(
            config=cfg,
            command=_wrap_command_with_env_setup(command_with_work_dir, payload),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            attempt_id=attempt_id,
        )

    try:
        _run_batch_loop(
            payload=payload,
            executor=executor,
            store=store,
            context=context,
            summary=summary,
            batch_log_dir=batch_log_dir,
            task_events_path=task_events_path,
            result_path=result_path,
            grouped=grouped,
            selected_config_ids=selected_config_ids,
            previous_successes=previous_successes,
            attempt_history=attempt_history,
            retry_on_status=retry_on_status,
            retries_used_by_config=retries_used_by_config,
            terminal_status_by_config=terminal_status_by_config,
            terminal_counters={"total": terminal_total, "failed": terminal_failed},
            fail_fast_holder=[fail_fast_reason],
            build_launch=_build_launch,
            submission_id=submission_id,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        error_repr = repr(exc)
        _log.error("Executor error during batch processing: %s", exc)
        try:
            _append_task_event(
                task_events_path=task_events_path,
                event={
                    "event": "executor_error",
                    "time": utc_now_iso(),
                    "error": error_repr,
                },
                context=context,
                submission_id=submission_id,
                progress_callback=progress_callback,
            )
        except OSError:
            pass
        _write_task_end(
            task_events_path,
            summary,
            context=context,
            submission_id=submission_id,
            executor_error=error_repr,
            progress_callback=progress_callback,
        )
        raise

    _write_task_end(
        task_events_path,
        summary,
        context=context,
        submission_id=submission_id,
        progress_callback=progress_callback,
    )
    return summary


def _run_batch_loop(  # noqa: PLR0913
    *,
    payload: TaskPayload,
    executor: ProcessBatchExecutor | TmuxBatchExecutor | PyLauncherBatchExecutor,
    store: ArtifactStore,
    context: TaskContext,
    summary: dict[str, Any],
    batch_log_dir: Path,
    task_events_path: Path,
    result_path: Path,
    grouped: dict[int, list[PlannedConfig]],
    selected_config_ids: set[str],
    previous_successes: set[str],
    attempt_history: dict[str, list[dict[str, Any]]],
    retry_on_status: set[str],
    retries_used_by_config: dict[str, int],
    terminal_status_by_config: dict[str, str],
    terminal_counters: dict[str, int],
    fail_fast_holder: list[str | None],
    build_launch: Any,
    submission_id: str | None,
    progress_callback: ProgressEventCallback | None = None,
) -> None:
    """Inner batch-processing loop, extracted so the caller can wrap with try/except."""
    fail_fast_reason = fail_fast_holder[0]
    terminal_total = terminal_counters["total"]
    terminal_failed = terminal_counters["failed"]

    for batch_index in payload.batch_indices:
        batch_configs = grouped.get(batch_index, [])
        if not batch_configs:
            raise RuntimeError(
                f"Batch {batch_index} has no planned configs "
                f"(plan data may be corrupt or out of sync)"
            )

        batch_log_path = batch_log_dir / f"batch_{batch_index:03d}.jsonl"
        batch_start_event = {
            "event": "batch_start",
            "time": utc_now_iso(),
            "run_id": context.run_id,
            "submission_id": submission_id,
            "batch_index": batch_index,
            "num_commands": len(batch_configs),
            "executor": payload.executor,
            "job_id": context.job_id,
            "array_task_id": context.array_task_id,
        }
        _append_batch_event(
            batch_log_path=batch_log_path,
            event=batch_start_event,
            context=context,
            submission_id=submission_id,
        )
        _append_task_event(
            task_events_path=task_events_path,
            event=batch_start_event,
            context=context,
            submission_id=submission_id,
            progress_callback=progress_callback,
        )

        batch_selected = batch_configs
        skipped_not_selected = 0
        skipped_resume = 0
        if selected_config_ids:
            batch_selected = []
            for cfg in batch_configs:
                if cfg.config_id in selected_config_ids:
                    batch_selected.append(cfg)
                else:
                    skipped_not_selected += 1

        if payload.resume:
            resume_filtered: list[PlannedConfig] = []
            for cfg in batch_selected:
                if cfg.config_id in previous_successes:
                    skipped_resume += 1
                    continue
                resume_filtered.append(cfg)
            batch_selected = resume_filtered

        summary["skipped_not_selected"] += skipped_not_selected
        summary["skipped_resume"] += skipped_resume

        batch_filter_event = {
            "event": "batch_filter",
            "time": utc_now_iso(),
            "run_id": context.run_id,
            "submission_id": submission_id,
            "batch_index": batch_index,
            "planned_commands": len(batch_configs),
            "selected_commands": len(batch_selected),
            "skipped_not_selected": skipped_not_selected,
            "skipped_resume": skipped_resume,
            "job_id": context.job_id,
            "array_task_id": context.array_task_id,
        }
        _append_batch_event(
            batch_log_path=batch_log_path,
            event=batch_filter_event,
            context=context,
            submission_id=submission_id,
        )
        _append_task_event(
            task_events_path=task_events_path,
            event=batch_filter_event,
            context=context,
            submission_id=submission_id,
            progress_callback=progress_callback,
        )

        if not batch_selected:
            _append_batch_event(
                batch_log_path=batch_log_path,
                event={
                    "event": "batch_end",
                    "time": utc_now_iso(),
                    "run_id": context.run_id,
                    "submission_id": submission_id,
                    "batch_index": batch_index,
                    "status": "skipped",
                },
                context=context,
                submission_id=submission_id,
            )
            skipped_event = {
                "event": "batch_skipped",
                "time": utc_now_iso(),
                "batch_index": batch_index,
                "skipped_not_selected": skipped_not_selected,
                "skipped_resume": skipped_resume,
                "job_id": context.job_id,
                "array_task_id": context.array_task_id,
            }
            _append_task_event(
                task_events_path=task_events_path,
                event=skipped_event,
                context=context,
                submission_id=submission_id,
                progress_callback=progress_callback,
            )
            continue

        if fail_fast_reason is not None:
            summary["skipped_fail_fast"] += len(batch_selected)
            _append_batch_event(
                batch_log_path=batch_log_path,
                event={
                    "event": "batch_end",
                    "time": utc_now_iso(),
                    "run_id": context.run_id,
                    "submission_id": submission_id,
                    "batch_index": batch_index,
                    "status": "skipped_fail_fast",
                    "reason": fail_fast_reason,
                },
                context=context,
                submission_id=submission_id,
            )
            skipped_fail_fast_event = {
                "event": "batch_skipped_fail_fast",
                "time": utc_now_iso(),
                "batch_index": batch_index,
                "num_commands": len(batch_selected),
                "reason": fail_fast_reason,
                "job_id": context.job_id,
                "array_task_id": context.array_task_id,
            }
            _append_task_event(
                task_events_path=task_events_path,
                event=skipped_fail_fast_event,
                context=context,
                submission_id=submission_id,
                progress_callback=progress_callback,
            )
            continue

        launches: list[CommandLaunch] = [build_launch(cfg) for cfg in batch_selected]

        if payload.dry_run:
            summary["total"] += len(launches)
            dry_run_end_event = {
                "event": "batch_end",
                "time": utc_now_iso(),
                "run_id": context.run_id,
                "submission_id": submission_id,
                "batch_index": batch_index,
                "status": "dry_run",
                "job_id": context.job_id,
                "array_task_id": context.array_task_id,
            }
            _append_batch_event(
                batch_log_path=batch_log_path,
                event=dry_run_end_event,
                context=context,
                submission_id=submission_id,
            )
            _notify_progress_event(progress_callback, dry_run_end_event)
            continue

        wave_index = 0
        pending_launches = list(launches)
        while pending_launches:
            wave_index += 1
            current_launches = pending_launches
            pending_launches = []

            batch_launch_event = {
                "event": "batch_launch",
                "time": utc_now_iso(),
                "run_id": context.run_id,
                "submission_id": submission_id,
                "batch_index": batch_index,
                "wave_index": wave_index,
                "num_commands": len(current_launches),
                "job_id": context.job_id,
                "array_task_id": context.array_task_id,
            }
            _append_task_event(
                task_events_path=task_events_path,
                event=batch_launch_event,
                context=context,
                submission_id=submission_id,
                progress_callback=progress_callback,
            )

            def _on_command_start(launch: CommandLaunch, meta: dict[str, Any]) -> None:
                start_event = {
                    "event": "command_start",
                    "time": utc_now_iso(),
                    "run_id": context.run_id,
                    "submission_id": submission_id,
                    "batch_index": launch.config.batch_index,
                    "line_index": launch.config.line_index,
                    "config_id": launch.config.config_id,
                    "attempt_id": launch.attempt_id,
                    "wave_index": wave_index,
                    "job_id": context.job_id,
                    "array_task_id": context.array_task_id,
                    "pid": meta.get("pid"),
                    "tmux_session": meta.get("tmux_session"),
                    "assigned_cores": list(meta.get("assigned_cores") or []),
                    "executor": str(meta.get("executor") or payload.executor),
                }
                _append_task_event(
                    task_events_path=task_events_path,
                    event=start_event,
                    context=context,
                    submission_id=submission_id,
                    progress_callback=progress_callback,
                )
                _append_batch_event(
                    batch_log_path=batch_log_path,
                    event=start_event,
                    context=context,
                    submission_id=submission_id,
                )

            for completed in executor.run(current_launches, on_start=_on_command_start):
                cfg = completed.launch.config
                history = attempt_history.get(cfg.config_id, [])
                attempt_index = len(history)
                parent_attempt_id = None
                if history:
                    parent_raw = str(history[-1].get("attempt_id", ""))
                    parent_attempt_id = parent_raw or None

                record = AttemptRecord(
                    run_id=context.run_id,
                    config_id=cfg.config_id,
                    batch_index=cfg.batch_index,
                    line_index=cfg.line_index,
                    attempt_id=completed.launch.attempt_id,
                    attempt_index=attempt_index,
                    parent_attempt_id=parent_attempt_id,
                    hostname=context.hostname,
                    pid=completed.pid,
                    tmux_session=completed.tmux_session,
                    job_id=context.job_id,
                    array_task_id=context.array_task_id,
                    start_time=completed.start_time,
                    end_time=completed.end_time,
                    duration_sec=completed.duration_sec,
                    exit_code=completed.exit_code,
                    status=completed.status,
                    status_reason=completed.status_reason,
                    stdout_path=str(completed.launch.stdout_path),
                    stderr_path=str(completed.launch.stderr_path),
                    command=completed.launch.command,
                    executor=payload.executor,
                    selected_options=cfg.selected_options,
                )

                payload_record = record.to_json()
                append_jsonl(result_path, payload_record)
                attempt_history.setdefault(cfg.config_id, []).append(payload_record)
                _append_batch_event(
                    batch_log_path=batch_log_path,
                    event={
                        "event": "command_complete",
                        "time": completed.end_time,
                        "run_id": context.run_id,
                        "submission_id": submission_id,
                        "batch_index": cfg.batch_index,
                        "line_index": cfg.line_index,
                        "config_id": cfg.config_id,
                        "status": completed.status,
                        "status_reason": completed.status_reason,
                        "exit_code": completed.exit_code,
                        "attempt_id": completed.launch.attempt_id,
                        "attempt_index": attempt_index,
                        "parent_attempt_id": parent_attempt_id,
                        "job_id": context.job_id,
                        "array_task_id": context.array_task_id,
                        "pid": completed.pid,
                        "tmux_session": completed.tmux_session,
                        "assigned_cores": list(completed.assigned_cores),
                    },
                    context=context,
                    submission_id=submission_id,
                )
                command_complete_event = {
                    "event": "command_complete",
                    "time": completed.end_time,
                    "run_id": context.run_id,
                    "submission_id": submission_id,
                    "batch_index": cfg.batch_index,
                    "line_index": cfg.line_index,
                    "config_id": cfg.config_id,
                    "status": completed.status,
                    "status_reason": completed.status_reason,
                    "exit_code": completed.exit_code,
                    "attempt_id": completed.launch.attempt_id,
                    "attempt_index": attempt_index,
                    "parent_attempt_id": parent_attempt_id,
                    "duration_sec": completed.duration_sec,
                    "job_id": context.job_id,
                    "array_task_id": context.array_task_id,
                    "pid": completed.pid,
                    "tmux_session": completed.tmux_session,
                    "assigned_cores": list(completed.assigned_cores),
                }
                _append_task_event(
                    task_events_path=task_events_path,
                    event=command_complete_event,
                    context=context,
                    submission_id=submission_id,
                    progress_callback=progress_callback,
                )

                summary["total"] += 1
                if completed.status in summary:
                    summary[completed.status] += 1
                else:
                    _log.warning(
                        "Unexpected command status %r for config %s (batch %d, line %d)",
                        completed.status,
                        cfg.config_id,
                        cfg.batch_index,
                        cfg.line_index,
                    )
                    summary["failed"] += 1
                if completed.status_reason == "timeout_kill":
                    timeout_event = {
                        "event": "timeout_kill",
                        "time": completed.end_time,
                        "run_id": context.run_id,
                        "submission_id": submission_id,
                        "batch_index": cfg.batch_index,
                        "line_index": cfg.line_index,
                        "config_id": cfg.config_id,
                        "attempt_id": completed.launch.attempt_id,
                        "attempt_index": attempt_index,
                        "job_id": context.job_id,
                        "array_task_id": context.array_task_id,
                    }
                    _append_task_event(
                        task_events_path=task_events_path,
                        event=timeout_event,
                        context=context,
                        submission_id=submission_id,
                        progress_callback=progress_callback,
                    )

                retry_eligible = (
                    completed.status in retry_on_status
                    and retries_used_by_config[cfg.config_id] < payload.max_retries
                    and fail_fast_reason is None
                )

                if completed.status == "success":
                    previous_successes.add(cfg.config_id)
                    prev_status = terminal_status_by_config.get(cfg.config_id)
                    if prev_status is None:
                        terminal_total += 1
                    elif prev_status in {"failed", "terminated"}:
                        terminal_failed -= 1
                    terminal_status_by_config[cfg.config_id] = "success"
                    retry_eligible = False

                if retry_eligible:
                    retries_used_by_config[cfg.config_id] += 1
                    pending_launches.append(build_launch(cfg))
                    summary["retry_scheduled"] += 1
                    retry_event = {
                        "event": "retry_scheduled",
                        "time": utc_now_iso(),
                        "run_id": context.run_id,
                        "submission_id": submission_id,
                        "batch_index": cfg.batch_index,
                        "line_index": cfg.line_index,
                        "config_id": cfg.config_id,
                        "previous_attempt_id": completed.launch.attempt_id,
                        "next_attempt_index": attempt_index + 1,
                        "retry_count": retries_used_by_config[cfg.config_id],
                        "max_retries": payload.max_retries,
                        "status": completed.status,
                        "job_id": context.job_id,
                        "array_task_id": context.array_task_id,
                    }
                    _append_task_event(
                        task_events_path=task_events_path,
                        event=retry_event,
                        context=context,
                        submission_id=submission_id,
                        progress_callback=progress_callback,
                    )
                    _append_batch_event(
                        batch_log_path=batch_log_path,
                        event={
                            "event": "retry_scheduled",
                            "time": utc_now_iso(),
                            "run_id": context.run_id,
                            "submission_id": submission_id,
                            "batch_index": cfg.batch_index,
                            "line_index": cfg.line_index,
                            "config_id": cfg.config_id,
                            "retry_count": retries_used_by_config[cfg.config_id],
                            "max_retries": payload.max_retries,
                            "status": completed.status,
                        },
                        context=context,
                        submission_id=submission_id,
                    )
                    continue

                if completed.status != "success":
                    prev_status = terminal_status_by_config.get(cfg.config_id)
                    if prev_status is None:
                        terminal_total += 1
                        terminal_failed += 1
                    elif prev_status == "success":
                        terminal_failed += 1
                    terminal_status_by_config[cfg.config_id] = completed.status

                triggered = _check_fail_fast(
                    terminal_failed=terminal_failed,
                    terminal_total=terminal_total,
                    max_failures=payload.max_failures,
                    max_failure_rate=payload.max_failure_rate,
                )
                if triggered and fail_fast_reason is None:
                    fail_fast_reason = triggered
                    summary["fail_fast_triggered"] = True
                    summary["fail_fast_reason"] = triggered
                    fail_fast_event = {
                        "event": "fail_fast_stop",
                        "time": utc_now_iso(),
                        "run_id": context.run_id,
                        "submission_id": submission_id,
                        "reason": triggered,
                        "batch_index": cfg.batch_index,
                        "line_index": cfg.line_index,
                        "config_id": cfg.config_id,
                        "terminal_total": terminal_total,
                        "terminal_failed": terminal_failed,
                        "failure_rate": (terminal_failed / terminal_total)
                        if terminal_total
                        else 0.0,
                        "max_failures": payload.max_failures,
                        "max_failure_rate": payload.max_failure_rate,
                        "job_id": context.job_id,
                        "array_task_id": context.array_task_id,
                    }
                    _append_task_event(
                        task_events_path=task_events_path,
                        event=fail_fast_event,
                        context=context,
                        submission_id=submission_id,
                        progress_callback=progress_callback,
                    )
                    _append_batch_event(
                        batch_log_path=batch_log_path,
                        event={
                            "event": "fail_fast_stop",
                            "time": utc_now_iso(),
                            "run_id": context.run_id,
                            "submission_id": submission_id,
                            "reason": triggered,
                            "terminal_total": terminal_total,
                            "terminal_failed": terminal_failed,
                        },
                        context=context,
                        submission_id=submission_id,
                    )

            if fail_fast_reason is not None and pending_launches:
                summary["skipped_fail_fast"] += len(pending_launches)
                drop_event = {
                    "event": "fail_fast_drop_pending",
                    "time": utc_now_iso(),
                    "run_id": context.run_id,
                    "submission_id": submission_id,
                    "reason": fail_fast_reason,
                    "dropped_commands": len(pending_launches),
                    "wave_index": wave_index,
                    "batch_index": batch_index,
                    "job_id": context.job_id,
                    "array_task_id": context.array_task_id,
                }
                _append_batch_event(
                    batch_log_path=batch_log_path,
                    event=drop_event,
                    context=context,
                    submission_id=submission_id,
                )
                _append_task_event(
                    task_events_path=task_events_path,
                    event=drop_event,
                    context=context,
                    submission_id=submission_id,
                    progress_callback=progress_callback,
                )
                pending_launches = []

        batch_end_event = {
            "event": "batch_end",
            "time": utc_now_iso(),
            "run_id": context.run_id,
            "submission_id": submission_id,
            "batch_index": batch_index,
            "status": "fail_fast_stop" if fail_fast_reason is not None else "completed",
            "job_id": context.job_id,
            "array_task_id": context.array_task_id,
        }
        _append_batch_event(
            batch_log_path=batch_log_path,
            event=batch_end_event,
            context=context,
            submission_id=submission_id,
        )
        _append_task_event(
            task_events_path=task_events_path,
            event=batch_end_event,
            context=context,
            submission_id=submission_id,
            progress_callback=progress_callback,
        )


def _write_task_end(
    task_events_path: Path,
    summary: dict[str, Any],
    *,
    context: TaskContext,
    submission_id: str | None,
    executor_error: str | None = None,
    progress_callback: ProgressEventCallback | None = None,
) -> None:
    """Write the task_end event. Called from both the normal path and error recovery."""
    event: dict[str, Any] = {
        "event": "task_end",
        "time": utc_now_iso(),
        "summary": {
            "total": summary["total"],
            "success": summary["success"],
            "failed": summary["failed"],
            "terminated": summary["terminated"],
            "retry_scheduled": summary["retry_scheduled"],
            "skipped_fail_fast": summary["skipped_fail_fast"],
            "fail_fast_triggered": summary["fail_fast_triggered"],
            "fail_fast_reason": summary["fail_fast_reason"],
        },
    }
    if executor_error is not None:
        event["executor_error"] = executor_error
    try:
        _append_task_event(
            task_events_path=task_events_path,
            event=event,
            context=context,
            submission_id=submission_id,
            progress_callback=progress_callback,
        )
    except OSError:
        _log.error("Failed to write task_end event to %s", task_events_path)
