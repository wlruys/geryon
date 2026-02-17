from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geryon.models import ConfigError

TASK_EVENT_SCHEMA_VERSION = 1
BATCH_EVENT_SCHEMA_VERSION = 1

_COMMON_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "event",
    "time",
    "run_id",
    "job_id",
    "array_task_id",
    "submission_id",
)

_EVENT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "task_start": ("executor", "batches", "resume", "selected_config_ids", "policy"),
    "batch_start": ("batch_index", "num_commands", "executor"),
    "batch_filter": (
        "batch_index",
        "planned_commands",
        "selected_commands",
        "skipped_not_selected",
        "skipped_resume",
    ),
    "batch_launch": ("batch_index", "wave_index", "num_commands"),
    "command_start": (
        "batch_index",
        "line_index",
        "config_id",
        "attempt_id",
        "wave_index",
        "executor",
    ),
    "command_complete": (
        "batch_index",
        "line_index",
        "config_id",
        "status",
        "attempt_id",
        "attempt_index",
        "duration_sec",
    ),
    "timeout_kill": (
        "batch_index",
        "line_index",
        "config_id",
        "attempt_id",
        "attempt_index",
    ),
    "retry_scheduled": (
        "batch_index",
        "line_index",
        "config_id",
        "previous_attempt_id",
        "next_attempt_index",
        "retry_count",
        "max_retries",
        "status",
    ),
    "fail_fast_stop": (
        "reason",
        "batch_index",
        "line_index",
        "config_id",
        "terminal_total",
        "terminal_failed",
        "failure_rate",
    ),
    "fail_fast_drop_pending": (
        "reason",
        "dropped_commands",
        "wave_index",
        "batch_index",
    ),
    "batch_end": ("batch_index", "status"),
    "batch_skipped": ("batch_index", "skipped_not_selected", "skipped_resume"),
    "batch_skipped_fail_fast": ("batch_index", "num_commands", "reason"),
    "executor_error": ("error",),
    "task_end": ("summary",),
}

_BATCH_COMMON_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "event",
    "time",
    "run_id",
    "job_id",
    "array_task_id",
    "submission_id",
)

_BATCH_EVENT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "batch_start": ("batch_index", "num_commands", "executor"),
    "batch_filter": (
        "batch_index",
        "planned_commands",
        "selected_commands",
        "skipped_not_selected",
        "skipped_resume",
    ),
    "command_start": (
        "batch_index",
        "line_index",
        "config_id",
        "attempt_id",
        "wave_index",
        "executor",
    ),
    "command_complete": (
        "batch_index",
        "line_index",
        "config_id",
        "status",
        "attempt_id",
        "attempt_index",
    ),
    "retry_scheduled": (
        "batch_index",
        "line_index",
        "config_id",
        "retry_count",
        "max_retries",
        "status",
    ),
    "fail_fast_stop": ("reason", "terminal_total", "terminal_failed"),
    "fail_fast_drop_pending": (
        "reason",
        "dropped_commands",
        "wave_index",
        "batch_index",
    ),
    "batch_end": ("batch_index", "status"),
}


def validate_task_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    event_name = payload.get("event")
    if not isinstance(event_name, str) or not event_name:
        raise ConfigError("Task event is missing a non-empty 'event' field")

    required_fields = _EVENT_REQUIRED_FIELDS.get(event_name)
    if required_fields is None:
        raise ConfigError(f"Unsupported task event '{event_name}'")

    schema_version = payload.get("schema_version")
    if schema_version != TASK_EVENT_SCHEMA_VERSION:
        raise ConfigError(
            f"Task event '{event_name}' must use schema_version="
            f"{TASK_EVENT_SCHEMA_VERSION}, got {schema_version!r}"
        )

    missing = [field for field in _COMMON_REQUIRED_FIELDS if field not in payload]
    missing.extend(field for field in required_fields if field not in payload)
    if missing:
        raise ConfigError(
            f"Task event '{event_name}' is missing required fields: {sorted(missing)}"
        )

    if not isinstance(payload.get("time"), str) or not str(payload["time"]).strip():
        raise ConfigError(f"Task event '{event_name}' has invalid 'time'")
    if not isinstance(payload.get("run_id"), str) or not str(payload["run_id"]).strip():
        raise ConfigError(f"Task event '{event_name}' has invalid 'run_id'")
    if not isinstance(payload.get("job_id"), str) or not str(payload["job_id"]).strip():
        raise ConfigError(f"Task event '{event_name}' has invalid 'job_id'")
    if (
        not isinstance(payload.get("array_task_id"), str)
        or not str(payload["array_task_id"]).strip()
    ):
        raise ConfigError(f"Task event '{event_name}' has invalid 'array_task_id'")
    submission_id = payload.get("submission_id")
    if submission_id is not None and not isinstance(submission_id, str):
        raise ConfigError(
            f"Task event '{event_name}' has invalid 'submission_id': {submission_id!r}"
        )
    if event_name == "task_end" and not isinstance(payload.get("summary"), Mapping):
        raise ConfigError("Task event 'task_end' requires 'summary' to be an object")

    return payload


def validate_batch_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    event_name = payload.get("event")
    if not isinstance(event_name, str) or not event_name:
        raise ConfigError("Batch event is missing a non-empty 'event' field")

    required_fields = _BATCH_EVENT_REQUIRED_FIELDS.get(event_name)
    if required_fields is None:
        raise ConfigError(f"Unsupported batch event '{event_name}'")

    schema_version = payload.get("schema_version")
    if schema_version != BATCH_EVENT_SCHEMA_VERSION:
        raise ConfigError(
            f"Batch event '{event_name}' must use schema_version="
            f"{BATCH_EVENT_SCHEMA_VERSION}, got {schema_version!r}"
        )

    missing = [field for field in _BATCH_COMMON_REQUIRED_FIELDS if field not in payload]
    missing.extend(field for field in required_fields if field not in payload)
    if missing:
        raise ConfigError(
            f"Batch event '{event_name}' is missing required fields: {sorted(missing)}"
        )

    if not isinstance(payload.get("time"), str) or not str(payload["time"]).strip():
        raise ConfigError(f"Batch event '{event_name}' has invalid 'time'")
    if not isinstance(payload.get("run_id"), str) or not str(payload["run_id"]).strip():
        raise ConfigError(f"Batch event '{event_name}' has invalid 'run_id'")
    if not isinstance(payload.get("job_id"), str) or not str(payload["job_id"]).strip():
        raise ConfigError(f"Batch event '{event_name}' has invalid 'job_id'")
    if (
        not isinstance(payload.get("array_task_id"), str)
        or not str(payload["array_task_id"]).strip()
    ):
        raise ConfigError(f"Batch event '{event_name}' has invalid 'array_task_id'")
    submission_id = payload.get("submission_id")
    if submission_id is not None and not isinstance(submission_id, str):
        raise ConfigError(
            f"Batch event '{event_name}' has invalid 'submission_id': {submission_id!r}"
        )
    return payload
