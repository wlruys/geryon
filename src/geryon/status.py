from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geryon.models import ConfigError, PlannedConfig
from geryon.store import ArtifactStore
from geryon.utils import read_jsonl_with_stats, utc_now_iso

RERUN_STATUS_FILTERS = {"failed", "terminated", "missing"}


@dataclass(frozen=True)
class ConfigExecutionStatus:
    config_id: str
    batch_indices: tuple[int, ...]
    line_indices: tuple[int, ...]
    selected_options: dict[str, str]
    attempts: int
    latest_status: str
    latest_attempt_id: str | None
    has_success: bool  # True when the latest attempt status is success.


@dataclass(frozen=True)
class RunStatusIndex:
    run_id: str
    run_root: Path
    generated_at: str
    result_files: tuple[str, ...]
    attempt_records: tuple[dict[str, Any], ...]
    planned_configs: tuple[PlannedConfig, ...]
    planned_by_batch: dict[int, tuple[PlannedConfig, ...]]
    config_statuses: tuple[ConfigExecutionStatus, ...]
    config_status_by_id: dict[str, ConfigExecutionStatus]
    successful_config_ids: frozenset[str]
    orphan_attempt_config_ids: tuple[str, ...]
    corrupt_result_lines: int


def _attempt_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("end_time", "")),
        str(record.get("start_time", "")),
        str(record.get("attempt_id", "")),
    )


def _load_attempt_records(
    store: ArtifactStore, *, strict_jsonl: bool
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...], int]:
    result_files = sorted(store.results_dir.glob("task_*.jsonl"))
    records: list[dict[str, Any]] = []
    corrupt_result_lines = 0
    for path in result_files:
        try:
            file_records, corrupt_lines = read_jsonl_with_stats(
                path, strict=strict_jsonl
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        records.extend(file_records)
        corrupt_result_lines += corrupt_lines
    return (
        tuple(str(path) for path in result_files),
        tuple(records),
        corrupt_result_lines,
    )


def _group_planned_configs(
    planned_configs: list[PlannedConfig],
) -> dict[str, list[PlannedConfig]]:
    grouped: dict[str, list[PlannedConfig]] = defaultdict(list)
    for cfg in planned_configs:
        grouped[cfg.config_id].append(cfg)
    return grouped


def successful_config_ids(store_or_run: ArtifactStore | str | Path) -> set[str]:
    store = (
        store_or_run
        if isinstance(store_or_run, ArtifactStore)
        else ArtifactStore.from_run_dir(store_or_run)
    )
    _, records, _ = _load_attempt_records(store, strict_jsonl=False)

    attempts_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        config_id = str(record.get("config_id", ""))
        if not config_id:
            continue
        attempts_by_config[config_id].append(record)

    successful: set[str] = set()
    for config_id, attempts in attempts_by_config.items():
        attempts.sort(key=_attempt_sort_key)
        latest_status = str(attempts[-1].get("status", ""))
        if latest_status == "success":
            successful.add(config_id)
    return successful


def build_run_status_index(
    run_dir: str | Path, *, strict_jsonl: bool = False
) -> RunStatusIndex:
    store = ArtifactStore.from_run_dir(run_dir)
    planned_configs = sorted(
        store.read_planned_configs(),
        key=lambda cfg: (cfg.batch_index, cfg.line_index, cfg.config_id),
    )
    result_files, attempt_records, corrupt_result_lines = _load_attempt_records(
        store, strict_jsonl=strict_jsonl
    )

    attempts_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in attempt_records:
        config_id = str(record.get("config_id", ""))
        if not config_id:
            continue
        attempts_by_config[config_id].append(record)

    for records in attempts_by_config.values():
        records.sort(key=_attempt_sort_key)

    planned_grouped = _group_planned_configs(planned_configs)
    planned_order = [
        cfg.config_id for cfg in planned_configs if cfg.config_id in planned_grouped
    ]
    unique_order: list[str] = []
    seen: set[str] = set()
    for config_id in planned_order:
        if config_id in seen:
            continue
        seen.add(config_id)
        unique_order.append(config_id)

    config_statuses: list[ConfigExecutionStatus] = []
    config_status_by_id: dict[str, ConfigExecutionStatus] = {}
    for config_id in unique_order:
        planned_entries = planned_grouped[config_id]
        attempts = attempts_by_config.get(config_id, [])
        latest = attempts[-1] if attempts else None
        latest_status = str(latest.get("status", "missing")) if latest else "missing"
        latest_attempt_id = (
            None if latest is None else str(latest.get("attempt_id", "")) or None
        )
        latest_success = latest_status == "success"

        status = ConfigExecutionStatus(
            config_id=config_id,
            batch_indices=tuple(sorted({cfg.batch_index for cfg in planned_entries})),
            line_indices=tuple(sorted({cfg.line_index for cfg in planned_entries})),
            selected_options=dict(planned_entries[0].selected_options),
            attempts=len(attempts),
            latest_status=latest_status,
            latest_attempt_id=latest_attempt_id,
            has_success=latest_success,
        )
        config_statuses.append(status)
        config_status_by_id[config_id] = status

    successful_ids = {
        status.config_id for status in config_statuses if status.has_success
    }

    planned_set = set(unique_order)
    orphan_config_ids = sorted(
        {
            str(record.get("config_id", ""))
            for record in attempt_records
            if str(record.get("config_id", ""))
            and str(record.get("config_id", "")) not in planned_set
        }
    )

    planned_by_batch: dict[int, tuple[PlannedConfig, ...]] = {}
    grouped_by_batch: dict[int, list[PlannedConfig]] = defaultdict(list)
    for cfg in planned_configs:
        grouped_by_batch[cfg.batch_index].append(cfg)
    for batch_index, configs in grouped_by_batch.items():
        planned_by_batch[batch_index] = tuple(configs)

    run_meta = (
        store.read_run_meta()
        if store.run_meta_path.exists()
        else {"run_id": store.run_id}
    )
    run_id = str(run_meta.get("run_id", store.run_id))
    return RunStatusIndex(
        run_id=run_id,
        run_root=store.run_root,
        generated_at=utc_now_iso(),
        result_files=result_files,
        attempt_records=attempt_records,
        planned_configs=tuple(planned_configs),
        planned_by_batch=planned_by_batch,
        config_statuses=tuple(config_statuses),
        config_status_by_id=config_status_by_id,
        successful_config_ids=frozenset(successful_ids),
        orphan_attempt_config_ids=tuple(orphan_config_ids),
        corrupt_result_lines=corrupt_result_lines,
    )


def summarize_run_status(index: RunStatusIndex) -> dict[str, Any]:
    attempt_counter = Counter(
        str(record.get("status", "unknown")) for record in index.attempt_records
    )
    latest_counter = Counter(status.latest_status for status in index.config_statuses)
    completion_counter = Counter(
        "complete" if status.has_success else "pending"
        for status in index.config_statuses
    )

    failed_ids = sorted(
        [
            status.config_id
            for status in index.config_statuses
            if not status.has_success and status.latest_status == "failed"
        ]
    )
    terminated_ids = sorted(
        [
            status.config_id
            for status in index.config_statuses
            if not status.has_success and status.latest_status == "terminated"
        ]
    )
    missing_ids = sorted(
        [
            status.config_id
            for status in index.config_statuses
            if not status.has_success and status.latest_status == "missing"
        ]
    )
    unfinished_ids = sorted(
        [status.config_id for status in index.config_statuses if not status.has_success]
    )

    per_batch: dict[str, dict[str, int]] = {}
    unfinished_batches: list[int] = []
    for batch_index, configs in sorted(index.planned_by_batch.items()):
        counter: Counter[str] = Counter()
        for cfg in configs:
            status = index.config_status_by_id[cfg.config_id]
            counter["planned"] += 1
            if status.has_success:
                counter["complete"] += 1
            else:
                counter["pending"] += 1
            counter[f"latest_{status.latest_status}"] += 1

        if counter.get("pending", 0) > 0:
            unfinished_batches.append(batch_index)
        per_batch[str(batch_index)] = {
            key: int(value) for key, value in sorted(counter.items())
        }

    return {
        "run_id": index.run_id,
        "run_root": str(index.run_root),
        "generated_at": index.generated_at,
        "corrupt_result_lines": int(index.corrupt_result_lines),
        "result_files": list(index.result_files),
        "attempts": {
            "total": len(index.attempt_records),
            "by_status": {
                key: int(value) for key, value in sorted(attempt_counter.items())
            },
        },
        "configs": {
            "total": len(index.config_statuses),
            "latest_by_status": {
                key: int(value) for key, value in sorted(latest_counter.items())
            },
            "completion": {
                key: int(value) for key, value in sorted(completion_counter.items())
            },
            "failed_config_ids": failed_ids,
            "terminated_config_ids": terminated_ids,
            "missing_config_ids": missing_ids,
            "unfinished_config_ids": unfinished_ids,
        },
        "batches": {
            "total": len(index.planned_by_batch),
            "unfinished_batch_indices": unfinished_batches,
            "by_batch": per_batch,
        },
        "orphan_attempt_config_ids": list(index.orphan_attempt_config_ids),
    }


def summarize_status_groups(
    index: RunStatusIndex, *, by_packs: list[str]
) -> dict[str, Any]:
    packs = [pack.strip() for pack in by_packs if pack.strip()]
    if not packs:
        return {"packs": [], "groups": []}

    available_packs = sorted(
        {
            key
            for status in index.config_statuses
            for key in status.selected_options.keys()
        }
    )
    missing = [pack for pack in packs if pack not in available_packs]
    if missing:
        raise ConfigError(
            f"Unknown pack(s) for grouping: {missing}. "
            f"Available packs: {available_packs}"
        )

    grouped: dict[tuple[tuple[str, str], ...], list[ConfigExecutionStatus]] = (
        defaultdict(list)
    )
    for status in index.config_statuses:
        group_key = tuple(
            (pack, status.selected_options.get(pack, "<missing>")) for pack in packs
        )
        grouped[group_key].append(status)

    groups_payload: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        values = grouped[key]
        latest_counter = Counter(item.latest_status for item in values)
        complete = sum(1 for item in values if item.has_success)
        pending = len(values) - complete
        unfinished_ids = sorted(
            item.config_id for item in values if not item.has_success
        )
        groups_payload.append(
            {
                "key": {pack: value for pack, value in key},
                "total": len(values),
                "complete": complete,
                "pending": pending,
                "latest_by_status": {
                    name: int(count) for name, count in sorted(latest_counter.items())
                },
                "unfinished_config_ids": unfinished_ids,
            }
        )

    return {
        "packs": packs,
        "groups": groups_payload,
    }


def build_run_report_payload(
    index: RunStatusIndex, *, by_packs: list[str]
) -> dict[str, Any]:
    summary = summarize_run_status(index)
    grouped = summarize_status_groups(index, by_packs=by_packs)
    return {
        "run_id": summary["run_id"],
        "run_root": summary["run_root"],
        "generated_at": summary["generated_at"],
        "summary": summary,
        "grouped_by_packs": grouped,
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_report_markdown(report: dict[str, Any]) -> str:
    summary = dict(report.get("summary", {}))
    configs = dict(summary.get("configs", {}))
    attempts = dict(summary.get("attempts", {}))
    grouped = dict(report.get("grouped_by_packs", {}))
    grouped_rows = list(grouped.get("groups", []))
    grouped_packs = list(grouped.get("packs", []))

    lines = [
        "# Geryon Report",
        "",
        f"- Run ID: `{report.get('run_id', '')}`",
        f"- Run Root: `{report.get('run_root', '')}`",
        f"- Generated: `{report.get('generated_at', '')}`",
        "",
        "## Totals",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Configs", str(configs.get("total", 0))],
                ["Attempts", str(attempts.get("total", 0))],
                ["Corrupt JSONL Lines", str(summary.get("corrupt_result_lines", 0))],
                [
                    "Complete Configs",
                    str(dict(configs.get("completion", {})).get("complete", 0)),
                ],
                [
                    "Pending Configs",
                    str(dict(configs.get("completion", {})).get("pending", 0)),
                ],
            ],
        ),
        "",
        "## Latest Config Status",
        "",
    ]

    latest = dict(configs.get("latest_by_status", {}))
    latest_rows = [[name, str(count)] for name, count in sorted(latest.items())]
    if not latest_rows:
        latest_rows = [["<none>", "0"]]
    lines.append(_markdown_table(["Status", "Count"], latest_rows))
    lines.append("")

    unfinished = list(configs.get("unfinished_config_ids", []))
    lines.extend(["## Unfinished Config IDs", ""])
    if unfinished:
        for config_id in unfinished[:64]:
            lines.append(f"- `{config_id}`")
        if len(unfinished) > 64:
            lines.append(f"- ... and {len(unfinished) - 64} more")
    else:
        lines.append("- none")
    lines.append("")

    if grouped_packs:
        lines.extend(
            [
                "## Grouped Status",
                "",
                f"Grouped by: `{', '.join(grouped_packs)}`",
                "",
            ]
        )
        rows: list[list[str]] = []
        for item in grouped_rows:
            key = dict(item.get("key", {}))
            label = ", ".join(
                f"{pack}={key.get(pack, '<missing>')}" for pack in grouped_packs
            )
            rows.append(
                [
                    label,
                    str(item.get("total", 0)),
                    str(item.get("complete", 0)),
                    str(item.get("pending", 0)),
                    json.dumps(item.get("latest_by_status", {}), sort_keys=True),
                ]
            )
        if not rows:
            rows = [["<none>", "0", "0", "0", "{}"]]
        lines.append(
            _markdown_table(
                ["Group", "Total", "Complete", "Pending", "Latest By Status"],
                rows,
            )
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def select_rerun_config_ids(
    index: RunStatusIndex,
    *,
    status_filter: str | None,
    explicit_config_ids: list[str] | None,
) -> list[str]:
    requested_status = status_filter.strip() if status_filter else None
    if requested_status is not None and requested_status not in RERUN_STATUS_FILTERS:
        raise ConfigError(
            f"Unsupported rerun status filter '{requested_status}'. "
            f"Expected one of {sorted(RERUN_STATUS_FILTERS)}."
        )

    explicit = list(dict.fromkeys(explicit_config_ids or []))
    unknown = sorted(
        config_id
        for config_id in explicit
        if config_id not in index.config_status_by_id
    )
    if unknown:
        raise ConfigError(f"Unknown config IDs for rerun: {unknown}")

    target_ids: set[str] = set(explicit)
    if requested_status:
        for status in index.config_statuses:
            if status.has_success:
                continue
            if status.latest_status == requested_status:
                target_ids.add(status.config_id)

    ordered = [
        status.config_id
        for status in index.config_statuses
        if status.config_id in target_ids
    ]
    return ordered
