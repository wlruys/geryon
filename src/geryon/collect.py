from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geryon.models import ConfigError
from geryon.store import ArtifactStore
from geryon.utils import read_jsonl_with_stats, utc_now_iso


def _status_counter(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(record.get("status", "unknown") for record in records)
    return {key: int(value) for key, value in sorted(counter.items())}


def collect_run(
    run_dir: str | Path, *, dry_run: bool = False, strict_jsonl: bool = False
) -> dict[str, Any]:
    store = ArtifactStore.from_run_dir(run_dir)
    results_dir = store.results_dir
    result_files = sorted(results_dir.glob("task_*.jsonl"))

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

    latest_by_config: dict[str, dict[str, Any]] = {}
    for record in records:
        config_id = str(record.get("config_id", ""))
        if not config_id:
            continue
        previous = latest_by_config.get(config_id)
        if previous is None:
            latest_by_config[config_id] = record
            continue
        if str(record.get("end_time", "")) >= str(previous.get("end_time", "")):
            latest_by_config[config_id] = record

    final_counter = Counter(
        rec.get("status", "unknown") for rec in latest_by_config.values()
    )

    per_batch: dict[int, Counter[str]] = defaultdict(Counter)
    for record in records:
        batch_index = int(record.get("batch_index", -1))
        status = str(record.get("status", "unknown"))
        per_batch[batch_index][status] += 1

    summary = {
        "run_id": store.run_id,
        "run_root": str(store.run_root),
        "generated_at": utc_now_iso(),
        "num_result_files": len(result_files),
        "corrupt_result_lines": int(corrupt_result_lines),
        "result_files": [str(path) for path in result_files],
        "attempts": {
            "total": len(records),
            "by_status": _status_counter(records),
        },
        "configs": {
            "total": len(latest_by_config),
            "final_by_status": {
                key: int(value) for key, value in sorted(final_counter.items())
            },
            "failed_config_ids": sorted(
                [
                    config_id
                    for config_id, record in latest_by_config.items()
                    if record.get("status") in {"failed", "terminated"}
                ]
            ),
        },
        "batches": {
            "total": len(per_batch),
            "by_batch": {
                str(batch): {
                    status: int(count) for status, count in sorted(counter.items())
                }
                for batch, counter in sorted(per_batch.items())
            },
        },
    }

    if not dry_run:
        summary_path = store.results_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return summary
