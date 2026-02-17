from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from geryon.cli import main
from geryon.models import ConfigError
from geryon.planner import plan_experiment
from geryon.task_event_schema import (
    BATCH_EVENT_SCHEMA_VERSION,
    TASK_EVENT_SCHEMA_VERSION,
    validate_batch_event,
    validate_task_event,
)
from geryon.utils import read_jsonl


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _flush_geryon_handlers() -> None:
    logger = logging.getLogger("geryon")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()


def test_validate_task_event_rejects_missing_required_fields() -> None:
    bad_event = {
        "schema_version": TASK_EVENT_SCHEMA_VERSION,
        "event": "command_start",
        "time": "2026-01-01T00:00:00Z",
        "run_id": "run1",
        "job_id": "local",
        "array_task_id": "0",
        "submission_id": None,
        "batch_index": 0,
        "line_index": 0,
        "config_id": "cfg",
        "wave_index": 1,
        "executor": "process",
    }
    with pytest.raises(ConfigError, match="missing required fields"):
        validate_task_event(bad_event)


def test_validate_batch_event_rejects_missing_required_fields() -> None:
    bad_event = {
        "schema_version": BATCH_EVENT_SCHEMA_VERSION,
        "event": "command_complete",
        "time": "2026-01-01T00:00:00Z",
        "run_id": "run1",
        "job_id": "local",
        "array_task_id": "0",
        "submission_id": None,
        "batch_index": 0,
        "line_index": 0,
        "config_id": "cfg",
        "attempt_id": "attempt1",
    }
    with pytest.raises(ConfigError, match="missing required fields"):
        validate_batch_event(bad_event)


def test_run_local_writes_schema_versioned_task_and_batch_events(
    tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python3
  args: ["-c", "print('event-schema')"]
select:
  packs:
    - name: seed
      options:
        - id: s1
          params:
            seed: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="event_schema",
        batch_size=1,
    )

    rc = main(["run-local", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    task_events_path = Path(payload["payloads"][0]["task_events_path"])
    records = list(read_jsonl(task_events_path))
    assert records
    assert {item["schema_version"] for item in records} == {TASK_EVENT_SCHEMA_VERSION}

    expected_common = {
        "schema_version",
        "event",
        "time",
        "run_id",
        "job_id",
        "array_task_id",
        "submission_id",
    }
    for record in records:
        assert expected_common.issubset(record.keys())
        validated = validate_task_event(record)
        assert validated["schema_version"] == TASK_EVENT_SCHEMA_VERSION

    observed = {item["event"] for item in records}
    assert {"task_start", "command_start", "command_complete", "task_end"} <= observed

    batch_dir = task_events_path.parent
    batch_files = sorted(batch_dir.glob("batch_*.jsonl"))
    assert batch_files
    for batch_file in batch_files:
        batch_records = list(read_jsonl(batch_file))
        assert batch_records
        assert {item["schema_version"] for item in batch_records} == {
            BATCH_EVENT_SCHEMA_VERSION
        }
        for record in batch_records:
            validated = validate_batch_event(record)
            assert validated["schema_version"] == BATCH_EVENT_SCHEMA_VERSION


def test_log_file_includes_cli_command_lifecycle(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "geryon.log"
    profiles_path = tmp_path / "profiles.yaml"
    _write(
        profiles_path,
        """
profiles: {}
""",
    )
    monkeypatch.setenv("GERYON_LOG_FILE", str(log_path))
    monkeypatch.delenv("GERYON_LOG_LEVEL", raising=False)

    rc = main(
        [
            "list-profiles",
            "--profiles-file",
            str(profiles_path),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)
    _flush_geryon_handlers()
    log_text = log_path.read_text(encoding="utf-8")
    assert "cli_command_start" in log_text
    assert "command=list-profiles" in log_text
    assert "cli_command_end" in log_text
