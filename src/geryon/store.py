from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from geryon.models import ConfigError, PlannedBatch, PlannedConfig
from geryon.utils import append_jsonl, read_jsonl, sanitize_for_path, stable_json


class ArtifactStore:
    def __init__(self, run_root: Path):
        self.run_root = Path(run_root)
        self.run_id = self.run_root.name

    @property
    def plan_dir(self) -> Path:
        return self.run_root / "plan"

    @property
    def plan_batches_dir(self) -> Path:
        return self.plan_dir / "batches"

    @property
    def plan_configs_path(self) -> Path:
        return self.plan_dir / "configs.jsonl"

    @property
    def plan_batches_path(self) -> Path:
        return self.plan_dir / "batches.jsonl"

    @property
    def manifest_jsonl_path(self) -> Path:
        return self.plan_dir / "manifest.jsonl"

    @property
    def manifest_csv_path(self) -> Path:
        return self.plan_dir / "manifest.csv"

    @property
    def plan_diagnostics_path(self) -> Path:
        return self.plan_dir / "diagnostics.json"

    @property
    def plan_diagnostics_summary_path(self) -> Path:
        return self.plan_dir / "diagnostics.summary.txt"

    @property
    def plan_run_set_path(self) -> Path:
        return self.plan_dir / "run_set.json"

    @property
    def run_meta_path(self) -> Path:
        return self.run_root / "run.json"

    @property
    def exec_dir(self) -> Path:
        return self.run_root / "exec"

    @property
    def batch_logs_dir(self) -> Path:
        return self.exec_dir / "batch_logs"

    @property
    def cmd_logs_dir(self) -> Path:
        return self.exec_dir / "cmd_logs"

    @property
    def task_workdirs_dir(self) -> Path:
        return self.exec_dir / "workdirs"

    @property
    def results_dir(self) -> Path:
        return self.exec_dir / "results"

    @property
    def submitit_dir(self) -> Path:
        return self.exec_dir / "submitit"

    @property
    def retries_dir(self) -> Path:
        return self.exec_dir / "retries"

    @property
    def payloads_path(self) -> Path:
        return self.exec_dir / "task_payloads.jsonl"

    @property
    def payload_snapshots_dir(self) -> Path:
        return self.exec_dir / "task_payloads"

    @property
    def submitit_jobs_path(self) -> Path:
        return self.exec_dir / "submitit_jobs.json"

    @property
    def submitit_jobs_snapshots_dir(self) -> Path:
        return self.exec_dir / "submitit_jobs"

    @property
    def queue_status_path(self) -> Path:
        return self.exec_dir / "queue_status.json"

    @property
    def queue_status_snapshots_dir(self) -> Path:
        return self.exec_dir / "queue_status"

    @property
    def launcher_log_path(self) -> Path:
        return self.exec_dir / "launcher.log"

    def ensure_plan_layout(self) -> None:
        self.plan_batches_dir.mkdir(parents=True, exist_ok=True)

    def ensure_exec_layout(self) -> None:
        self.batch_logs_dir.mkdir(parents=True, exist_ok=True)
        self.cmd_logs_dir.mkdir(parents=True, exist_ok=True)
        self.task_workdirs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.submitit_dir.mkdir(parents=True, exist_ok=True)
        self.retries_dir.mkdir(parents=True, exist_ok=True)
        self.payload_snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.submitit_jobs_snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.queue_status_snapshots_dir.mkdir(parents=True, exist_ok=True)

    def batch_file(self, batch_index: int) -> Path:
        return self.plan_batches_dir / f"batch_{batch_index:03d}.txt"

    def task_result_file(self, job_id: str, array_task_id: str) -> Path:
        safe_job = sanitize_for_path(str(job_id))
        safe_task = sanitize_for_path(str(array_task_id))
        return self.results_dir / f"task_{safe_job}_{safe_task}.jsonl"

    def task_log_dir(self, job_id: str, array_task_id: str) -> Path:
        safe_job = sanitize_for_path(str(job_id))
        safe_task = sanitize_for_path(str(array_task_id))
        return self.batch_logs_dir / f"task_{safe_job}_{safe_task}"

    def task_stdout_log_file(self, job_id: str, array_task_id: str) -> Path:
        return self.task_log_dir(job_id, array_task_id) / "task.stdout.log"

    def task_stderr_log_file(self, job_id: str, array_task_id: str) -> Path:
        return self.task_log_dir(job_id, array_task_id) / "task.stderr.log"

    def task_events_file(self, job_id: str, array_task_id: str) -> Path:
        return self.task_log_dir(job_id, array_task_id) / "task.events.jsonl"

    def command_work_dir(
        self,
        job_id: str,
        array_task_id: str,
        *,
        batch_index: int,
        line_index: int,
        config_id: str,
        attempt_id: str,
    ) -> Path:
        safe_job = sanitize_for_path(str(job_id))
        safe_task = sanitize_for_path(str(array_task_id))
        safe_config = sanitize_for_path(str(config_id))
        safe_attempt = sanitize_for_path(str(attempt_id))
        return (
            self.task_workdirs_dir
            / f"task_{safe_job}_{safe_task}"
            / f"batch_{int(batch_index):03d}"
            / f"line_{int(line_index):04d}_{safe_config[:8]}_{safe_attempt[:8]}"
        )

    def write_run_meta(self, payload: Mapping[str, Any]) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.run_meta_path.write_text(
            stable_json(dict(payload)) + "\n", encoding="utf-8"
        )

    def read_run_meta(self) -> dict[str, Any]:
        if not self.run_meta_path.exists():
            raise ConfigError(
                f"Run metadata not found: {self.run_meta_path}. "
                f"Has 'plan' been run for this directory?"
            )
        try:
            return json.loads(self.run_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Corrupt run metadata at {self.run_meta_path}: {exc}"
            ) from exc

    def write_planned_batches(self, batches: Iterable[PlannedBatch]) -> None:
        self.plan_batches_path.parent.mkdir(parents=True, exist_ok=True)
        with self.plan_batches_path.open("w", encoding="utf-8") as handle:
            for batch in batches:
                handle.write(stable_json(batch.to_json()) + "\n")

    def write_planned_configs(self, configs: Iterable[PlannedConfig]) -> None:
        self.plan_configs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.plan_configs_path.open("w", encoding="utf-8") as handle:
            for cfg in configs:
                handle.write(stable_json(cfg.to_json()) + "\n")

    def append_payload(self, payload: Mapping[str, Any]) -> None:
        append_jsonl(self.payloads_path, payload)

    def write_manifest(self, configs: Iterable[PlannedConfig]) -> None:
        rows = []
        for cfg in configs:
            row = {
                "id": cfg.config_id,
                "wandb_name": cfg.wandb_name,
                "tags": list(cfg.tags),
                "params": cfg.params,
            }
            rows.append(row)

        with self.manifest_jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(stable_json(row) + "\n")

        with self.manifest_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["id", "wandb_name", "tags", "params"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "wandb_name": row["wandb_name"],
                        "tags": json.dumps(row["tags"], separators=(",", ":")),
                        "params": json.dumps(
                            row["params"], sort_keys=True, separators=(",", ":")
                        ),
                    }
                )

    def read_planned_configs(self) -> list[PlannedConfig]:
        if not self.plan_configs_path.exists():
            raise ConfigError(
                f"Planned configs not found: {self.plan_configs_path}. "
                f"Has 'plan' been run for this directory?"
            )
        return [
            PlannedConfig.from_json(item) for item in read_jsonl(self.plan_configs_path)
        ]

    def read_planned_batches(self) -> list[PlannedBatch]:
        if not self.plan_batches_path.exists():
            raise ConfigError(
                f"Planned batches not found: {self.plan_batches_path}. "
                f"Has 'plan' been run for this directory?"
            )
        return [
            PlannedBatch.from_json(item) for item in read_jsonl(self.plan_batches_path)
        ]

    def read_batch_commands(self, batch_index: int) -> list[str]:
        path = self.batch_file(batch_index)
        if not path.exists():
            raise ConfigError(f"Batch file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle]
        return [line for line in lines if line.strip()]

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "ArtifactStore":
        return cls(Path(run_dir).resolve())
