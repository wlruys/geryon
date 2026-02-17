from __future__ import annotations

import contextlib
import json
import logging
import os
import traceback
import uuid
from typing import Any

from geryon.execution import execute_task_payload
from geryon.models import ResolvedSlurmConfig, TaskPayload
from geryon.store import ArtifactStore
from geryon.utils import append_jsonl, sanitize_for_path, stable_json, utc_now_iso

_log = logging.getLogger("geryon.submitit")


def _load_submitit() -> Any:
    try:
        import submitit  # type: ignore
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised in environments without submitit
        raise RuntimeError(
            "submitit is required for run-slurm. Install with: uv sync --extra slurm"
        ) from exc
    return submitit


def _extract_submitit_job_state(job: Any) -> str | None:
    for attr in ("state", "status"):
        try:
            raw = getattr(job, attr)
        except Exception:
            continue
        try:
            value = raw() if callable(raw) else raw
        except Exception:
            continue
        text = str(value).strip()
        if text and text.lower() != "none":
            return text
    return None


def run_batch_task(payload_json: dict[str, Any]) -> dict[str, Any]:
    payload = TaskPayload.from_json(payload_json)
    store = ArtifactStore.from_run_dir(payload.run_root)
    store.ensure_exec_layout()
    run_id = str(payload_json.get("run_id", "")).strip()
    if not run_id:
        if store.run_meta_path.exists():
            try:
                meta = store.read_run_meta()
                run_id = str(meta.get("run_id", "")).strip()
            except Exception:
                run_id = ""
    if not run_id:
        run_id = store.run_id
    submission_id_raw = str(payload_json.get("submission_id", "")).strip()
    submission_id = submission_id_raw or None

    job_id = os.environ.get("SLURM_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    try:
        submitit = _load_submitit()
        env = submitit.JobEnvironment()
        job_id = job_id or str(env.job_id)
        array_candidate = getattr(env, "array_task_id", None)
        if array_task_id is None or str(array_task_id).strip() == "":
            if array_candidate is not None and str(array_candidate).strip():
                array_task_id = str(array_candidate)
        array_task_id = array_task_id or "0"
    except Exception as env_exc:
        _log.warning(
            "Could not resolve submitit JobEnvironment, falling back to "
            "job_id='local': %s",
            env_exc,
        )
        job_id = job_id or "local"
        array_task_id = array_task_id or "0"

    task_stdout = store.task_stdout_log_file(job_id, array_task_id)
    task_stderr = store.task_stderr_log_file(job_id, array_task_id)
    task_events = store.task_events_file(job_id, array_task_id)
    task_stdout.parent.mkdir(parents=True, exist_ok=True)

    with (
        task_stdout.open("a", encoding="utf-8") as stdout_handle,
        task_stderr.open("a", encoding="utf-8") as stderr_handle,
    ):
        with (
            contextlib.redirect_stdout(stdout_handle),
            contextlib.redirect_stderr(stderr_handle),
        ):
            append_jsonl(
                task_events,
                {
                    "event": "submitit_task_entry",
                    "time": utc_now_iso(),
                    "run_id": run_id,
                    "submission_id": submission_id,
                    "job_id": job_id,
                    "array_task_id": array_task_id,
                    "payload_batches": list(payload.batch_indices),
                },
            )
            try:
                return execute_task_payload(
                    payload,
                    job_id=job_id,
                    array_task_id=array_task_id,
                    submission_id=submission_id,
                )
            except Exception as exc:
                _log.error(
                    "Task failed (job_id=%s, array_task_id=%s): %s",
                    job_id,
                    array_task_id,
                    exc,
                    exc_info=True,
                )
                try:
                    append_jsonl(
                        task_events,
                        {
                            "event": "submitit_task_exception",
                            "time": utc_now_iso(),
                            "run_id": run_id,
                            "submission_id": submission_id,
                            "job_id": job_id,
                            "array_task_id": array_task_id,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                except OSError as log_exc:
                    _log.error(
                        "Failed to write task exception event to %s: %s",
                        task_events,
                        log_exc,
                    )
                raise


def submit_payloads(
    *,
    store: ArtifactStore,
    payloads: list[TaskPayload],
    slurm_config: ResolvedSlurmConfig,
    dry_run: bool,
    profile_name: str | None = None,
    profiles_file: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    store.ensure_exec_layout()

    stamp = sanitize_for_path(utc_now_iso().replace(":", "").replace(".", ""))
    nonce = uuid.uuid4().hex[:8]
    snapshot_id = f"{stamp}_{nonce}"
    resolved_run_id = str(run_id or "").strip() or store.run_id

    payload_records = [
        {
            **payload.to_json(),
            "run_id": resolved_run_id,
            "submission_id": snapshot_id,
        }
        for payload in payloads
    ]
    with store.payloads_path.open("w", encoding="utf-8") as handle:
        for record in payload_records:
            handle.write(stable_json(record) + "\n")
    payload_snapshot_path = (
        store.payload_snapshots_dir / f"task_payloads_{snapshot_id}.jsonl"
    )
    with payload_snapshot_path.open("w", encoding="utf-8") as handle:
        for record in payload_records:
            handle.write(stable_json(record) + "\n")

    resolved_slurm_additional: dict[str, str | int | float | bool] = {}
    resolved_slurm_additional.update(slurm_config.slurm_additional_parameters)
    if slurm_config.mail_user:
        resolved_slurm_additional["mail_user"] = str(slurm_config.mail_user)
    if slurm_config.mail_type:
        resolved_slurm_additional["mail_type"] = str(slurm_config.mail_type)

    if dry_run:
        return {
            "submitted": False,
            "dry_run": True,
            "run_id": resolved_run_id,
            "submission_id": snapshot_id,
            "num_payloads": len(payloads),
            "payloads_path": str(store.payloads_path),
            "payloads_snapshot_path": str(payload_snapshot_path),
            "profile_name": profile_name,
            "profiles_file": profiles_file,
            "partition": slurm_config.partition,
            "time_min": slurm_config.time_min,
            "cpus_per_task": slurm_config.cpus_per_task,
            "gpus_per_node": slurm_config.gpus_per_node,
            "mem_gb": slurm_config.mem_gb,
            "job_name": slurm_config.job_name,
            "mail_user": slurm_config.mail_user,
            "mail_type": slurm_config.mail_type,
            "slurm_additional_parameters": resolved_slurm_additional,
            "slurm_setup": list(slurm_config.slurm_setup_cmds),
        }

    submitit = _load_submitit()

    executor = submitit.AutoExecutor(folder=str(store.submitit_dir))
    submitit_params: dict[str, Any] = dict(
        timeout_min=slurm_config.time_min,
        slurm_partition=slurm_config.partition,
        tasks_per_node=1,
        cpus_per_task=slurm_config.cpus_per_task,
        gpus_per_node=slurm_config.gpus_per_node,
        mem_gb=slurm_config.mem_gb,
        name=slurm_config.job_name,
    )
    if slurm_config.slurm_setup_cmds:
        submitit_params["slurm_setup"] = list(slurm_config.slurm_setup_cmds)
    if resolved_slurm_additional:
        submitit_params["slurm_additional_parameters"] = resolved_slurm_additional
    executor.update_parameters(**submitit_params)

    try:
        jobs = executor.map_array(run_batch_task, payload_records)
    except Exception as exc:
        raise RuntimeError(
            f"Slurm job submission failed (partition={slurm_config.partition}, "
            f"num_payloads={len(payload_records)}): {exc}"
        ) from exc
    job_ids = [str(job.job_id) for job in jobs]
    submitit_states: list[dict[str, str]] = []
    for job in jobs:
        state = _extract_submitit_job_state(job)
        if state is None:
            continue
        submitit_states.append({"job_id": str(job.job_id), "state": state})

    metadata = {
        "submitted": True,
        "submitted_at": utc_now_iso(),
        "run_id": resolved_run_id,
        "submission_id": snapshot_id,
        "num_payloads": len(payload_records),
        "job_ids": job_ids,
        "partition": slurm_config.partition,
        "time_min": slurm_config.time_min,
        "cpus_per_task": slurm_config.cpus_per_task,
        "gpus_per_node": slurm_config.gpus_per_node,
        "mem_gb": slurm_config.mem_gb,
        "job_name": slurm_config.job_name,
        "mail_user": slurm_config.mail_user,
        "mail_type": slurm_config.mail_type,
        "query_status": slurm_config.query_status,
        "slurm_additional_parameters": resolved_slurm_additional,
        "slurm_setup": list(slurm_config.slurm_setup_cmds),
        "profile_name": profile_name,
        "profiles_file": profiles_file,
        "payloads_path": str(store.payloads_path),
        "payloads_snapshot_path": str(payload_snapshot_path),
    }
    if submitit_states:
        metadata["submitit_states"] = submitit_states
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    store.submitit_jobs_path.write_text(metadata_text, encoding="utf-8")
    submitit_snapshot_path = (
        store.submitit_jobs_snapshots_dir / f"submitit_jobs_{snapshot_id}.json"
    )
    submitit_snapshot_path.write_text(metadata_text, encoding="utf-8")
    return {
        **metadata,
        "submitit_jobs_path": str(store.submitit_jobs_path),
        "submitit_jobs_snapshot_path": str(submitit_snapshot_path),
    }
