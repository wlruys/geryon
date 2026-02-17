from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class GeryonError(RuntimeError):
    """Base error for launcher failures."""


class ConfigError(GeryonError):
    """Raised when experiment config is invalid."""


@dataclass(frozen=True)
class OptionSpec:
    pack_name: str
    index: int
    params: dict[str, Any]
    option_id: str
    tag: str | None = None
    priority: int = 0

    @property
    def label(self) -> str:
        return self.option_id

    @property
    def identifiers(self) -> set[str]:
        # Constraint matching is ID-only in schema v3.
        return {self.option_id}


@dataclass(frozen=True)
class PackSpec:
    name: str
    options: tuple[OptionSpec, ...]
    priority: int = 0


@dataclass(frozen=True)
class ExcludeConstraint:
    when: dict[str, tuple[str, ...]]
    mode: str = "exclude"


@dataclass(frozen=True)
class PredicateArgSpec:
    name: str
    source_kind: str  # pack_id | param
    source_key: str
    has_default: bool = False
    default_value: Any = None


@dataclass(frozen=True)
class ParameterPredicateSpec:
    predicate_id: str
    args: tuple[PredicateArgSpec, ...]
    expr: Any
    on_error: str = "error"  # error | false


@dataclass(frozen=True)
class ExperimentSpec:
    base_command: str
    packs: tuple[PackSpec, ...]
    defaults_params: dict[str, Any]
    defaults_tags: tuple[str, ...]
    constraints: tuple[ExcludeConstraint, ...]
    predicates: tuple[ParameterPredicateSpec, ...]
    merge_policy: "MergePolicy"


@dataclass(frozen=True)
class MergePolicy:
    mode: str = "merge"
    key_strategies: dict[str, str] = field(default_factory=dict)
    delete_sentinel: str = "__delete__"


@dataclass(frozen=True)
class PlannedConfig:
    run_id: str
    config_id: str
    batch_index: int
    line_index: int
    command: str
    params: dict[str, Any]
    tags: tuple[str, ...]
    wandb_name: str
    selected_options: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_id": self.config_id,
            "batch_index": self.batch_index,
            "line_index": self.line_index,
            "command": self.command,
            "params": self.params,
            "tags": list(self.tags),
            "wandb_name": self.wandb_name,
            "selected_options": self.selected_options,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PlannedConfig":
        return cls(
            run_id=str(payload["run_id"]),
            config_id=str(payload["config_id"]),
            batch_index=int(payload["batch_index"]),
            line_index=int(payload["line_index"]),
            command=str(payload["command"]),
            params=dict(payload["params"]),
            tags=tuple(payload.get("tags", [])),
            wandb_name=str(payload.get("wandb_name", "")),
            selected_options=dict(payload.get("selected_options", {})),
        )


@dataclass(frozen=True)
class PlannedBatch:
    batch_index: int
    path: str
    num_commands: int

    def to_json(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "path": self.path,
            "num_commands": self.num_commands,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PlannedBatch":
        return cls(
            batch_index=int(payload["batch_index"]),
            path=str(payload["path"]),
            num_commands=int(payload["num_commands"]),
        )


@dataclass(frozen=True)
class PlanSummary:
    run_id: str
    run_root: Path
    total_configs: int
    total_batches: int
    preview_configs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TaskContext:
    run_id: str
    run_root: Path
    job_id: str
    array_task_id: str
    hostname: str


SlurmParameterValue = str | int | float | bool


@dataclass(frozen=True)
class ResolvedSlurmConfig:
    partition: str
    time_min: int
    cpus_per_task: int
    mem_gb: int
    gpus_per_node: int
    job_name: str
    mail_user: str | None = None
    mail_type: str | None = None
    query_status: bool = False
    slurm_setup_cmds: tuple[str, ...] = ()
    slurm_additional_parameters: dict[str, SlurmParameterValue] = field(
        default_factory=dict
    )
    sources: dict[str, str] = field(default_factory=dict)
    sbatch_option_sources: dict[str, str] = field(default_factory=dict)
    slurm_setup_sources: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "time_min": self.time_min,
            "cpus_per_task": self.cpus_per_task,
            "mem_gb": self.mem_gb,
            "gpus_per_node": self.gpus_per_node,
            "job_name": self.job_name,
            "mail_user": self.mail_user,
            "mail_type": self.mail_type,
            "query_status": self.query_status,
            "slurm_setup_cmds": list(self.slurm_setup_cmds),
            "slurm_additional_parameters": dict(self.slurm_additional_parameters),
            "sources": dict(self.sources),
            "sbatch_option_sources": dict(self.sbatch_option_sources),
            "slurm_setup_sources": dict(self.slurm_setup_sources),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class TaskPayload:
    run_root: str
    batch_indices: tuple[int, ...]
    executor: str
    max_workers: int = 1
    k_per_session: int = 1
    max_total_cores: int | None = None
    cores: str | None = None
    tmux_prefix: str = "geryon"
    env_setup_script: str | None = None
    env_setup_commands: tuple[str, ...] = ()
    env_vars: dict[str, str] = field(default_factory=dict)
    include_config_ids: tuple[str, ...] = ()
    resume: bool = False
    command_timeout_sec: int | None = None
    max_retries: int = 0
    retry_on_status: tuple[str, ...] = ("failed", "terminated")
    max_failures: int | None = None
    max_failure_rate: float | None = None
    dry_run: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "run_root": self.run_root,
            "batch_indices": list(self.batch_indices),
            "executor": self.executor,
            "max_workers": self.max_workers,
            "k_per_session": self.k_per_session,
            "max_total_cores": self.max_total_cores,
            "cores": self.cores,
            "tmux_prefix": self.tmux_prefix,
            "env_setup_script": self.env_setup_script,
            "env_setup_commands": list(self.env_setup_commands),
            "env_vars": dict(self.env_vars),
            "include_config_ids": list(self.include_config_ids),
            "resume": self.resume,
            "command_timeout_sec": self.command_timeout_sec,
            "max_retries": self.max_retries,
            "retry_on_status": list(self.retry_on_status),
            "max_failures": self.max_failures,
            "max_failure_rate": self.max_failure_rate,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TaskPayload":
        command_timeout_raw = payload.get("command_timeout_sec")
        max_failures_raw = payload.get("max_failures")
        max_failure_rate_raw = payload.get("max_failure_rate")
        max_total_cores_raw = payload.get("max_total_cores")
        return cls(
            run_root=str(payload["run_root"]),
            batch_indices=tuple(int(x) for x in payload["batch_indices"]),
            executor=str(payload["executor"]),
            max_workers=int(payload.get("max_workers", 1)),
            k_per_session=int(payload.get("k_per_session", 1)),
            max_total_cores=(
                None if max_total_cores_raw is None else int(max_total_cores_raw)
            ),
            cores=payload.get("cores"),
            tmux_prefix=str(payload.get("tmux_prefix", "geryon")),
            env_setup_script=payload.get("env_setup_script"),
            env_setup_commands=tuple(
                str(x) for x in payload.get("env_setup_commands", [])
            ),
            env_vars={
                str(k): str(v) for k, v in dict(payload.get("env_vars", {})).items()
            },
            include_config_ids=tuple(
                str(x) for x in payload.get("include_config_ids", [])
            ),
            resume=bool(payload.get("resume", False)),
            command_timeout_sec=(
                None if command_timeout_raw is None else int(command_timeout_raw)
            ),
            max_retries=int(payload.get("max_retries", 0)),
            retry_on_status=tuple(
                str(x) for x in payload.get("retry_on_status", ["failed", "terminated"])
            ),
            max_failures=(None if max_failures_raw is None else int(max_failures_raw)),
            max_failure_rate=(
                None if max_failure_rate_raw is None else float(max_failure_rate_raw)
            ),
            dry_run=bool(payload.get("dry_run", False)),
        )


@dataclass
class AttemptRecord:
    run_id: str
    config_id: str
    batch_index: int
    line_index: int
    attempt_id: str
    attempt_index: int
    parent_attempt_id: str | None
    hostname: str
    pid: int | None
    tmux_session: str | None
    job_id: str
    array_task_id: str
    start_time: str
    end_time: str
    duration_sec: float
    exit_code: int | None
    status: str
    status_reason: str | None
    stdout_path: str
    stderr_path: str
    command: str
    executor: str
    selected_options: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_id": self.config_id,
            "batch_index": self.batch_index,
            "line_index": self.line_index,
            "attempt_id": self.attempt_id,
            "attempt_index": self.attempt_index,
            "parent_attempt_id": self.parent_attempt_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "tmux_session": self.tmux_session,
            "job_id": self.job_id,
            "array_task_id": self.array_task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "exit_code": self.exit_code,
            "status": self.status,
            "status_reason": self.status_reason,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "command": self.command,
            "executor": self.executor,
            "selected_options": self.selected_options,
        }
