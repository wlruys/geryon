from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml

from geryon.models import ConfigError, SlurmParameterValue

DEFAULT_PROFILES_FILE = "profiles.yaml"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROFILE_ALLOWED_KEYS = {
    "partition",
    "time_min",
    "cpus_per_task",
    "mem_gb",
    "gpus_per_node",
    "job_name",
    "mail_user",
    "mail_type",
    "env_script",
    "env_setup_cmds",
    "slurm_setup_cmds",
    "slurm_additional_parameters",
    "env",
    "defaults",
}
_PROFILE_DEFAULTS_ALLOWED_KEYS = {"run_local", "run_slurm"}
_COMMON_RUN_DEFAULT_KEYS = {
    "executor",
    "max_concurrent_tasks",
    "cores_per_task",
    "max_total_cores",
    "cores",
    "batches_per_task",
    "command_timeout_sec",
    "max_retries",
    "retry_on_status",
    "max_failures",
    "fail_fast_threshold",
    "resume",
}
RUN_LOCAL_DEFAULT_ALLOWED_KEYS = {
    *_COMMON_RUN_DEFAULT_KEYS,
    "progress",
}
RUN_SLURM_DEFAULT_ALLOWED_KEYS = {
    *_COMMON_RUN_DEFAULT_KEYS,
    "slurm_setup_cmds",
    "sbatch_option",
    "partition",
    "time_min",
    "cpus_per_task",
    "mem_gb",
    "gpus_per_node",
    "job_name",
    "mail_user",
    "mail_type",
    "query_status",
}
_ALLOWED_RETRY_ON_STATUS = {"failed", "terminated"}
_ALLOWED_EXECUTORS = {"process", "tmux", "pylauncher"}


@dataclass(frozen=True)
class RunProfile:
    name: str
    partition: str | None = None
    time_min: int | None = None
    cpus_per_task: int | None = None
    mem_gb: int | None = None
    gpus_per_node: int | None = None
    job_name: str | None = None
    mail_user: str | None = None
    mail_type: str | None = None
    env_script: str | None = None
    env_setup_cmds: tuple[str, ...] = ()
    slurm_setup_cmds: tuple[str, ...] = ()
    slurm_additional_parameters: dict[str, SlurmParameterValue] = field(
        default_factory=dict
    )
    env: dict[str, str] = field(default_factory=dict)
    run_local_defaults: dict[str, Any] = field(default_factory=dict)
    run_slurm_defaults: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "partition": self.partition,
            "time_min": self.time_min,
            "cpus_per_task": self.cpus_per_task,
            "mem_gb": self.mem_gb,
            "gpus_per_node": self.gpus_per_node,
            "job_name": self.job_name,
            "mail_user": self.mail_user,
            "mail_type": self.mail_type,
            "env_script": self.env_script,
            "env_setup_cmds": list(self.env_setup_cmds),
            "slurm_setup_cmds": list(self.slurm_setup_cmds),
            "slurm_additional_parameters": dict(self.slurm_additional_parameters),
            "env": dict(self.env),
            "defaults": {
                "run_local": dict(self.run_local_defaults),
                "run_slurm": dict(self.run_slurm_defaults),
            },
        }


def default_profiles_path() -> Path:
    return Path(DEFAULT_PROFILES_FILE).expanduser().resolve()


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _coerce_optional_str(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string")
    trimmed = value.strip()
    return trimmed or None


def _coerce_optional_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return int(value)


def _coerce_required_int(value: Any, *, label: str) -> int:
    if value is None:
        raise ConfigError(f"{label} cannot be null")
    parsed = _coerce_optional_int(value, label=label)
    if parsed is None:
        raise ConfigError(f"{label} must be an integer")
    return parsed


def _coerce_optional_bool(value: Any, *, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a boolean")
    return value


def _coerce_optional_mail_type(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = ",".join(part.strip() for part in value.split(",") if part.strip())
        return cleaned or None
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(f"{label} must contain only strings")
            stripped = item.strip()
            if stripped:
                values.append(stripped)
        cleaned = ",".join(values)
        return cleaned or None
    raise ConfigError(f"{label} must be a string or list of strings")


def _coerce_str_list(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{label} must contain only strings")
        stripped = item.strip()
        if stripped:
            out.append(stripped)
    return tuple(out)


def _coerce_env_map(value: Any, *, label: str) -> dict[str, str]:
    if value is None:
        return {}
    raw = _require_mapping(value, label=label)
    env: dict[str, str] = {}
    for key, val in raw.items():
        env_key = str(key)
        if not _ENV_NAME_RE.match(env_key):
            raise ConfigError(
                f"{label}.{env_key} is not a valid environment variable name"
            )
        if not isinstance(val, (str, int, float, bool)):
            raise ConfigError(
                f"{label}.{key} must be a scalar value (str/int/float/bool)"
            )
        env[env_key] = str(val)
    return env


def _normalize_sbatch_option_key(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ConfigError(f"{label} must be a non-empty string")
    return cleaned.replace("-", "_")


def _coerce_slurm_additional_parameters(
    value: Any,
    *,
    label: str,
) -> dict[str, SlurmParameterValue]:
    if value is None:
        return {}
    raw = _require_mapping(value, label=label)
    parsed: dict[str, SlurmParameterValue] = {}
    for raw_key, raw_value in raw.items():
        key = _normalize_sbatch_option_key(raw_key, label=f"{label} key")
        if not isinstance(raw_value, (str, int, float, bool)):
            raise ConfigError(
                f"{label}.{raw_key} must be a scalar value (str/int/float/bool)"
            )
        parsed[key] = raw_value
    return parsed


def _coerce_executor(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string")
    text = value.strip()
    if text not in _ALLOWED_EXECUTORS:
        raise ConfigError(f"{label} must be one of {sorted(_ALLOWED_EXECUTORS)}")
    return text


def _coerce_retry_on_status_defaults(value: Any, *, label: str) -> list[str]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        raw_values = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(f"{label} must contain only strings")
            raw_values.extend(part.strip() for part in item.split(","))
    else:
        raise ConfigError(f"{label} must be a string or list of strings")

    resolved: list[str] = []
    for item in raw_values:
        if not item:
            continue
        if item not in _ALLOWED_RETRY_ON_STATUS:
            raise ConfigError(
                f"{label} contains unsupported status '{item}'. "
                f"Allowed: {sorted(_ALLOWED_RETRY_ON_STATUS)}"
            )
        if item not in resolved:
            resolved.append(item)
    if not resolved:
        raise ConfigError(f"{label} cannot be empty")
    return resolved


def _coerce_fail_fast_threshold(value: Any, *, label: str) -> float:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError(f"{label} cannot be empty")
        try:
            if text.endswith("%"):
                parsed = float(text[:-1]) / 100.0
            else:
                parsed = float(text)
                if parsed > 1.0:
                    parsed = parsed / 100.0
        except ValueError as exc:
            raise ConfigError(
                f"{label} must be decimal (0-1) or percent (e.g. 20%)"
            ) from exc
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if parsed > 1.0:
            parsed = parsed / 100.0
    else:
        raise ConfigError(f"{label} must be decimal (0-1) or percent string (e.g. 20%)")
    if not (0.0 < parsed <= 1.0):
        raise ConfigError(f"{label} must be in (0, 1] or (0%, 100%]")
    return parsed


def _coerce_run_local_defaults(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    raw = _require_mapping(value, label=label)
    unknown = sorted(set(raw.keys()) - RUN_LOCAL_DEFAULT_ALLOWED_KEYS)
    if unknown:
        raise ConfigError(
            f"{label} has unknown keys: {unknown}. "
            f"Allowed keys: {sorted(RUN_LOCAL_DEFAULT_ALLOWED_KEYS)}"
        )
    parsed: dict[str, Any] = {}
    if "executor" in raw:
        parsed["executor"] = _coerce_executor(
            raw["executor"], label=f"{label}.executor"
        )
    if "max_concurrent_tasks" in raw:
        parsed["max_concurrent_tasks"] = _coerce_required_int(
            raw["max_concurrent_tasks"], label=f"{label}.max_concurrent_tasks"
        )
    if "cores_per_task" in raw:
        parsed["cores_per_task"] = _coerce_required_int(
            raw["cores_per_task"], label=f"{label}.cores_per_task"
        )
    if "max_total_cores" in raw:
        parsed["max_total_cores"] = _coerce_optional_int(
            raw["max_total_cores"], label=f"{label}.max_total_cores"
        )
    if "cores" in raw:
        parsed["cores"] = _coerce_optional_str(raw["cores"], label=f"{label}.cores")
    if "batches_per_task" in raw:
        parsed["batches_per_task"] = _coerce_required_int(
            raw["batches_per_task"], label=f"{label}.batches_per_task"
        )
    if "command_timeout_sec" in raw:
        parsed["command_timeout_sec"] = _coerce_optional_int(
            raw["command_timeout_sec"], label=f"{label}.command_timeout_sec"
        )
    if "max_retries" in raw:
        parsed["max_retries"] = _coerce_required_int(
            raw["max_retries"], label=f"{label}.max_retries"
        )
    if "retry_on_status" in raw:
        parsed["retry_on_status"] = _coerce_retry_on_status_defaults(
            raw["retry_on_status"], label=f"{label}.retry_on_status"
        )
    if "max_failures" in raw:
        parsed["max_failures"] = _coerce_optional_int(
            raw["max_failures"], label=f"{label}.max_failures"
        )
    if "fail_fast_threshold" in raw:
        parsed["fail_fast_threshold"] = _coerce_fail_fast_threshold(
            raw["fail_fast_threshold"], label=f"{label}.fail_fast_threshold"
        )
    if "progress" in raw:
        parsed["progress"] = _coerce_optional_bool(
            raw["progress"], label=f"{label}.progress"
        )
    if "resume" in raw:
        parsed["resume"] = _coerce_optional_bool(raw["resume"], label=f"{label}.resume")
    return parsed


def _coerce_run_slurm_defaults(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    raw = _require_mapping(value, label=label)
    unknown = sorted(set(raw.keys()) - RUN_SLURM_DEFAULT_ALLOWED_KEYS)
    if unknown:
        raise ConfigError(
            f"{label} has unknown keys: {unknown}. "
            f"Allowed keys: {sorted(RUN_SLURM_DEFAULT_ALLOWED_KEYS)}"
        )
    parsed = _coerce_run_local_defaults(
        {key: value for key, value in raw.items() if key in _COMMON_RUN_DEFAULT_KEYS},
        label=label,
    )
    if "partition" in raw:
        parsed["partition"] = _coerce_optional_str(
            raw["partition"], label=f"{label}.partition"
        )
    if "time_min" in raw:
        parsed["time_min"] = _coerce_optional_int(
            raw["time_min"], label=f"{label}.time_min"
        )
    if "cpus_per_task" in raw:
        parsed["cpus_per_task"] = _coerce_optional_int(
            raw["cpus_per_task"], label=f"{label}.cpus_per_task"
        )
    if "mem_gb" in raw:
        parsed["mem_gb"] = _coerce_optional_int(raw["mem_gb"], label=f"{label}.mem_gb")
    if "gpus_per_node" in raw:
        parsed["gpus_per_node"] = _coerce_optional_int(
            raw["gpus_per_node"], label=f"{label}.gpus_per_node"
        )
    if "job_name" in raw:
        parsed["job_name"] = _coerce_optional_str(
            raw["job_name"], label=f"{label}.job_name"
        )
    if "mail_user" in raw:
        parsed["mail_user"] = _coerce_optional_str(
            raw["mail_user"], label=f"{label}.mail_user"
        )
    if "mail_type" in raw:
        parsed["mail_type"] = _coerce_optional_mail_type(
            raw["mail_type"], label=f"{label}.mail_type"
        )
    if "query_status" in raw:
        parsed["query_status"] = _coerce_optional_bool(
            raw["query_status"], label=f"{label}.query_status"
        )
    if "slurm_setup_cmds" in raw:
        parsed["slurm_setup_cmds"] = list(
            _coerce_str_list(raw["slurm_setup_cmds"], label=f"{label}.slurm_setup_cmds")
        )
    if "sbatch_option" in raw:
        parsed["sbatch_option"] = _coerce_slurm_additional_parameters(
            raw["sbatch_option"], label=f"{label}.sbatch_option"
        )
    return parsed


def _parse_profile(name: str, payload: Any) -> RunProfile:
    data = _require_mapping(payload, label=f"profiles.{name}")
    unknown = sorted(set(data.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ConfigError(
            f"profiles.{name} has unknown keys: {unknown}. "
            f"Allowed keys: {sorted(PROFILE_ALLOWED_KEYS)}"
        )

    defaults_raw = data.get("defaults")
    if defaults_raw is None:
        defaults = {}
    else:
        defaults = _require_mapping(defaults_raw, label=f"profiles.{name}.defaults")
    defaults_unknown = sorted(set(defaults.keys()) - _PROFILE_DEFAULTS_ALLOWED_KEYS)
    if defaults_unknown:
        raise ConfigError(
            f"profiles.{name}.defaults has unknown keys: {defaults_unknown}. "
            f"Allowed keys: {sorted(_PROFILE_DEFAULTS_ALLOWED_KEYS)}"
        )

    return RunProfile(
        name=name,
        partition=_coerce_optional_str(
            data.get("partition"), label=f"profiles.{name}.partition"
        ),
        time_min=_coerce_optional_int(
            data.get("time_min"), label=f"profiles.{name}.time_min"
        ),
        cpus_per_task=_coerce_optional_int(
            data.get("cpus_per_task"),
            label=f"profiles.{name}.cpus_per_task",
        ),
        mem_gb=_coerce_optional_int(
            data.get("mem_gb"), label=f"profiles.{name}.mem_gb"
        ),
        gpus_per_node=_coerce_optional_int(
            data.get("gpus_per_node"),
            label=f"profiles.{name}.gpus_per_node",
        ),
        job_name=_coerce_optional_str(
            data.get("job_name"), label=f"profiles.{name}.job_name"
        ),
        mail_user=_coerce_optional_str(
            data.get("mail_user"), label=f"profiles.{name}.mail_user"
        ),
        mail_type=_coerce_optional_mail_type(
            data.get("mail_type"), label=f"profiles.{name}.mail_type"
        ),
        env_script=_coerce_optional_str(
            data.get("env_script"), label=f"profiles.{name}.env_script"
        ),
        env_setup_cmds=_coerce_str_list(
            data.get("env_setup_cmds"),
            label=f"profiles.{name}.env_setup_cmds",
        ),
        slurm_setup_cmds=_coerce_str_list(
            data.get("slurm_setup_cmds"),
            label=f"profiles.{name}.slurm_setup_cmds",
        ),
        slurm_additional_parameters=_coerce_slurm_additional_parameters(
            data.get("slurm_additional_parameters"),
            label=f"profiles.{name}.slurm_additional_parameters",
        ),
        env=_coerce_env_map(data.get("env"), label=f"profiles.{name}.env"),
        run_local_defaults=_coerce_run_local_defaults(
            defaults.get("run_local"),
            label=f"profiles.{name}.defaults.run_local",
        ),
        run_slurm_defaults=_coerce_run_slurm_defaults(
            defaults.get("run_slurm"),
            label=f"profiles.{name}.defaults.run_slurm",
        ),
    )


def load_profiles(
    path: str | Path | None, *, required: bool
) -> tuple[Path, dict[str, RunProfile]]:
    resolved_path = (
        default_profiles_path() if path is None else Path(path).expanduser().resolve()
    )
    if not resolved_path.exists():
        if required:
            raise ConfigError(f"Profiles file not found: {resolved_path}")
        return resolved_path, {}

    loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Profiles file root must be a mapping: {resolved_path}")

    raw_profiles = loaded.get("profiles", {})
    if raw_profiles is None:
        raw_profiles = {}
    if not isinstance(raw_profiles, dict):
        raise ConfigError(f"'profiles' must be a mapping in {resolved_path}")

    parsed: dict[str, RunProfile] = {}
    for name in sorted(raw_profiles.keys()):
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(
                f"Profile names must be non-empty strings: {resolved_path}"
            )
        trimmed = name.strip()
        if trimmed in parsed:
            raise ConfigError(f"Duplicate profile name: {trimmed}")
        profile = _parse_profile(trimmed, raw_profiles[name])
        env_script = profile.env_script
        if env_script:
            script_path = Path(env_script).expanduser()
            if not script_path.is_absolute():
                env_script = str((resolved_path.parent / script_path).resolve())
            else:
                env_script = str(script_path.resolve())
            profile = RunProfile(
                name=profile.name,
                partition=profile.partition,
                time_min=profile.time_min,
                cpus_per_task=profile.cpus_per_task,
                mem_gb=profile.mem_gb,
                gpus_per_node=profile.gpus_per_node,
                job_name=profile.job_name,
                mail_user=profile.mail_user,
                mail_type=profile.mail_type,
                env_script=env_script,
                env_setup_cmds=profile.env_setup_cmds,
                slurm_setup_cmds=profile.slurm_setup_cmds,
                slurm_additional_parameters=profile.slurm_additional_parameters,
                env=profile.env,
                run_local_defaults=profile.run_local_defaults,
                run_slurm_defaults=profile.run_slurm_defaults,
            )
        parsed[trimmed] = profile
    return resolved_path, parsed


def resolve_profile(
    *,
    profile_name: str | None,
    profiles_file: str | Path | None,
) -> tuple[Path, RunProfile | None]:
    required = profile_name is not None
    resolved_path, profiles = load_profiles(profiles_file, required=required)
    if profile_name is None:
        return resolved_path, None
    profile = profiles.get(profile_name)
    if profile is None:
        available = ", ".join(sorted(profiles.keys())) or "<none>"
        raise ConfigError(
            f"Unknown profile '{profile_name}' in {resolved_path}. Available: {available}"
        )
    return resolved_path, profile


def merge_profile_env(
    profile: RunProfile | None,
    *,
    cli_env_script: str | None,
    cli_env_setup_cmds: list[str],
    cli_env_vars: dict[str, str],
) -> tuple[str | None, list[str], dict[str, str]]:
    env_script = (
        cli_env_script if cli_env_script else (profile.env_script if profile else None)
    )
    env_setup_cmds: list[str] = []
    if profile:
        env_setup_cmds.extend(profile.env_setup_cmds)
    env_setup_cmds.extend([cmd for cmd in cli_env_setup_cmds if cmd.strip()])

    merged_env: dict[str, str] = {}
    if profile:
        merged_env.update(profile.env)
    merged_env.update(cli_env_vars)
    return env_script, env_setup_cmds, merged_env
