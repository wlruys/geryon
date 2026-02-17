from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from geryon.utils import parse_cpulist

_THREAD_LIMIT_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _supports_sched_setaffinity() -> bool:
    return _is_linux() and hasattr(os, "sched_setaffinity")


def _sched_setaffinity(pid: int, mask: set[int] | frozenset[int]) -> None:
    setter = getattr(os, "sched_setaffinity", None)
    if setter is None:
        raise OSError("sched_setaffinity is not available on this platform")
    cast(Callable[[int, set[int] | frozenset[int]], None], setter)(pid, mask)


def supports_linux_affinity() -> bool:
    return _supports_sched_setaffinity()


def has_taskset() -> bool:
    return _is_linux() and shutil.which("taskset") is not None


def discover_core_pool(explicit_cores: str | None) -> list[int]:
    for cpulist in (explicit_cores, os.environ.get("CORES")):
        if cpulist:
            values = parse_cpulist(cpulist)
            if not values:
                raise ValueError("CPU core list resolved to an empty set")
            return values

    if _is_linux() and hasattr(os, "sched_getaffinity"):
        try:
            values = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
            if values:
                return values
        except OSError:
            pass

    cpu_count = os.cpu_count() or 1
    return list(range(cpu_count))


def core_csv(cores: Sequence[int]) -> str:
    return ",".join(str(core) for core in cores)


class CoreAllocator:
    def __init__(self, *, core_pool: list[int], cores_per_job: int):
        if cores_per_job < 1:
            raise ValueError("cores_per_job must be >= 1")
        if len(core_pool) < cores_per_job:
            raise RuntimeError(
                f"CPU pool has {len(core_pool)} cores; need at least {cores_per_job}"
            )

        self.core_pool = list(core_pool)
        self.cores_per_job = cores_per_job
        self.max_parallel = max(1, len(self.core_pool) // self.cores_per_job)
        self._available = list(self.core_pool)

    def can_allocate(self) -> bool:
        return len(self._available) >= self.cores_per_job

    def pop(self) -> list[int]:
        if not self.can_allocate():
            return []
        picked = self._available[: self.cores_per_job]
        self._available = self._available[self.cores_per_job :]
        return picked

    def push(self, values: list[int]) -> None:
        self._available.extend(values)


def default_thread_env(
    assigned_cores: Sequence[int], *, base_env: Mapping[str, str]
) -> dict[str, str]:
    if not assigned_cores:
        return {}

    threads = str(max(1, len(assigned_cores)))
    updates: dict[str, str] = {}
    for key in _THREAD_LIMIT_ENV_KEYS:
        if key not in base_env:
            updates[key] = threads

    if "OMP_PROC_BIND" not in base_env:
        updates["OMP_PROC_BIND"] = "true"

    updates["GERYON_ASSIGNED_CORES"] = core_csv(assigned_cores)
    updates["GERYON_ASSIGNED_CORE_COUNT"] = threads
    return updates


def build_affinity_preexec(assigned_cores: Sequence[int]) -> Callable[[], None] | None:
    if not assigned_cores or not _supports_sched_setaffinity():
        return None

    mask = frozenset(int(cpu) for cpu in assigned_cores)

    def _set_affinity() -> None:
        try:
            _sched_setaffinity(0, mask)
        except OSError:
            pass

    return _set_affinity


def set_pid_affinity(pid: int | None, assigned_cores: Sequence[int]) -> bool:
    if pid is None or not assigned_cores or not _supports_sched_setaffinity():
        return False
    try:
        _sched_setaffinity(pid, set(assigned_cores))
        return True
    except OSError:
        return False
