from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from geryon.models import PlannedConfig


@dataclass(frozen=True)
class CommandLaunch:
    config: PlannedConfig
    command: str
    stdout_path: Path
    stderr_path: Path
    attempt_id: str


@dataclass(frozen=True)
class CompletedCommand:
    launch: CommandLaunch
    pid: int | None
    tmux_session: str | None
    start_time: str
    end_time: str
    duration_sec: float
    exit_code: int | None
    status: str
    status_reason: str | None = None
    assigned_cores: tuple[int, ...] = ()


ExecutorStartCallback = Callable[[CommandLaunch, dict[str, Any]], None]


class BatchExecutor(Protocol):
    name: str

    def run(
        self,
        launches: list[CommandLaunch],
        on_start: ExecutorStartCallback | None = None,
    ) -> Iterator[CompletedCommand]: ...
