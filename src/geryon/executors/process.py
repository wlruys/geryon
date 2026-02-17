from __future__ import annotations

from collections import deque
import datetime as dt
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Iterator, TextIO

from geryon.executors.affinity import (
    CoreAllocator,
    build_affinity_preexec,
    default_thread_env,
    discover_core_pool,
)
from geryon.executors.base import CommandLaunch, CompletedCommand, ExecutorStartCallback

_log = logging.getLogger("geryon.executors.process")


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class _ProcessState:
    launch: CommandLaunch
    process: subprocess.Popen[str]
    start_time: str
    start_perf: float
    stdout_handle: TextIO
    stderr_handle: TextIO
    assigned_cores: list[int]
    timed_out: bool = False
    timeout_kill_started_perf: float | None = None


class ProcessBatchExecutor:
    name = "process"
    _TIMEOUT_KILL_GRACE_SEC = 10.0

    def __init__(
        self,
        *,
        max_workers: int = 1,
        k_per_worker: int = 1,
        cores: str | None = None,
        command_timeout_sec: int | None = None,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if k_per_worker < 1:
            raise ValueError("k_per_worker must be >= 1")

        allocator = CoreAllocator(
            core_pool=discover_core_pool(cores),
            cores_per_job=k_per_worker,
        )
        self.k_per_worker = k_per_worker
        self.max_parallel = min(max_workers, allocator.max_parallel)
        self.core_allocator = allocator
        if command_timeout_sec is not None and command_timeout_sec <= 0:
            raise ValueError("command_timeout_sec must be > 0 when provided")
        self.command_timeout_sec = command_timeout_sec

    def _signal_process_tree(self, process: subprocess.Popen[str], signum: int) -> None:
        if process.poll() is not None:
            return

        if os.name == "posix":
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signum)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass

        try:
            process.send_signal(signum)
        except ProcessLookupError:
            return

    def _enforce_timeout_kill(self, state: _ProcessState) -> None:
        process = state.process
        if process.poll() is not None:
            return

        self._signal_process_tree(process, signal.SIGTERM)
        try:
            process.wait(timeout=0.5)
            return
        except subprocess.TimeoutExpired:
            pass

        self._signal_process_tree(process, signal.SIGKILL)
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass

    def _cleanup_failed_start(self, state: _ProcessState) -> None:
        self._enforce_timeout_kill(state)
        for handle in (state.stdout_handle, state.stderr_handle):
            try:
                handle.close()
            except OSError:
                pass
        self.core_allocator.push(state.assigned_cores)

    def run(
        self,
        launches: list[CommandLaunch],
        on_start: ExecutorStartCallback | None = None,
    ) -> Iterator[CompletedCommand]:
        running: list[_ProcessState] = []
        pending = deque(launches)

        while pending or running:
            while (
                pending
                and len(running) < self.max_parallel
                and self.core_allocator.can_allocate()
            ):
                launch = pending.popleft()
                assigned_cores = self.core_allocator.pop()
                if not assigned_cores:
                    break
                launch.stdout_path.parent.mkdir(parents=True, exist_ok=True)
                launch.stderr_path.parent.mkdir(parents=True, exist_ok=True)

                stdout_handle = launch.stdout_path.open("w", encoding="utf-8")
                stderr_handle = launch.stderr_path.open("w", encoding="utf-8")
                start_time = _now_iso()
                start_perf = time.perf_counter()

                child_env = os.environ.copy()
                child_env.update(default_thread_env(assigned_cores, base_env=child_env))
                preexec_fn = build_affinity_preexec(assigned_cores)

                try:
                    process = subprocess.Popen(
                        ["bash", "-lc", launch.command],
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        env=child_env,
                        preexec_fn=preexec_fn,
                        start_new_session=True,
                    )
                except Exception as exc:
                    stdout_handle.close()
                    stderr_handle.close()
                    self.core_allocator.push(assigned_cores)
                    raise RuntimeError(
                        f"Failed to spawn process for config "
                        f"{launch.config.config_id} "
                        f"(batch {launch.config.batch_index}, "
                        f"line {launch.config.line_index}): {exc}"
                    ) from exc
                state = _ProcessState(
                    launch=launch,
                    process=process,
                    start_time=start_time,
                    start_perf=start_perf,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                    assigned_cores=assigned_cores,
                )
                if on_start is not None:
                    try:
                        on_start(
                            launch,
                            {
                                "pid": process.pid,
                                "assigned_cores": list(assigned_cores),
                                "tmux_session": None,
                                "executor": self.name,
                            },
                        )
                    except Exception:
                        self._cleanup_failed_start(state)
                        raise
                running.append(state)

            if not running:
                continue

            next_running: list[_ProcessState] = []
            for state in running:
                rc = state.process.poll()
                if rc is None and self.command_timeout_sec is not None:
                    elapsed = time.perf_counter() - state.start_perf
                    if elapsed > float(self.command_timeout_sec):
                        if not state.timed_out:
                            state.timed_out = True
                            state.timeout_kill_started_perf = time.perf_counter()
                            self._enforce_timeout_kill(state)
                        elif state.timeout_kill_started_perf is not None:
                            kill_elapsed = (
                                time.perf_counter() - state.timeout_kill_started_perf
                            )
                            if kill_elapsed > self._TIMEOUT_KILL_GRACE_SEC:
                                raise RuntimeError(
                                    f"Process pid={state.process.pid} did not exit after timeout kill "
                                    f"for config {state.launch.config.config_id} "
                                    f"(batch {state.launch.config.batch_index}, "
                                    f"line {state.launch.config.line_index})."
                                )
                        rc = state.process.poll()
                if rc is None:
                    next_running.append(state)
                    continue

                state.stdout_handle.close()
                state.stderr_handle.close()
                self.core_allocator.push(state.assigned_cores)

                end_time = _now_iso()
                duration = time.perf_counter() - state.start_perf
                status = "success" if rc == 0 else "failed"
                if rc < 0:
                    status = "terminated"

                yield CompletedCommand(
                    launch=state.launch,
                    pid=state.process.pid,
                    tmux_session=None,
                    start_time=state.start_time,
                    end_time=end_time,
                    duration_sec=round(duration, 6),
                    exit_code=rc,
                    status=status,
                    status_reason="timeout_kill" if state.timed_out else None,
                    assigned_cores=tuple(state.assigned_cores),
                )

            running = next_running
            if running:
                time.sleep(0.05)
