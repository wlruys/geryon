from __future__ import annotations

from collections import deque
import datetime as dt
import logging
import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from geryon.executors.affinity import (
    CoreAllocator,
    core_csv,
    default_thread_env,
    discover_core_pool,
    has_taskset,
    set_pid_affinity,
    supports_linux_affinity,
)
from geryon.executors.base import CommandLaunch, CompletedCommand, ExecutorStartCallback
from geryon.utils import sanitize_for_path

_log = logging.getLogger("geryon.executors.tmux")


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class _TmuxState:
    launch: CommandLaunch
    session_name: str
    pane_pid: int | None
    rc_path: Path
    wrapper_path: Path
    start_time: str
    start_perf: float
    assigned_cores: list[int]
    timed_out: bool = False
    timeout_kill_started_perf: float | None = None


class TmuxBatchExecutor:
    name = "tmux"
    _TIMEOUT_KILL_GRACE_SEC = 10.0

    def __init__(
        self,
        *,
        state_dir: Path,
        k_per_session: int = 1,
        max_parallel_tasks: int | None = None,
        cores: str | None = None,
        prefix: str = "geryon",
        command_timeout_sec: int | None = None,
    ):
        if k_per_session < 1:
            raise ValueError("k_per_session must be >= 1")
        if max_parallel_tasks is not None and max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be >= 1 when provided")

        if shutil.which("tmux") is None:
            raise RuntimeError("tmux is not installed or not available on PATH")

        self.k_per_session = k_per_session
        allocator = CoreAllocator(
            core_pool=discover_core_pool(cores),
            cores_per_job=k_per_session,
        )
        resolved_parallel = allocator.max_parallel
        if max_parallel_tasks is not None:
            resolved_parallel = min(resolved_parallel, max_parallel_tasks)
        self.max_parallel = resolved_parallel
        self.core_allocator = allocator
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = sanitize_for_path(prefix)
        if command_timeout_sec is not None and command_timeout_sec <= 0:
            raise ValueError("command_timeout_sec must be >= 1 when provided")
        self.command_timeout_sec = command_timeout_sec

    def _pop_cores(self) -> list[int]:
        return self.core_allocator.pop()

    def _push_cores(self, values: list[int]) -> None:
        self.core_allocator.push(values)

    def _session_exists(self, name: str) -> bool:
        probe = subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return probe.returncode == 0

    def _pane_pid(self, session_name: str) -> int | None:
        pane = subprocess.run(
            ["tmux", "display-message", "-p", "-t", session_name, "#{pane_pid}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if pane.returncode != 0:
            return None
        raw = pane.stdout.strip()
        if not raw.isdigit():
            return None
        return int(raw)

    def _signal_pid_tree(self, pid: int, signum: int) -> bool:
        if pid <= 0:
            return False

        if os.name == "posix":
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signum)
                return True
            except ProcessLookupError:
                return False
            except OSError:
                pass

        try:
            os.kill(pid, signum)
            return True
        except OSError:
            return False

    def _terminate_session_on_timeout(self, state: _TmuxState) -> bool:
        kill_result = subprocess.run(
            ["tmux", "kill-session", "-t", state.session_name],
            check=False,
            capture_output=True,
            text=True,
        )
        if kill_result.returncode != 0 and self._session_exists(state.session_name):
            _log.warning(
                "tmux kill-session failed for '%s': %s",
                state.session_name,
                kill_result.stderr.strip(),
            )

        if not self._session_exists(state.session_name):
            return True

        pane_pid = self._pane_pid(state.session_name) or state.pane_pid
        if pane_pid is None:
            return False
        state.pane_pid = pane_pid

        self._signal_pid_tree(pane_pid, signal.SIGTERM)
        time.sleep(0.2)
        if self._session_exists(state.session_name):
            self._signal_pid_tree(pane_pid, signal.SIGKILL)
            time.sleep(0.2)
            subprocess.run(
                ["tmux", "kill-session", "-t", state.session_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        return not self._session_exists(state.session_name)

    def _cleanup_failed_start(self, state: _TmuxState) -> None:
        self._terminate_session_on_timeout(state)
        state.rc_path.unlink(missing_ok=True)
        state.wrapper_path.unlink(missing_ok=True)

    def _launch_session(
        self, launch: CommandLaunch, assigned_cores: list[int]
    ) -> _TmuxState:
        launch.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        launch.stderr_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = f"b{launch.config.batch_index:03d}_l{launch.config.line_index:04d}_{launch.attempt_id[:6]}"
        session_name = sanitize_for_path(f"{self.prefix}_{suffix}")[:180]

        wrapper_path = self.state_dir / f"run_{session_name}.sh"
        rc_path = self.state_dir / f"run_{session_name}.rc"
        core_values_csv = core_csv(assigned_cores)

        cmd = f"bash -lc {shlex.quote(launch.command)}"
        if core_values_csv:
            if has_taskset():
                cmd = f"taskset -c {core_values_csv} {cmd}"
            elif supports_linux_affinity():
                affinity_program = "\n".join(
                    [
                        "import os",
                        "import subprocess",
                        "import sys",
                        f"cores = {{{core_values_csv}}}",
                        "try:",
                        "    os.sched_setaffinity(0, cores)",
                        "except OSError:",
                        "    pass",
                        "raise SystemExit(subprocess.call(['bash', '-lc', sys.argv[1]]))",
                    ]
                )
                cmd = (
                    f"{shlex.quote(sys.executable)} -c {shlex.quote(affinity_program)} "
                    f"{shlex.quote(launch.command)}"
                )

        child_env = default_thread_env(assigned_cores, base_env=os.environ)

        script_lines = [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            "set +e",
            *[
                f"export {name}={shlex.quote(value)}"
                for name, value in sorted(child_env.items())
            ],
            f"{cmd} > {shlex.quote(str(launch.stdout_path))} 2> {shlex.quote(str(launch.stderr_path))}",
            "rc=$?",
            f"printf '%s\\n' \"$rc\" > {shlex.quote(str(rc_path))}",
            "exit 0",
        ]
        wrapper_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
        wrapper_path.chmod(0o755)

        try:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name, str(wrapper_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"tmux new-session failed for session '{session_name}' "
                f"(config {launch.config.config_id}, "
                f"batch {launch.config.batch_index}, "
                f"line {launch.config.line_index}): {exc.stderr.strip() or exc}"
            ) from exc

        pane_pid = self._pane_pid(session_name)
        if pane_pid is not None:
            set_pid_affinity(pane_pid, assigned_cores)

        return _TmuxState(
            launch=launch,
            session_name=session_name,
            pane_pid=pane_pid,
            rc_path=rc_path,
            wrapper_path=wrapper_path,
            start_time=_now_iso(),
            start_perf=time.perf_counter(),
            assigned_cores=assigned_cores,
        )

    def run(
        self,
        launches: list[CommandLaunch],
        on_start: ExecutorStartCallback | None = None,
    ) -> Iterator[CompletedCommand]:
        running: list[_TmuxState] = []
        pending = deque(launches)

        while pending or running:
            while (
                pending
                and len(running) < self.max_parallel
                and self.core_allocator.can_allocate()
            ):
                launch = pending.popleft()
                assigned = self._pop_cores()
                if not assigned:
                    break
                state: _TmuxState | None = None
                try:
                    state = self._launch_session(launch, assigned)
                    if on_start is not None:
                        on_start(
                            launch,
                            {
                                "pid": state.pane_pid,
                                "assigned_cores": list(assigned),
                                "tmux_session": state.session_name,
                                "executor": self.name,
                            },
                        )
                    running.append(state)
                except Exception:
                    if state is not None:
                        self._cleanup_failed_start(state)
                    self._push_cores(assigned)
                    raise

            if not running:
                continue

            next_running: list[_TmuxState] = []
            for state in running:
                session_active = self._session_exists(state.session_name)
                if session_active and self.command_timeout_sec is not None:
                    elapsed = time.perf_counter() - state.start_perf
                    if elapsed > float(self.command_timeout_sec):
                        if not state.timed_out:
                            state.timed_out = True
                            state.timeout_kill_started_perf = time.perf_counter()

                        session_active = not self._terminate_session_on_timeout(state)
                        if session_active:
                            kill_elapsed = 0.0
                            if state.timeout_kill_started_perf is not None:
                                kill_elapsed = (
                                    time.perf_counter()
                                    - state.timeout_kill_started_perf
                                )
                            if kill_elapsed > self._TIMEOUT_KILL_GRACE_SEC:
                                raise RuntimeError(
                                    f"tmux session '{state.session_name}' did not exit after timeout kill "
                                    f"for config {state.launch.config.config_id} "
                                    f"(batch {state.launch.config.batch_index}, "
                                    f"line {state.launch.config.line_index})."
                                )

                if session_active:
                    next_running.append(state)
                    continue

                rc: int | None = None
                if state.rc_path.exists():
                    raw = state.rc_path.read_text(encoding="utf-8").strip()
                    if raw and (raw.lstrip("-").isdigit()):
                        rc = int(raw)

                end_time = _now_iso()
                duration = time.perf_counter() - state.start_perf

                if rc is None:
                    if not state.timed_out:
                        _log.warning(
                            "No exit code file for session '%s' "
                            "(config %s, batch %d, line %d); "
                            "marking as terminated",
                            state.session_name,
                            state.launch.config.config_id,
                            state.launch.config.batch_index,
                            state.launch.config.line_index,
                        )
                    status = "terminated"
                elif rc == 0:
                    status = "success"
                elif rc < 0:
                    status = "terminated"
                else:
                    status = "failed"

                self._push_cores(state.assigned_cores)
                state.rc_path.unlink(missing_ok=True)
                state.wrapper_path.unlink(missing_ok=True)

                yield CompletedCommand(
                    launch=state.launch,
                    pid=state.pane_pid,
                    tmux_session=state.session_name,
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
                time.sleep(0.1)
