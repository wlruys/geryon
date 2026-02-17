from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import os
import shlex
import shutil
import socket
import sys
import time
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from geryon.executors.affinity import discover_core_pool
from geryon.executors.base import CommandLaunch, CompletedCommand, ExecutorStartCallback


def _iso_from_epoch(timestamp: float) -> str:
    return (
        dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_rc(path: Path) -> int | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if raw and raw.lstrip("-").isdigit():
        return int(raw)
    return None


def _parse_timing(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None

    start_epoch: float | None = None
    end_epoch: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep != "=":
            continue
        key = key.strip()
        value = value.strip()
        try:
            parsed = float(value)
        except ValueError:
            continue
        if key == "start":
            start_epoch = parsed
        elif key == "end":
            end_epoch = parsed
    return start_epoch, end_epoch


@dataclass(frozen=True)
class _PreparedLaunch:
    launch: CommandLaunch
    wrapper_path: Path
    rc_path: Path
    timing_path: Path


class PyLauncherBatchExecutor:
    name = "pylauncher"

    def __init__(
        self,
        *,
        state_dir: Path,
        k_per_launch: int = 1,
        max_parallel_tasks: int | None = None,
        cores: str | None = None,
        command_timeout_sec: int | None = None,
    ):
        if k_per_launch < 1:
            raise ValueError("k_per_launch must be >= 1")
        if max_parallel_tasks is not None and max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be >= 1 when provided")
        if command_timeout_sec is not None and command_timeout_sec <= 0:
            raise ValueError("command_timeout_sec must be >= 1 when provided")

        core_pool = discover_core_pool(cores)
        if len(core_pool) < k_per_launch:
            raise RuntimeError(
                f"CPU pool has {len(core_pool)} cores; need at least {k_per_launch}"
            )

        max_parallel = len(core_pool) // k_per_launch
        if max_parallel_tasks is not None:
            max_parallel = min(max_parallel, max_parallel_tasks)
        if max_parallel < 1:
            raise RuntimeError(
                f"Insufficient CPU capacity for k_per_launch={k_per_launch} with requested limits."
            )

        self.k_per_launch = k_per_launch
        self.max_parallel = max_parallel
        self.core_pool = core_pool[: max_parallel * k_per_launch]
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.command_timeout_sec = command_timeout_sec

    def _install_hostlist3_shim(self) -> None:
        if "hostlist3" in sys.modules:
            return

        def _expand_piece(piece: str) -> list[str]:
            if "-" not in piece:
                return [piece]
            start, end = piece.split("-", 1)
            if not (start.isdigit() and end.isdigit()):
                return [piece]
            start_int = int(start)
            end_int = int(end)
            if end_int < start_int:
                return [piece]
            width = max(len(start), len(end))
            return [str(value).zfill(width) for value in range(start_int, end_int + 1)]

        def expand_hostlist(spec: str) -> list[str]:
            text = str(spec).strip()
            if not text:
                return []
            if "[" not in text or "]" not in text:
                return [text]

            left = text.index("[")
            right = text.index("]", left + 1)
            prefix = text[:left]
            suffix = text[right + 1 :]
            body = text[left + 1 : right]

            names: list[str] = []
            for chunk in body.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                for value in _expand_piece(chunk):
                    names.append(f"{prefix}{value}{suffix}")
            return names or [text]

        shim = types.ModuleType("hostlist3")
        shim.expand_hostlist = expand_hostlist  # type: ignore[attr-defined]
        sys.modules["hostlist3"] = shim

    def _load_pylauncher(self) -> Any:
        module_names = ("pylauncher.pylauncher_core", "pylauncher")
        import_errors: list[tuple[str, Exception]] = []
        needs_retry = False
        tried_hostlist_shim = False

        while True:
            for name in module_names:
                try:
                    module = importlib.import_module(name)
                except SystemExit as exc:
                    raise RuntimeError(
                        "pylauncher failed to initialize. Ensure paramiko is installed in this environment."
                    ) from exc
                except ModuleNotFoundError as exc:
                    if exc.name == "hostlist3" and not tried_hostlist_shim:
                        self._install_hostlist3_shim()
                        tried_hostlist_shim = True
                        needs_retry = True
                        break
                    import_errors.append((name, exc))
                except Exception as exc:
                    import_errors.append((name, exc))
                else:
                    if not hasattr(module, "ClassicLauncher"):
                        raise RuntimeError(
                            f"Imported module '{name}' does not expose ClassicLauncher."
                        )
                    return module
            if needs_retry:
                needs_retry = False
                continue
            break

        if not import_errors:
            raise RuntimeError("Unable to import pylauncher.")
        error_details = "; ".join(f"{name}: {exc}" for name, exc in import_errors)
        raise RuntimeError(
            "pylauncher is required for executor='pylauncher'. "
            "Install it with: uv pip install pylauncher. "
            f"Import errors: {error_details}"
        ) from import_errors[-1][1]

    def _write_timeout_driver(self, run_dir: Path) -> Path:
        timeout_driver = run_dir / "timeout_driver.py"
        timeout_driver.write_text(
            "\n".join(
                [
                    "import os",
                    "import signal",
                    "import subprocess",
                    "import sys",
                    "cmd = sys.argv[1]",
                    "timeout = float(sys.argv[2])",
                    "proc = subprocess.Popen(['bash', '-lc', cmd], start_new_session=True)",
                    "try:",
                    "    raise SystemExit(proc.wait(timeout=timeout))",
                    "except subprocess.TimeoutExpired:",
                    "    if os.name == 'posix':",
                    "        for signum in (signal.SIGTERM, signal.SIGKILL):",
                    "            try:",
                    "                os.killpg(proc.pid, signum)",
                    "            except OSError:",
                    "                pass",
                    "            try:",
                    "                proc.wait(timeout=0.5)",
                    "                break",
                    "            except subprocess.TimeoutExpired:",
                    "                continue",
                    "    else:",
                    "        try:",
                    "            proc.terminate()",
                    "            proc.wait(timeout=0.5)",
                    "        except Exception:",
                    "            try:",
                    "                proc.kill()",
                    "                proc.wait(timeout=0.5)",
                    "            except Exception:",
                    "                pass",
                    "    raise SystemExit(124)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        timeout_driver.chmod(0o755)
        return timeout_driver

    def _prepare_launches(
        self,
        launches: list[CommandLaunch],
        run_dir: Path,
        timeout_driver_path: Path | None,
    ) -> list[_PreparedLaunch]:
        wrappers_dir = run_dir / "wrappers"
        wrappers_dir.mkdir(parents=True, exist_ok=True)
        prepared: list[_PreparedLaunch] = []

        for index, launch in enumerate(launches):
            launch.stdout_path.parent.mkdir(parents=True, exist_ok=True)
            launch.stderr_path.parent.mkdir(parents=True, exist_ok=True)

            rc_path = run_dir / f"rc_{index:04d}_{launch.attempt_id[:8]}.txt"
            timing_path = run_dir / f"timing_{index:04d}_{launch.attempt_id[:8]}.txt"
            wrapper_path = wrappers_dir / f"cmd_{index:04d}_{launch.attempt_id[:8]}.sh"

            if timeout_driver_path is None:
                wrapped_command = f"bash -lc {shlex.quote(launch.command)}"
            else:
                wrapped_command = (
                    f"{shlex.quote(sys.executable)} {shlex.quote(str(timeout_driver_path))} "
                    f"{shlex.quote(launch.command)} {shlex.quote(str(self.command_timeout_sec))}"
                )

            wrapper_path.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set +e",
                        f"start_epoch=$({shlex.quote(sys.executable)} -c 'import time; print(time.time())')",
                        f"printf 'start=%s\\n' \"$start_epoch\" > {shlex.quote(str(timing_path))}",
                        f"{wrapped_command} > {shlex.quote(str(launch.stdout_path))} 2> {shlex.quote(str(launch.stderr_path))}",
                        "rc=$?",
                        f"end_epoch=$({shlex.quote(sys.executable)} -c 'import time; print(time.time())')",
                        f"printf 'end=%s\\n' \"$end_epoch\" >> {shlex.quote(str(timing_path))}",
                        f"printf '%s\\n' \"$rc\" > {shlex.quote(str(rc_path))}",
                        "exit 0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            wrapper_path.chmod(0o755)
            prepared.append(
                _PreparedLaunch(
                    launch=launch,
                    wrapper_path=wrapper_path,
                    rc_path=rc_path,
                    timing_path=timing_path,
                )
            )

        return prepared

    def _has_slurm_context(self) -> bool:
        return bool(os.environ.get("SLURM_NODELIST")) and bool(
            os.environ.get("SLURM_CPUS_ON_NODE")
        )

    @contextlib.contextmanager
    def _classic_local_shim(self, pylauncher_module: Any) -> Iterator[None]:
        required = ("HostListByName", "SSHExecutor", "LocalExecutor", "HostList")
        missing = [name for name in required if not hasattr(pylauncher_module, name)]
        if missing:
            raise RuntimeError(
                "pylauncher module does not support local ClassicLauncher shim; "
                f"missing: {', '.join(missing)}"
            )

        host_name = socket.gethostname()
        total_slots = len(self.core_pool)
        original_host_list_by_name = pylauncher_module.HostListByName
        original_ssh_executor = pylauncher_module.SSHExecutor

        env_keys = (
            "SLURM_CPUS_ON_NODE",
            "SLURM_NPROCS",
            "SLURM_NNODES",
            "SLURM_NODELIST",
            "SLURM_JOB_ID",
        )
        old_env = {key: os.environ.get(key) for key in env_keys}

        def _local_host_list_by_name(**_kwargs: Any) -> Any:
            host_list = pylauncher_module.HostList()
            for slot in range(total_slots):
                host_list.append(
                    {
                        "host": host_name,
                        "hostnum": 0,
                        "task_loc": slot,
                        "phys_core": f"{slot}-{slot}",
                    }
                )
            return host_list

        pylauncher_module.HostListByName = _local_host_list_by_name
        pylauncher_module.SSHExecutor = pylauncher_module.LocalExecutor
        os.environ["SLURM_CPUS_ON_NODE"] = str(total_slots)
        os.environ["SLURM_NPROCS"] = str(total_slots)
        os.environ["SLURM_NNODES"] = "1"
        os.environ["SLURM_NODELIST"] = host_name
        os.environ.setdefault("SLURM_JOB_ID", "local")

        try:
            yield
        finally:
            pylauncher_module.HostListByName = original_host_list_by_name
            pylauncher_module.SSHExecutor = original_ssh_executor
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _run_launcher(
        self,
        pylauncher_module: Any,
        *,
        command_file: Path,
        run_dir: Path,
    ) -> str:
        kwargs: dict[str, Any] = {
            "cores": self.k_per_launch,
            "workdir": str(run_dir / "workdir"),
            "queuestate": str(run_dir / "queuestate"),
        }

        if self._has_slurm_context():
            pylauncher_module.ClassicLauncher(str(command_file), **kwargs)
            return "classic"

        local_launcher = getattr(pylauncher_module, "LocalLauncher", None)
        if callable(local_launcher):
            local_launcher(str(command_file), nhosts=len(self.core_pool), **kwargs)
            return "local"

        with self._classic_local_shim(pylauncher_module):
            pylauncher_module.ClassicLauncher(str(command_file), **kwargs)
        return "classic-shim"

    def run(
        self,
        launches: list[CommandLaunch],
        on_start: ExecutorStartCallback | None = None,
    ) -> Iterator[CompletedCommand]:
        if not launches:
            return

        pylauncher_module = self._load_pylauncher()
        run_dir = self.state_dir / f"run_{uuid.uuid4().hex[:12]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        timeout_driver_path: Path | None = None
        if self.command_timeout_sec is not None:
            timeout_driver_path = self._write_timeout_driver(run_dir)

        prepared = self._prepare_launches(
            launches,
            run_dir=run_dir,
            timeout_driver_path=timeout_driver_path,
        )

        if on_start is not None:
            try:
                for item in prepared:
                    on_start(
                        item.launch,
                        {
                            "pid": None,
                            "assigned_cores": [],
                            "tmux_session": None,
                            "executor": self.name,
                        },
                    )
            except Exception:
                shutil.rmtree(run_dir, ignore_errors=True)
                raise

        command_file = run_dir / "commandlines.txt"
        command_file.write_text(
            "\n".join(shlex.quote(str(item.wrapper_path)) for item in prepared) + "\n",
            encoding="utf-8",
        )

        run_start_epoch = time.time()
        launcher_mode = "unknown"
        try:
            launcher_mode = self._run_launcher(
                pylauncher_module,
                command_file=command_file,
                run_dir=run_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"pylauncher execution failed "
                f"(command_file={command_file}, mode={launcher_mode}, "
                f"num_commands={len(prepared)}, cores_per_launch={self.k_per_launch}): "
                f"{exc}"
            ) from exc
        run_end_epoch = time.time()

        for item in prepared:
            rc = _parse_rc(item.rc_path)

            status_reason: str | None = None
            if rc is None:
                status = "terminated"
            elif rc == 0:
                status = "success"
            elif rc == 124 and self.command_timeout_sec is not None:
                status = "terminated"
                status_reason = "timeout_kill"
            elif rc < 0:
                status = "terminated"
            else:
                status = "failed"

            parsed_start_epoch, parsed_end_epoch = _parse_timing(item.timing_path)
            start_epoch = (
                parsed_start_epoch
                if parsed_start_epoch is not None
                else run_start_epoch
            )
            if parsed_end_epoch is not None:
                end_epoch = parsed_end_epoch
            elif item.rc_path.exists():
                end_epoch = item.rc_path.stat().st_mtime
            else:
                end_epoch = run_end_epoch
            duration = max(0.0, end_epoch - start_epoch)

            yield CompletedCommand(
                launch=item.launch,
                pid=None,
                tmux_session=None,
                start_time=_iso_from_epoch(start_epoch),
                end_time=_iso_from_epoch(end_epoch),
                duration_sec=round(duration, 6),
                exit_code=rc,
                status=status,
                status_reason=status_reason,
                assigned_cores=(),
            )
