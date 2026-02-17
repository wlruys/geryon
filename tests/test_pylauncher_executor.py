from __future__ import annotations

import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

from geryon.executors.base import CommandLaunch
from geryon.executors.pylauncher import PyLauncherBatchExecutor
from geryon.models import PlannedConfig


def _make_launch(tmp_path: Path, *, line_index: int, command: str) -> CommandLaunch:
    config = PlannedConfig(
        run_id="run",
        config_id=f"cfg{line_index:04d}",
        batch_index=0,
        line_index=line_index,
        command=command,
        params={},
        tags=(),
        wandb_name=f"cfg{line_index:04d}",
        selected_options={},
    )
    return CommandLaunch(
        config=config,
        command=command,
        stdout_path=tmp_path / f"stdout_{line_index}.log",
        stderr_path=tmp_path / f"stderr_{line_index}.log",
        attempt_id=f"attempt{line_index:04d}",
    )


def _execute_command_file(command_file: str) -> None:
    for raw in Path(command_file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        subprocess.run(["bash", "-lc", line], check=True)


def _fake_pylauncher_module(
    call_log: dict[str, list[dict[str, object]]],
) -> types.SimpleNamespace:
    def _classic_launcher(commandfile: str, **kwargs: object) -> None:
        call_log["classic"].append({"commandfile": commandfile, "kwargs": dict(kwargs)})
        _execute_command_file(commandfile)

    def _local_launcher(commandfile: str, nhosts: int, **kwargs: object) -> None:
        call_log["local"].append(
            {"commandfile": commandfile, "nhosts": nhosts, "kwargs": dict(kwargs)}
        )
        _execute_command_file(commandfile)

    return types.SimpleNamespace(
        ClassicLauncher=_classic_launcher,
        LocalLauncher=_local_launcher,
    )


def test_pylauncher_executor_uses_local_launcher_outside_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("SLURM_NODELIST", "SLURM_CPUS_ON_NODE", "SLURM_NNODES", "SLURM_NPROCS"):
        monkeypatch.delenv(key, raising=False)

    launch_a = _make_launch(tmp_path, line_index=0, command='printf "a\\n"')
    launch_b = _make_launch(tmp_path, line_index=1, command='printf "b\\n"')
    executor = PyLauncherBatchExecutor(
        state_dir=tmp_path / "state",
        k_per_launch=1,
        max_parallel_tasks=2,
        cores="0-3",
    )

    call_log: dict[str, list[dict[str, object]]] = {"classic": [], "local": []}
    fake_module = _fake_pylauncher_module(call_log)
    monkeypatch.setattr(executor, "_load_pylauncher", lambda: fake_module)

    starts: list[dict[str, object]] = []

    def _on_start(_launch: CommandLaunch, meta: dict[str, object]) -> None:
        starts.append(dict(meta))

    results = list(executor.run([launch_a, launch_b], on_start=_on_start))

    assert len(call_log["classic"]) == 0
    assert len(call_log["local"]) == 1
    assert int(call_log["local"][0]["nhosts"]) == len(executor.core_pool)
    assert len(starts) == 2
    assert all(item["pid"] is None for item in starts)
    assert all(item["tmux_session"] is None for item in starts)
    assert all(item["assigned_cores"] == [] for item in starts)
    assert all(item["executor"] == "pylauncher" for item in starts)

    assert [result.status for result in results] == ["success", "success"]
    assert launch_a.stdout_path.read_text(encoding="utf-8").strip() == "a"
    assert launch_b.stdout_path.read_text(encoding="utf-8").strip() == "b"


def test_pylauncher_executor_uses_classic_launcher_with_slurm_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_NODELIST", "node001")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    monkeypatch.setenv("SLURM_NNODES", "1")
    monkeypatch.setenv("SLURM_NPROCS", "8")

    launch = _make_launch(tmp_path, line_index=0, command='printf "classic\\n"')
    executor = PyLauncherBatchExecutor(
        state_dir=tmp_path / "state",
        k_per_launch=2,
        max_parallel_tasks=2,
        cores="0-7",
    )

    call_log: dict[str, list[dict[str, object]]] = {"classic": [], "local": []}
    fake_module = _fake_pylauncher_module(call_log)
    monkeypatch.setattr(executor, "_load_pylauncher", lambda: fake_module)

    results = list(executor.run([launch]))

    assert len(call_log["classic"]) == 1
    assert len(call_log["local"]) == 0
    kwargs = dict(call_log["classic"][0]["kwargs"])
    assert kwargs["cores"] == 2
    assert "workdir" in kwargs
    assert "queuestate" in kwargs
    assert len(results) == 1
    assert results[0].status == "success"
    assert launch.stdout_path.read_text(encoding="utf-8").strip() == "classic"


def test_pylauncher_executor_marks_timeout_as_terminated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("SLURM_NODELIST", "SLURM_CPUS_ON_NODE", "SLURM_NNODES", "SLURM_NPROCS"):
        monkeypatch.delenv(key, raising=False)

    py = shlex.quote(sys.executable)
    launch = _make_launch(
        tmp_path,
        line_index=0,
        command=f'{py} -c "import time; time.sleep(2)"',
    )
    executor = PyLauncherBatchExecutor(
        state_dir=tmp_path / "state",
        k_per_launch=1,
        max_parallel_tasks=1,
        cores="0-1",
        command_timeout_sec=1,
    )

    call_log: dict[str, list[dict[str, object]]] = {"classic": [], "local": []}
    fake_module = _fake_pylauncher_module(call_log)
    monkeypatch.setattr(executor, "_load_pylauncher", lambda: fake_module)

    results = list(executor.run([launch]))

    assert len(results) == 1
    result = results[0]
    assert result.status == "terminated"
    assert result.status_reason == "timeout_kill"
    assert result.exit_code == 124


def test_pylauncher_executor_on_start_error_skips_launcher_and_cleans_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("SLURM_NODELIST", "SLURM_CPUS_ON_NODE", "SLURM_NNODES", "SLURM_NPROCS"):
        monkeypatch.delenv(key, raising=False)

    launch = _make_launch(tmp_path, line_index=0, command='printf "ignored\\n"')
    state_dir = tmp_path / "state"
    executor = PyLauncherBatchExecutor(
        state_dir=state_dir,
        k_per_launch=1,
        max_parallel_tasks=1,
    )

    call_log: dict[str, list[dict[str, object]]] = {"classic": [], "local": []}
    fake_module = _fake_pylauncher_module(call_log)
    monkeypatch.setattr(executor, "_load_pylauncher", lambda: fake_module)

    def _on_start(_launch: CommandLaunch, _meta: dict[str, object]) -> None:
        raise RuntimeError("start callback failed")

    with pytest.raises(RuntimeError, match="start callback failed"):
        list(executor.run([launch], on_start=_on_start))

    assert len(call_log["classic"]) == 0
    assert len(call_log["local"]) == 0
    assert not list(state_dir.glob("run_*"))
