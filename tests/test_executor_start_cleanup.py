from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path

import pytest

from geryon.executors.base import CommandLaunch
from geryon.executors.process import ProcessBatchExecutor
from geryon.executors.tmux import TmuxBatchExecutor, _TmuxState
from geryon.models import PlannedConfig


def _make_launch(tmp_path: Path, *, command: str) -> CommandLaunch:
    config = PlannedConfig(
        run_id="run",
        config_id="cfg0001",
        batch_index=0,
        line_index=0,
        command=command,
        params={},
        tags=(),
        wandb_name="cfg0001",
        selected_options={},
    )
    return CommandLaunch(
        config=config,
        command=command,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        attempt_id="attempt0001",
    )


def test_process_executor_kills_child_if_on_start_raises(tmp_path: Path) -> None:
    launch = _make_launch(
        tmp_path,
        command='python3 -c "import time; time.sleep(30)"',
    )
    executor = ProcessBatchExecutor(max_workers=1, k_per_worker=1)
    started_pid: int | None = None

    def _on_start(_launch: CommandLaunch, meta: dict[str, object]) -> None:
        nonlocal started_pid
        started_pid = int(meta["pid"])
        raise RuntimeError("start callback failed")

    with pytest.raises(RuntimeError, match="start callback failed"):
        list(executor.run([launch], on_start=_on_start))

    assert started_pid is not None
    deadline = time.time() + 3.0
    alive = True
    while time.time() < deadline:
        try:
            os.kill(started_pid, 0)
        except OSError:
            alive = False
            break
        time.sleep(0.05)

    if alive:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(started_pid), signal.SIGKILL)
        with contextlib.suppress(OSError):
            os.kill(started_pid, signal.SIGKILL)

    assert not alive
    assert executor.core_allocator.can_allocate()


def test_tmux_executor_terminates_session_if_on_start_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("geryon.executors.tmux.shutil.which", lambda _: "/usr/bin/tmux")

    state_dir = tmp_path / "state"
    launch = _make_launch(tmp_path, command='python3 -c "print(1)"')
    executor = TmuxBatchExecutor(
        state_dir=state_dir,
        k_per_session=1,
        max_parallel_tasks=1,
        prefix="geryon-test",
    )
    cleanup_calls: list[str] = []
    rc_path_holder: dict[str, Path] = {}
    wrapper_path_holder: dict[str, Path] = {}

    def _fake_launch_session(
        self: TmuxBatchExecutor,
        launch: CommandLaunch,
        assigned_cores: list[int],
    ) -> _TmuxState:
        rc_path = state_dir / "fake.rc"
        wrapper_path = state_dir / "fake.sh"
        rc_path.parent.mkdir(parents=True, exist_ok=True)
        rc_path.write_text("0\n", encoding="utf-8")
        wrapper_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        rc_path_holder["path"] = rc_path
        wrapper_path_holder["path"] = wrapper_path
        return _TmuxState(
            launch=launch,
            session_name="geryon_fake",
            pane_pid=None,
            rc_path=rc_path,
            wrapper_path=wrapper_path,
            start_time="2026-01-01T00:00:00Z",
            start_perf=time.perf_counter(),
            assigned_cores=list(assigned_cores),
        )

    def _fake_terminate(self: TmuxBatchExecutor, state: _TmuxState) -> bool:
        cleanup_calls.append(state.session_name)
        return True

    monkeypatch.setattr(TmuxBatchExecutor, "_launch_session", _fake_launch_session)
    monkeypatch.setattr(
        TmuxBatchExecutor, "_terminate_session_on_timeout", _fake_terminate
    )

    def _on_start(_launch: CommandLaunch, _meta: dict[str, object]) -> None:
        raise RuntimeError("start callback failed")

    with pytest.raises(RuntimeError, match="start callback failed"):
        list(executor.run([launch], on_start=_on_start))

    assert cleanup_calls == ["geryon_fake"]
    assert not rc_path_holder["path"].exists()
    assert not wrapper_path_holder["path"].exists()
    assert executor.core_allocator.can_allocate()
