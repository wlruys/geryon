from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from geryon.cli import main
from geryon.models import ConfigError, PlanSummary, PlannedConfig
from geryon.planner import plan_experiment
from geryon.status import successful_config_ids
from geryon.store import ArtifactStore
from geryon.utils import read_jsonl


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _collect_attempts(store: ArtifactStore) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for path in sorted(store.results_dir.glob("task_*.jsonl")):
        attempts.extend(list(read_jsonl(path)))
    return attempts


class _FakeJob:
    def __init__(self, job_id: str):
        self.job_id = job_id


class _FakeExecutor:
    def __init__(self) -> None:
        self.parameters: dict[str, Any] = {}
        self.calls: list[tuple[Any, list[Any]]] = []

    def update_parameters(self, **kwargs: Any) -> None:
        self.parameters = kwargs

    def map_array(self, fn: Any, payloads: Any) -> list[_FakeJob]:
        payload_items = list(payloads)
        self.calls.append((fn, payload_items))
        return [_FakeJob(str(3000 + idx)) for idx in range(len(payload_items))]


class _FakeSubmitit:
    def __init__(self, executor: _FakeExecutor):
        self._executor = executor

    def AutoExecutor(self, folder: str) -> _FakeExecutor:
        return self._executor


def test_plan_inspect_and_run_sets_workflow(tmp_path: Path, capsys) -> None:
    defs_dir = tmp_path / "defs"
    _write(
        defs_dir / "options.yaml",
        """
option_sets:
  arch:
    - id: resnet18
      params:
        model:
          name: resnet18
    - id: vit_tiny
      params:
        model:
          name: vit_tiny
  profile:
    - id: cnn_profile
      params:
        train:
          lr: 0.001
    - id: vit_profile
      params:
        train:
          lr: 0.0003
  seed:
    - id: s1
      params:
        seed: 1
    - id: s2
      params:
        seed: 2
""",
    )
    _write(
        defs_dir / "packs.yaml",
        """
packs:
  architecture:
    name: architecture
    options_from:
      - ref: arch
  hparam_profile:
    name: hparam_profile
    options_from:
      - ref: profile
  seed_pack:
    name: seed
    options_from:
      - ref: seed
""",
    )
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
imports:
  - path: defs/options.yaml
    package: presets
  - path: defs/packs.yaml
    package: presets
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('plan workflow')"]
constraints:
  predicates:
    - id: arch_profile_compat
      args:
        arch:
          pack_id: architecture
        profile:
          pack_id: hparam_profile
      expr:
        or:
          - and:
              - eq: ["$arch", "resnet18"]
              - eq: ["$profile", "cnn_profile"]
          - and:
              - eq: ["$arch", "vit_tiny"]
              - eq: ["$profile", "vit_profile"]
select:
  packs:
    - ref: presets.architecture
    - ref: presets.hparam_profile
    - ref: presets.seed_pack
run_sets:
  baseline: {}
  narrow:
    replace:
      select:
        packs:
          - ref: presets.architecture
            replace_options: true
            options_from:
              - ref: presets.arch
                include_ids: [resnet18]
          - ref: presets.hparam_profile
            replace_options: true
            options_from:
              - ref: presets.profile
                include_ids: [cnn_profile]
          - ref: presets.seed_pack
""",
    )

    rc = main(
        [
            "inspect-config",
            "--experiment",
            str(experiment),
            "--run-set",
            "narrow",
            "--show-diagnostics",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["diagnostics"]["run_set"]["selected"] == "narrow"
    packs_by_name = {
        str(pack["name"]): list(pack.get("options", []))
        for pack in inspect_payload["experiment"]["packs"]
    }
    assert len(packs_by_name["architecture"]) == 1
    assert len(packs_by_name["hparam_profile"]) == 1
    assert len(packs_by_name["seed"]) == 2

    rc = main(
        [
            "plan",
            "--experiment",
            str(experiment),
            "--out",
            str(tmp_path / "out"),
            "--batch-size",
            "8",
            "--run-id",
            "suite",
            "--all-run-sets",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["all_run_sets"] is True
    assert payload["num_run_sets"] == 2

    items = {item["run_set"]: item for item in payload["items"]}
    assert items["baseline"]["run_id"] == "suite_baseline"
    assert items["narrow"]["run_id"] == "suite_narrow"
    assert items["baseline"]["total_configs"] == 4
    assert items["narrow"]["total_configs"] == 2

    baseline_store = ArtifactStore(Path(items["baseline"]["run_root"]))
    baseline_configs = baseline_store.read_planned_configs()
    pairs = {
        (
            cfg.selected_options["architecture"],
            cfg.selected_options["hparam_profile"],
        )
        for cfg in baseline_configs
    }
    assert pairs == {
        ("resnet18", "cnn_profile"),
        ("vit_tiny", "vit_profile"),
    }

    diagnostics = json.loads(
        baseline_store.plan_diagnostics_path.read_text(encoding="utf-8")
    )
    assert diagnostics["predicates"]["counters"]["dropped_candidates"] == 4

    assert baseline_store.plan_run_set_path.exists()
    narrow_store = ArtifactStore(Path(items["narrow"]["run_root"]))
    assert narrow_store.plan_run_set_path.exists()


def test_plan_preview_and_show_config(tmp_path: Path, capsys) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('preview')"]
select:
  packs:
    - name: model
      options:
        - id: a
          params: {model: {name: a}}
        - id: b
          params: {model: {name: b}}
    - name: seed
      options:
        - id: s1
          params: {seed: 1}
        - id: s2
          params: {seed: 2}
""",
    )

    rc = main(
        [
            "plan",
            "--experiment",
            str(experiment),
            "--out",
            str(tmp_path / "out"),
            "--batch-size",
            "2",
            "--run-id",
            "preview",
            "--dry-run",
            "--preview-configs",
            "3",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    preview_payload = json.loads(capsys.readouterr().out)
    assert preview_payload["dry_run"] is True
    assert len(preview_payload["preview_configs"]) == 3
    assert preview_payload["preview_configs"][0]["config_id"]

    rc = main(
        [
            "plan",
            "--experiment",
            str(experiment),
            "--out",
            str(tmp_path / "out"),
            "--batch-size",
            "2",
            "--run-id",
            "show_cfg",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    plan_payload = json.loads(capsys.readouterr().out)
    store = ArtifactStore(Path(plan_payload["run_root"]))
    cfg = store.read_planned_configs()[0]

    rc = main(
        [
            "show-config",
            "--run",
            str(store.run_root),
            cfg.config_id[:10],
            "--format",
            "json",
        ]
    )
    assert rc == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["config_id"] == cfg.config_id
    assert shown["batch_file"].endswith(".txt")


def test_show_config_rejects_ambiguous_prefix(tmp_path: Path, capsys) -> None:
    run_root = tmp_path / "out" / "runs" / "demo"
    store = ArtifactStore(run_root)
    store.ensure_plan_layout()
    store.write_planned_configs(
        [
            PlannedConfig(
                run_id="demo",
                config_id="abc1111111111111111111111111111111111111",
                batch_index=0,
                line_index=0,
                command="echo a",
                params={},
                tags=(),
                wandb_name="a",
                selected_options={},
            ),
            PlannedConfig(
                run_id="demo",
                config_id="abc2222222222222222222222222222222222222",
                batch_index=0,
                line_index=1,
                command="echo b",
                params={},
                tags=(),
                wandb_name="b",
                selected_options={},
            ),
        ]
    )

    rc = main(["show-config", "--run", str(run_root), "abc", "--format", "json"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "ambiguous" in captured.err


def test_plan_policy_gate_rejects_unknown_constraint_reference(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: merge
constraints:
  exclude:
    - when:
        missing_pack: [foo]
select:
  packs:
    - name: seed
      options:
        - id: s1
          params:
            seed: 1
""",
    )

    with pytest.raises(ConfigError, match="Unknown constraint references"):
        plan_experiment(
            experiment_path=experiment,
            out_dir=tmp_path / "out",
            run_id="bad_constraint_ref",
            batch_size=1,
            dry_run=True,
        )


def test_import_cycle_error_shows_chain(tmp_path: Path) -> None:
    a_yaml = tmp_path / "a.yaml"
    b_yaml = tmp_path / "b.yaml"
    _write(
        a_yaml,
        """
imports:
  - path: b.yaml
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('a')"]
select:
  packs:
    - name: seed
      options:
        - id: s1
          params: {seed: 1}
""",
    )
    _write(
        b_yaml,
        """
imports:
  - path: a.yaml
schema:
  version: 4
""",
    )

    with pytest.raises(ConfigError) as excinfo:
        plan_experiment(
            experiment_path=a_yaml,
            out_dir=tmp_path / "out",
            run_id="cycle",
            batch_size=1,
            dry_run=True,
        )
    message = str(excinfo.value)
    assert "Import cycle detected" in message
    assert "cycle chain:" in message


def test_merge_conflict_error_suggests_fix(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
select:
  packs:
    - name: alpha
      options:
        - id: a
          params:
            train:
              lr: 0.1
    - name: beta
      options:
        - id: b
          params:
            train:
              lr: 0.2
""",
    )
    with pytest.raises(ConfigError, match="Parameter conflict at equal priority"):
        plan_experiment(
            experiment_path=experiment,
            out_dir=tmp_path / "out",
            run_id="merge_conflict",
            batch_size=1,
            dry_run=True,
        )


def test_predicate_error_includes_resolved_args(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
constraints:
  predicates:
    - id: needs_missing_param
      args:
        p:
          param: train.missing
      expr:
        eq: ["$p", 1]
select:
  packs:
    - name: seed
      options:
        - id: s1
          params: {seed: 1}
""",
    )
    with pytest.raises(ConfigError, match="resolved args:"):
        plan_experiment(
            experiment_path=experiment,
            out_dir=tmp_path / "out",
            run_id="pred_ctx",
            batch_size=1,
            dry_run=True,
        )


def test_run_local_status_rerun_collect_and_report_lifecycle(
    tmp_path: Path, capsys
) -> None:
    marker_dir = tmp_path / "markers"
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        f"""
schema:
  version: 4
command:
  program: python3
  args:
    - -c
    - |
      import pathlib
      import sys
      mode = "ok"
      marker_dir = None
      for arg in sys.argv[1:]:
          if arg.startswith("mode="):
              mode = arg.split("=", 1)[1]
          if arg.startswith("marker_dir="):
              marker_dir = arg.split("=", 1)[1]
      if mode == "flaky":
          marker_path = pathlib.Path(marker_dir) / "flaky.once"
          marker_path.parent.mkdir(parents=True, exist_ok=True)
          if not marker_path.exists():
              marker_path.write_text("first-failure", encoding="utf-8")
              sys.exit(7)
      sys.exit(0)
select:
  packs:
    - name: exit_mode
      options:
        - id: ok
          params:
            mode: ok
            marker_dir: {json.dumps(str(marker_dir))}
        - id: flaky
          params:
            mode: flaky
            marker_dir: {json.dumps(str(marker_dir))}
""",
    )

    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="lifecycle",
        batch_size=2,
    )

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)

    rc = main(["status", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    status_before = json.loads(capsys.readouterr().out)
    assert status_before["configs"]["completion"]["complete"] == 1
    assert status_before["configs"]["completion"]["pending"] == 1
    failed_ids = status_before["configs"]["failed_config_ids"]
    assert len(failed_ids) == 1

    rc = main(
        [
            "rerun",
            "--run",
            str(summary.run_root),
            "--status",
            "failed",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    rerun_payload = json.loads(capsys.readouterr().out)
    retry_file = Path(rerun_payload["retry_file"])
    assert rerun_payload["target_config_ids"] == failed_ids
    assert retry_file.exists()

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--retry-file",
            str(retry_file),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)

    rc = main(
        [
            "status",
            "--run",
            str(summary.run_root),
            "--format",
            "json",
            "--by-pack",
            "exit_mode",
        ]
    )
    assert rc == 0
    status_after = json.loads(capsys.readouterr().out)
    assert status_after["configs"]["completion"]["complete"] == 2
    assert status_after["configs"]["completion"].get("pending", 0) == 0
    assert status_after["configs"]["failed_config_ids"] == []
    grouped = status_after["grouped_by_packs"]
    assert grouped["packs"] == ["exit_mode"]
    assert {item["key"]["exit_mode"] for item in grouped["groups"]} == {"ok", "flaky"}

    store = ArtifactStore(summary.run_root)
    attempts = _collect_attempts(store)
    assert len(attempts) == 3
    attempts_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in attempts:
        attempts_by_config[str(record["config_id"])].append(record)
    assert sorted(len(items) for items in attempts_by_config.values()) == [1, 2]

    retried = next(items for items in attempts_by_config.values() if len(items) == 2)
    retried = sorted(retried, key=lambda item: int(item["attempt_index"]))
    assert retried[0]["status"] == "failed"
    assert retried[1]["status"] == "success"
    assert retried[1]["parent_attempt_id"] == retried[0]["attempt_id"]

    rc = main(["collect", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    collect_payload = json.loads(capsys.readouterr().out)
    assert collect_payload["attempts"]["total"] == 3
    assert collect_payload["configs"]["final_by_status"]["success"] == 2

    report_path = tmp_path / "report.md"
    rc = main(
        [
            "report",
            "--run",
            str(summary.run_root),
            "--format",
            "markdown",
            "--by-pack",
            "exit_mode",
            "--out",
            str(report_path),
        ]
    )
    assert rc == 0
    report_payload = json.loads(capsys.readouterr().out)
    assert Path(report_payload["output_path"]) == report_path.resolve()
    report_text = report_path.read_text(encoding="utf-8")
    assert "# Geryon Report" in report_text
    assert "Grouped by: `exit_mode`" in report_text


def test_status_and_rerun_use_latest_attempt_outcome(
    tmp_path: Path, capsys: Any
) -> None:
    marker_path = tmp_path / "latest_attempt.marker"
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        f"""
schema:
  version: 4
command:
  program: python3
  args:
    - -c
    - |
      import pathlib
      import sys
      marker_path = None
      for arg in sys.argv[1:]:
          if arg.startswith("marker_path="):
              marker_path = arg.split("=", 1)[1]
      if marker_path is None:
          raise SystemExit(2)
      marker = pathlib.Path(marker_path)
      marker.parent.mkdir(parents=True, exist_ok=True)
      if marker.exists():
          raise SystemExit(9)
      marker.write_text("first-success", encoding="utf-8")
      raise SystemExit(0)
select:
  packs:
    - name: single
      options:
        - id: only
          params:
            marker_path: {json.dumps(str(marker_path))}
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="latest_attempt_status",
        batch_size=1,
    )
    store = ArtifactStore(summary.run_root)
    cfg = store.read_planned_configs()[0]

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)

    rc = main(["status", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["configs"]["latest_by_status"] == {"failed": 1}
    assert status_payload["configs"]["completion"].get("complete", 0) == 0
    assert status_payload["configs"]["completion"]["pending"] == 1
    assert status_payload["configs"]["failed_config_ids"] == [cfg.config_id]
    assert successful_config_ids(summary.run_root) == set()

    rc = main(
        [
            "rerun",
            "--run",
            str(summary.run_root),
            "--status",
            "failed",
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    rerun_payload = json.loads(capsys.readouterr().out)
    assert rerun_payload["target_config_ids"] == [cfg.config_id]


def test_run_local_profile_bootstrap(tmp_path: Path, capsys) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python3
  args:
    - -c
    - |
      import os
      import sys
      if os.environ.get("FROM_PROFILE_ENV") != "enabled":
          sys.exit(3)
      if os.environ.get("FROM_PROFILE_SCRIPT") != "yes":
          sys.exit(4)
      if os.environ.get("FROM_PROFILE_CMD") != "ok":
          sys.exit(5)
      sys.exit(0)
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="local_profile",
        batch_size=1,
    )

    script = tmp_path / "env_setup.sh"
    _write(script, "export FROM_PROFILE_SCRIPT=yes")

    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        f"""
profiles:
  local_bootstrap:
    env_script: {script}
    env_setup_cmds:
      - export FROM_PROFILE_CMD=ok
    env:
      FROM_PROFILE_ENV: enabled
""",
    )

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--profile",
            "local_bootstrap",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)

    store = ArtifactStore(summary.run_root)
    attempts = _collect_attempts(store)
    assert len(attempts) == 1
    assert attempts[0]["status"] == "success"


def test_run_local_progress_refreshes_on_command_events(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('progress-refresh')"]
select:
  packs:
    - name: seed
      options:
        - id: s1
          params:
            seed: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="progress_refresh",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  local_progress:
    defaults:
      run_local:
        progress: true
""",
    )

    class _FakeLive:
        instances: list["_FakeLive"] = []

        def __init__(self, renderable: Any, **kwargs: Any) -> None:
            self.renderable = renderable
            self.kwargs = kwargs
            self.update_calls = 0
            _FakeLive.instances.append(self)

        def __enter__(self) -> "_FakeLive":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def update(self, renderable: Any, *, refresh: bool = False) -> None:
            self.renderable = renderable
            self.update_calls += 1

    def _fake_execute_task_payload(
        payload: Any,
        *,
        job_id: str | None = None,
        array_task_id: str | None = None,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "task_start",
                    "run_id": "progress_refresh",
                    "executor": "process",
                    "job_id": job_id or "local",
                    "array_task_id": array_task_id or "0",
                }
            )
            progress_callback(
                {
                    "event": "command_start",
                    "attempt_id": "a1",
                    "job_id": job_id or "local",
                    "array_task_id": array_task_id or "0",
                    "batch_index": 0,
                    "line_index": 0,
                    "config_id": "cfg_a",
                    "pid": 1234,
                    "tmux_session": None,
                    "assigned_cores": [0],
                    "executor": "process",
                }
            )
            progress_callback(
                {
                    "event": "command_complete",
                    "attempt_id": "a1",
                    "job_id": job_id or "local",
                    "array_task_id": array_task_id or "0",
                    "batch_index": 0,
                    "line_index": 0,
                    "config_id": "cfg_a",
                    "status": "success",
                    "duration_sec": 0.1,
                    "assigned_cores": [0],
                    "pid": 1234,
                    "executor": "process",
                }
            )
        return {
            "run_id": "progress_refresh",
            "job_id": job_id or "local",
            "array_task_id": array_task_id or "0",
            "executor": "process",
            "batches": list(getattr(payload, "batch_indices", [0])),
            "result_path": str(
                summary.run_root / "exec" / "results" / "task_local_0.jsonl"
            ),
            "total": 1,
            "success": 1,
            "failed": 0,
            "terminated": 0,
            "skipped_resume": 0,
            "skipped_not_selected": 0,
            "skipped_fail_fast": 0,
            "retry_scheduled": 0,
            "dry_run": False,
            "resume": False,
            "selected_config_ids": [],
            "policy": {},
            "fail_fast_triggered": False,
            "fail_fast_reason": None,
            "task_events_path": str(
                summary.run_root
                / "exec"
                / "batch_logs"
                / "task_local_0"
                / "task.events.jsonl"
            ),
            "task_stdout_log": str(
                summary.run_root
                / "exec"
                / "batch_logs"
                / "task_local_0"
                / "task.stdout.log"
            ),
            "task_stderr_log": str(
                summary.run_root
                / "exec"
                / "batch_logs"
                / "task_local_0"
                / "task.stderr.log"
            ),
        }

    monkeypatch.setattr("geryon.cli.Live", _FakeLive)
    monkeypatch.setattr("geryon.cli.execute_task_payload", _fake_execute_task_payload)

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--profile",
            "local_progress",
            "--profiles-file",
            str(profiles),
            "--format",
            "table",
        ]
    )
    assert rc == 0
    _ = capsys.readouterr().out
    assert _FakeLive.instances
    # Must refresh on command lifecycle events, not only payload boundaries.
    assert _FakeLive.instances[0].update_calls >= 6


def test_run_local_profile_defaults_cap_effective_parallelism(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('concurrency')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="concurrency_flags",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  local_cap:
    defaults:
      run_local:
        executor: tmux
        max_concurrent_tasks: 8
        cores_per_task: 2
        max_total_cores: 6
""",
    )

    captured_payloads: list[Any] = []

    def _fake_execute_task_payload(
        payload: Any,
        *,
        job_id: str | None = None,
        array_task_id: str | None = None,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        captured_payloads.append(payload)
        return {
            "run_id": "concurrency_flags",
            "job_id": job_id or "local",
            "array_task_id": array_task_id or "0",
            "executor": "tmux",
            "batches": list(getattr(payload, "batch_indices", [0])),
            "result_path": str(
                summary.run_root / "exec" / "results" / "task_local_0.jsonl"
            ),
            "total": 1,
            "success": 1,
            "failed": 0,
            "terminated": 0,
            "skipped_resume": 0,
            "skipped_not_selected": 0,
            "skipped_fail_fast": 0,
            "retry_scheduled": 0,
            "dry_run": False,
            "resume": False,
            "selected_config_ids": [],
            "policy": {},
            "fail_fast_triggered": False,
            "fail_fast_reason": None,
            "task_events_path": str(
                summary.run_root
                / "exec"
                / "batch_logs"
                / "task_local_0"
                / "task.events.jsonl"
            ),
            "task_stdout_log": str(
                summary.run_root
                / "exec"
                / "batch_logs"
                / "task_local_0"
                / "task.stdout.log"
            ),
            "task_stderr_log": str(
                summary.run_root
                / "exec"
                / "batch_logs"
                / "task_local_0"
                / "task.stderr.log"
            ),
        }

    monkeypatch.setattr("geryon.cli.execute_task_payload", _fake_execute_task_payload)

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--profile",
            "local_cap",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)
    assert captured_payloads
    payload = captured_payloads[0]
    assert payload.max_workers == 3
    assert payload.k_per_session == 2
    assert payload.max_total_cores == 6


def test_run_local_rejects_invalid_total_core_cap_from_profile(
    tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('invalid-cap')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="invalid_cap",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  bad_cap:
    defaults:
      run_local:
        cores_per_task: 4
        max_total_cores: 2
""",
    )

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--profile",
            "bad_cap",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "max_total_cores must be >= cores_per_task" in captured.err


@pytest.mark.parametrize(
    ("profiles_content", "expected_rc", "expected_snippet"),
    [
        (
            """
profiles:
  short:
    partition: gpu
    time_min: 60
""",
            0,
            '"count": 1',
        ),
        (
            """
profiles:
  bad:
    unknown_field: 123
""",
            2,
            "unknown keys",
        ),
    ],
)
def test_list_profiles_validation(
    tmp_path: Path,
    capsys,
    profiles_content: str,
    expected_rc: int,
    expected_snippet: str,
) -> None:
    profiles = tmp_path / "profiles.yaml"
    _write(profiles, profiles_content)

    rc = main(["list-profiles", "--profiles-file", str(profiles), "--format", "json"])
    assert rc == expected_rc

    captured = capsys.readouterr()
    if expected_rc == 0:
        assert expected_snippet in captured.out
    else:
        assert expected_snippet in captured.err


def test_list_profiles_accepts_run_defaults_blocks(tmp_path: Path, capsys) -> None:
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  local_fast:
    defaults:
      run_local:
        executor: process
        max_concurrent_tasks: 3
        cores_per_task: 2
        retry_on_status: [failed, terminated]
        progress: true
  a100_short:
    partition: gpu
    time_min: 60
    defaults:
      run_slurm:
        query_status: true
        sbatch_option:
          qos: debug
""",
    )

    rc = main(["list-profiles", "--profiles-file", str(profiles), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["profiles"]["local_fast"]["defaults"]["run_local"][
            "max_concurrent_tasks"
        ]
        == 3
    )
    assert (
        payload["profiles"]["a100_short"]["defaults"]["run_slurm"]["query_status"]
        is True
    )


def test_list_profiles_rejects_invalid_run_defaults_keys_and_types(
    tmp_path: Path, capsys
) -> None:
    bad_unknown = tmp_path / "profiles_bad_unknown.yaml"
    _write(
        bad_unknown,
        """
profiles:
  bad:
    defaults:
      run_local:
        max_workers: 2
""",
    )
    rc = main(
        ["list-profiles", "--profiles-file", str(bad_unknown), "--format", "json"]
    )
    assert rc == 2
    assert "unknown keys" in capsys.readouterr().err

    bad_type = tmp_path / "profiles_bad_type.yaml"
    _write(
        bad_type,
        """
profiles:
  bad:
    defaults:
      run_slurm:
        query_status: "yes"
""",
    )
    rc = main(["list-profiles", "--profiles-file", str(bad_type), "--format", "json"])
    assert rc == 2
    assert "must be a boolean" in capsys.readouterr().err


def test_list_profiles_rejects_null_required_run_default_ints(
    tmp_path: Path, capsys: Any
) -> None:
    bad_null = tmp_path / "profiles_bad_null.yaml"
    _write(
        bad_null,
        """
profiles:
  bad:
    defaults:
      run_local:
        max_concurrent_tasks: null
""",
    )

    rc = main(["list-profiles", "--profiles-file", str(bad_null), "--format", "json"])
    assert rc == 2
    assert "max_concurrent_tasks cannot be null" in capsys.readouterr().err


def test_run_local_profile_defaults_precedence(tmp_path: Path, capsys) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('local-defaults')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="local_defaults",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  local_fast:
    env:
      EXAMPLE_ENV: root
    defaults:
      run_local:
        executor: process
        max_concurrent_tasks: 3
        cores_per_task: 2
        max_retries: 2
""",
    )

    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--profile",
            "local_fast",
            "--profiles-file",
            str(profiles),
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    conc = payload["payloads"][0]["concurrency"]
    assert conc["max_concurrent_tasks"] == 3
    assert conc["cores_per_task"] == 2
    assert (
        payload["resolved_defaults_sources"]["max_concurrent_tasks"]
        == "profile.defaults"
    )
    assert payload["resolved_defaults_sources"]["max_retries"] == "profile.defaults"


def test_run_slurm_profile_precedence_and_queue_query(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('slurm')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="slurm_profile",
        batch_size=1,
    )

    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  a100_short:
    partition: gpu
    time_min: 240
    cpus_per_task: 8
    mem_gb: 64
    gpus_per_node: 1
    mail_user: profile@example.org
    mail_type: END,FAIL
    defaults:
      run_slurm:
        executor: process
        time_min: 480
        mail_type: END
        query_status: true
        sbatch_option:
          account: profile_project
          qos: high
          constraint: a100
          nodelist: gpu001
          exclusive: true
""",
    )

    import geryon.submitit_backend as backend

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(backend, "_load_submitit", lambda: _FakeSubmitit(fake_executor))

    def _fake_squeue(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[0] == "squeue"
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="3000|R|00:02:10|1|gpu|geryon|tester|None\n",
            stderr="",
        )

    monkeypatch.setattr("geryon.cli.subprocess.run", _fake_squeue)

    rc = main(
        [
            "run-slurm",
            "--run",
            str(summary.run_root),
            "--profile",
            "a100_short",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["partition"] == "gpu"
    assert payload["time_min"] == 240
    assert payload["cpus_per_task"] == 8
    assert payload["mem_gb"] == 64
    assert payload["gpus_per_node"] == 1
    assert payload["mail_user"] == "profile@example.org"
    assert payload["mail_type"] == "END,FAIL"
    assert payload["queue_status"]["source"] == "squeue"
    assert payload["queue_status"]["queried"] is True
    assert payload["queue_status"]["available"] is True
    assert payload["queue_status"]["by_state"] == {"R": 1}

    expected_sbatch = {
        "account": "profile_project",
        "constraint": "a100",
        "exclusive": True,
        "mail_type": "END,FAIL",
        "mail_user": "profile@example.org",
        "nodelist": "gpu001",
        "qos": "high",
    }
    assert payload["slurm_additional_parameters"] == expected_sbatch
    assert fake_executor.parameters["slurm_additional_parameters"] == expected_sbatch

    store = ArtifactStore(summary.run_root)
    submit_meta = json.loads(store.submitit_jobs_path.read_text(encoding="utf-8"))
    assert submit_meta["profile_name"] == "a100_short"
    assert submit_meta["slurm_additional_parameters"] == expected_sbatch


def test_run_slurm_reports_effective_config_and_source_of_truth(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('slurm-effective')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="slurm_effective",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  gpu:
    partition: gpu
    time_min: 30
    slurm_setup_cmds:
      - module load cuda/12.2
      - export SHARED=1
    defaults:
      run_slurm:
        sbatch_option:
          qos: debug
        slurm_setup_cmds:
          - module load cuda/12.2
          - export BASE=1
""",
    )

    import geryon.submitit_backend as backend

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(backend, "_load_submitit", lambda: _FakeSubmitit(fake_executor))

    rc = main(
        [
            "run-slurm",
            "--run",
            str(summary.run_root),
            "--profile",
            "gpu",
            "--profiles-file",
            str(profiles),
            "--sbatch-option",
            "qos=high",
            "--slurm-setup-cmd",
            "export SHARED=1",
            "--slurm-setup-cmd",
            "echo READY",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["effective_config"]["partition"] == "gpu"
    assert payload["effective_config"]["time_min"] == 30
    assert payload["slurm_additional_parameters"]["qos"] == "high"
    assert payload["source_of_truth"]["sbatch_option"]["qos"] == "cli"
    assert payload["slurm_setup"] == [
        "module load cuda/12.2",
        "export BASE=1",
        "export SHARED=1",
        "echo READY",
    ]
    assert any(
        "duplicate slurm setup command ignored" in warning
        for warning in payload["resolution_warnings"]
    )
    assert payload["launcher_log_path"].endswith("/exec/launcher.log")
    assert Path(payload["launcher_log_path"]).exists()


def test_run_slurm_rejects_reserved_sbatch_option_keys(tmp_path: Path, capsys) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('reserved-sbatch')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="reserved_sbatch",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  gpu:
    partition: gpu
    time_min: 30
    defaults:
      run_slurm:
        sbatch_option:
          mail_user: should_fail@example.org
""",
    )

    rc = main(
        [
            "run-slurm",
            "--run",
            str(summary.run_root),
            "--profile",
            "gpu",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 2
    assert "managed by dedicated fields" in capsys.readouterr().err


def test_run_slurm_rejects_unknown_config_id_before_submission(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('unknown-id')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="unknown_config_id",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  gpu:
    partition: gpu
    time_min: 30
""",
    )

    import geryon.submitit_backend as backend

    submitit_load_called = {"value": False}

    def _unexpected_submitit_load() -> Any:
        submitit_load_called["value"] = True
        raise AssertionError("submitit loader must not be called for invalid config-id")

    monkeypatch.setattr(backend, "_load_submitit", _unexpected_submitit_load)

    rc = main(
        [
            "run-slurm",
            "--run",
            str(summary.run_root),
            "--profile",
            "gpu",
            "--profiles-file",
            str(profiles),
            "--config-id",
            "does_not_exist",
            "--format",
            "json",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "Unknown config IDs selected" in captured.err
    assert submitit_load_called["value"] is False


def test_run_slurm_rejects_missing_env_script_before_submission(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('env-preflight')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="missing_env_script_slurm",
        batch_size=1,
    )
    missing_script = tmp_path / "missing_env.sh"
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        f"""
profiles:
  gpu:
    partition: gpu
    time_min: 30
    env_script: {missing_script}
""",
    )

    import geryon.submitit_backend as backend

    submitit_load_called = {"value": False}

    def _unexpected_submitit_load() -> Any:
        submitit_load_called["value"] = True
        raise AssertionError(
            "submitit loader must not be called for missing env script"
        )

    monkeypatch.setattr(backend, "_load_submitit", _unexpected_submitit_load)

    rc = main(
        [
            "run-slurm",
            "--run",
            str(summary.run_root),
            "--profile",
            "gpu",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "Environment setup script not found" in captured.err
    assert submitit_load_called["value"] is False


def test_run_local_rejects_retry_file_run_mismatch(tmp_path: Path, capsys) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('retry-mismatch')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    run_a = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="run_a",
        batch_size=1,
    )
    run_b = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="run_b",
        batch_size=1,
    )

    store_b = ArtifactStore(run_b.run_root)
    cfg_b = store_b.read_planned_configs()[0]
    retry_file = tmp_path / "retry_mismatch.json"
    retry_file.write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "run_root": str(run_a.run_root),
                "target_config_ids": [cfg_b.config_id],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "run-local",
            "--run",
            str(run_b.run_root),
            "--retry-file",
            str(retry_file),
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "Retry metadata run root mismatch" in captured.err


def test_status_collect_corrupt_jsonl_reporting_and_strict_mode(
    tmp_path: Path, capsys
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('corrupt-jsonl')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="corrupt_jsonl",
        batch_size=1,
    )
    store = ArtifactStore(summary.run_root)
    cfg = store.read_planned_configs()[0]
    result_path = store.task_result_file("local", "0")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "config_id": cfg.config_id,
                        "status": "success",
                        "attempt_id": "a1",
                        "start_time": "2026-01-01T00:00:00Z",
                        "end_time": "2026-01-01T00:00:01Z",
                        "batch_index": cfg.batch_index,
                        "line_index": cfg.line_index,
                    },
                    sort_keys=True,
                ),
                "{not-json}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = main(["status", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["corrupt_result_lines"] == 1

    rc = main(["collect", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    collect_payload = json.loads(capsys.readouterr().out)
    assert collect_payload["corrupt_result_lines"] == 1

    rc = main(
        [
            "status",
            "--run",
            str(summary.run_root),
            "--strict-jsonl",
            "--format",
            "json",
        ]
    )
    assert rc == 2
    status_err = capsys.readouterr().err
    assert "Corrupt JSONL line" in status_err

    rc = main(
        [
            "collect",
            "--run",
            str(summary.run_root),
            "--strict-jsonl",
            "--format",
            "json",
        ]
    )
    assert rc == 2
    collect_err = capsys.readouterr().err
    assert "Corrupt JSONL line" in collect_err


def test_collect_reports_batch_totals_in_json_and_table(
    tmp_path: Path, capsys: Any
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python3
  args: ["-c", "print('collect-batches')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
        - id: b
          params:
            x: 2
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="collect_batches",
        batch_size=1,
    )

    rc = main(["run-local", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)

    rc = main(["collect", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    collect_payload = json.loads(capsys.readouterr().out)
    assert collect_payload["batches"]["total"] == 2
    assert len(collect_payload["batches"]["by_batch"]) == 2

    rc = main(["collect", "--run", str(summary.run_root), "--format", "table"])
    assert rc == 0
    table_output = capsys.readouterr().out
    assert "Batches" in table_output
    assert re.search(r"Batches\s+│\s+2\b", table_output)


def test_run_slurm_uses_task_runtime_env_without_duplicating_slurm_setup(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('slurm-env-split')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="slurm_env_split",
        batch_size=1,
    )

    script = tmp_path / "bootstrap.sh"
    _write(script, "export FROM_SCRIPT=yes")

    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        f"""
profiles:
  gpu_profile:
    partition: gpu
    time_min: 60
    env_script: {script}
    env_setup_cmds:
      - export FROM_SETUP_CMD=yes
    env:
      FROM_PROFILE_ENV: enabled
    slurm_setup_cmds:
      - module load cuda/12.2
""",
    )

    import geryon.submitit_backend as backend

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(backend, "_load_submitit", lambda: _FakeSubmitit(fake_executor))

    rc = main(
        [
            "run-slurm",
            "--run",
            str(summary.run_root),
            "--profile",
            "gpu_profile",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["slurm_setup"] == ["module load cuda/12.2"]

    payloads_path = Path(payload["payloads_path"])
    payload_records = list(read_jsonl(payloads_path))
    assert len(payload_records) == 1
    task_payload = payload_records[0]
    assert task_payload["env_setup_script"] == str(script.resolve())
    assert task_payload["env_setup_commands"] == ["export FROM_SETUP_CMD=yes"]
    assert task_payload["env_vars"] == {"FROM_PROFILE_ENV": "enabled"}


def test_run_slurm_surfaces_submission_failure(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('submit-failure')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="slurm_submit_failure",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  gpu:
    partition: gpu
    time_min: 30
""",
    )

    class _FailingExecutor(_FakeExecutor):
        def map_array(self, fn: Any, payloads: Any) -> list[_FakeJob]:
            raise RuntimeError("submission boom")

    import geryon.submitit_backend as backend

    monkeypatch.setattr(
        backend,
        "_load_submitit",
        lambda: _FakeSubmitit(_FailingExecutor()),
    )

    rc = main(
        [
            "run-slurm",
            "--run",
            str(summary.run_root),
            "--profile",
            "gpu",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "Slurm job submission failed" in captured.err


def test_launch_runs_validate_plan_execute_in_order(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    call_order: list[str] = []

    class _Spec:
        base_command = "python -c pass"
        packs: tuple[Any, ...] = ()
        constraints: tuple[Any, ...] = ()
        predicates: tuple[Any, ...] = ()

    def _fake_validate(
        experiment: str, run_set: str | None = None
    ) -> tuple[Any, dict[str, Any]]:
        call_order.append("validate")
        return _Spec(), {"ok": True, "run_set": run_set}

    def _fake_plan(
        *,
        experiment_path: str,
        out_dir: str,
        batch_size: int,
        run_id: str | None = None,
        run_set: str | None = None,
        dry_run: bool = False,
        preview_count: int = 0,
    ) -> PlanSummary:
        call_order.append("plan")
        run_root = Path(out_dir) / "runs" / (run_id or "auto")
        return PlanSummary(
            run_id=run_id or "auto",
            run_root=run_root,
            total_configs=2,
            total_batches=1,
        )

    def _fake_execute(args: Any) -> dict[str, Any]:
        call_order.append("execute")
        assert str(args.run).endswith("/runs/launch_order")
        return {"payloads": [{"total": 2, "success": 2}]}

    monkeypatch.setattr(
        "geryon.cli.parse_experiment_yaml_with_diagnostics", _fake_validate
    )
    monkeypatch.setattr("geryon.cli.plan_experiment", _fake_plan)
    monkeypatch.setattr("geryon.cli._execute_run_local", _fake_execute)

    rc = main(
        [
            "launch",
            "--experiment",
            str(tmp_path / "experiment.yaml"),
            "--out",
            str(tmp_path / "out"),
            "--batch-size",
            "8",
            "--run-id",
            "launch_order",
            "--backend",
            "local",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["run_id"] == "launch_order"
    assert payload["execution"]["payloads"][0]["success"] == 2
    assert call_order == ["validate", "plan", "execute"]


def test_launch_dry_run_skips_backend_execution(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('launch-dry-run')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params: {x: 1}
""",
    )

    def _unexpected_execute(_: Any) -> dict[str, Any]:
        raise AssertionError("backend execution must be skipped for launch --dry-run")

    monkeypatch.setattr("geryon.cli._execute_run_local", _unexpected_execute)

    rc = main(
        [
            "launch",
            "--experiment",
            str(experiment),
            "--out",
            str(tmp_path / "out"),
            "--batch-size",
            "4",
            "--backend",
            "local",
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution"]["skipped"] is True
    assert payload["execution"]["reason"] == "dry_run"


def test_recover_matches_rerun_selection_and_dry_run_skips_backend(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    marker_dir = tmp_path / "markers"
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        f"""
schema:
  version: 4
command:
  program: python3
  args:
    - -c
    - |
      import pathlib
      import sys
      mode = "ok"
      marker_dir = None
      for arg in sys.argv[1:]:
          if arg.startswith("mode="):
              mode = arg.split("=", 1)[1]
          if arg.startswith("marker_dir="):
              marker_dir = arg.split("=", 1)[1]
      if mode == "flaky":
          marker_path = pathlib.Path(marker_dir) / "flaky.once"
          marker_path.parent.mkdir(parents=True, exist_ok=True)
          if not marker_path.exists():
              marker_path.write_text("first-failure", encoding="utf-8")
              sys.exit(7)
      sys.exit(0)
select:
  packs:
    - name: exit_mode
      options:
        - id: ok
          params:
            mode: ok
            marker_dir: {json.dumps(str(marker_dir))}
        - id: flaky
          params:
            mode: flaky
            marker_dir: {json.dumps(str(marker_dir))}
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="recover_selection",
        batch_size=2,
    )
    rc = main(
        [
            "run-local",
            "--run",
            str(summary.run_root),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)

    rc = main(
        [
            "rerun",
            "--run",
            str(summary.run_root),
            "--status",
            "failed",
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    rerun_payload = json.loads(capsys.readouterr().out)

    def _unexpected_execute(_: Any) -> dict[str, Any]:
        raise AssertionError("backend execution must be skipped for recover --dry-run")

    monkeypatch.setattr("geryon.cli._execute_run_local", _unexpected_execute)
    rc = main(
        [
            "recover",
            "--run",
            str(summary.run_root),
            "--status",
            "failed",
            "--backend",
            "local",
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    recover_payload = json.loads(capsys.readouterr().out)
    assert recover_payload["execution"]["skipped"] is True
    assert recover_payload["retry"]["retry_file"] is None
    assert (
        recover_payload["retry"]["target_config_ids"]
        == rerun_payload["target_config_ids"]
    )
    assert (
        recover_payload["retry"]["target_batch_indices"]
        == rerun_payload["target_batch_indices"]
    )


def test_recover_backend_slurm_passes_overrides(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('recover-slurm')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="recover_slurm",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  gpu:
    partition: gpu
    time_min: 30
    defaults:
      run_slurm:
        query_status: true
        sbatch_option:
          qos: debug
""",
    )

    import geryon.submitit_backend as backend

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(backend, "_load_submitit", lambda: _FakeSubmitit(fake_executor))
    monkeypatch.setattr(
        "geryon.cli.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="3000|R|00:00:10|1|gpu|geryon|tester|None\n",
            stderr="",
        ),
    )

    rc = main(
        [
            "recover",
            "--run",
            str(summary.run_root),
            "--status",
            "missing",
            "--backend",
            "slurm",
            "--profile",
            "gpu",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    execution = payload["execution"]
    assert execution["partition"] == "gpu"
    assert execution["time_min"] == 30
    assert execution["queue_status"]["queried"] is True
    assert execution["slurm_additional_parameters"]["qos"] == "debug"


def test_queue_and_queue_refresh_commands(tmp_path: Path, capsys, monkeypatch) -> None:
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        """
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('queue-cmds')"]
select:
  packs:
    - name: one
      options:
        - id: a
          params:
            x: 1
""",
    )
    summary = plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id="queue_cmds",
        batch_size=1,
    )
    profiles = tmp_path / "profiles.yaml"
    _write(
        profiles,
        """
profiles:
  gpu:
    partition: gpu
    time_min: 30
""",
    )

    import geryon.submitit_backend as backend

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(backend, "_load_submitit", lambda: _FakeSubmitit(fake_executor))

    rc = main(
        [
            "run-slurm",
            "--run",
            str(summary.run_root),
            "--profile",
            "gpu",
            "--profiles-file",
            str(profiles),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["submission_id"]

    monkeypatch.setattr(
        "geryon.cli.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="3000|R|00:00:10|1|gpu|geryon|tester|None\n",
            stderr="",
        ),
    )

    rc = main(["queue", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    queue_payload = json.loads(capsys.readouterr().out)
    assert queue_payload["queried"] is True
    assert queue_payload["available"] is True
    assert queue_payload["by_state"] == {"R": 1}

    rc = main(["queue-refresh", "--run", str(summary.run_root), "--format", "json"])
    assert rc == 0
    refresh_payload = json.loads(capsys.readouterr().out)
    assert Path(refresh_payload["queue_status_path"]).exists()
    assert Path(refresh_payload["queue_status_snapshot_path"]).exists()


def test_help_for_run_commands_is_grouped(capsys: Any) -> None:
    from geryon.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["run-local", "--help"])
    assert excinfo.value.code == 0
    local_help = capsys.readouterr().out
    assert "Selection:" in local_help
    assert "Profile:" in local_help
    assert "Examples:" in local_help

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["run-slurm", "--help"])
    assert excinfo.value.code == 0
    slurm_help = capsys.readouterr().out
    assert "Profile:" in slurm_help
