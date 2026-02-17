from __future__ import annotations

from pathlib import Path

import pytest

from geryon.models import ConfigError
from geryon.planner import plan_experiment
from geryon.store import ArtifactStore


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _plan(
    *,
    tmp_path: Path,
    experiment_text: str,
    run_id: str = "merge_modes",
    dry_run: bool = True,
) -> object:
    experiment = tmp_path / "experiment.yaml"
    _write(experiment, experiment_text)
    return plan_experiment(
        experiment_path=experiment,
        out_dir=tmp_path / "out",
        run_id=run_id,
        batch_size=8,
        dry_run=dry_run,
    )


def test_none_mode_rejects_any_overlap(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="merge.mode=none does not allow overlaps"):
        _plan(
            tmp_path=tmp_path,
            experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: none
select:
  packs:
    - name: a
      options:
        - id: a1
          params:
            train:
              lr: 0.1
    - name: b
      options:
        - id: b1
          params:
            train:
              lr: 0.2
""",
        )


def test_none_mode_rejects_equal_overlap(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="merge.mode=none does not allow overlaps"):
        _plan(
            tmp_path=tmp_path,
            experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: none
select:
  packs:
    - name: a
      options:
        - id: a1
          params:
            x:
              y: 1
    - name: b
      options:
        - id: b1
          params:
            x:
              y: 1
""",
        )


def test_none_mode_allows_disjoint_packs(tmp_path: Path) -> None:
    summary = _plan(
        tmp_path=tmp_path,
        experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: none
select:
  packs:
    - name: model
      options:
        - id: resnet
          params:
            model:
              name: resnet18
    - name: training
      options:
        - id: default
          params:
            train:
              lr: 0.001
""",
    )
    assert summary.total_configs == 1


def test_merge_mode_resolves_by_priority(tmp_path: Path) -> None:
    summary = _plan(
        tmp_path=tmp_path,
        run_id="priority_resolve",
        dry_run=False,
        experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: merge
select:
  packs:
    - name: base
      priority: 0
      options:
        - id: base
          params:
            train:
              lr: 0.001
              wd: 0.0
    - name: optimizer
      priority: 10
      options:
        - id: adamw
          params:
            train:
              lr: 0.0005
              wd: 0.01
""",
    )
    store = ArtifactStore(summary.run_root)
    cfg = store.read_planned_configs()[0]
    assert cfg.params["train"]["lr"] == 0.0005
    assert cfg.params["train"]["wd"] == 0.01


def test_merge_mode_equal_priority_conflict_is_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Parameter conflict at equal priority"):
        _plan(
            tmp_path=tmp_path,
            experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: merge
select:
  packs:
    - name: a
      priority: 0
      options:
        - id: a1
          params:
            train:
              lr: 0.001
    - name: b
      priority: 0
      options:
        - id: b1
          params:
            train:
              lr: 0.002
""",
        )


def test_merge_mode_equal_values_allowed(tmp_path: Path) -> None:
    summary = _plan(
        tmp_path=tmp_path,
        experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: merge
select:
  packs:
    - name: a
      priority: 0
      options:
        - id: a1
          params:
            x:
              y: 1
    - name: b
      priority: 5
      options:
        - id: b1
          params:
            x:
              y: 1
""",
    )
    assert summary.total_configs == 1


def test_merge_mode_defaults_always_overridden(tmp_path: Path) -> None:
    summary = _plan(
        tmp_path=tmp_path,
        run_id="defaults_override",
        dry_run=False,
        experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
defaults:
  params:
    train:
      lr: 0.1
merge:
  mode: merge
select:
  packs:
    - name: optimizer
      options:
        - id: opt1
          params:
            train:
              lr: 0.001
""",
    )
    store = ArtifactStore(summary.run_root)
    cfg = store.read_planned_configs()[0]
    assert cfg.params["train"]["lr"] == 0.001


def test_merge_mode_duplicate_packs_auto_merge(tmp_path: Path) -> None:
    summary = _plan(
        tmp_path=tmp_path,
        experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: merge
select:
  packs:
    - name: seed
      options:
        - id: s1
          params: {seed: 1}
    - name: seed
      options:
        - id: s2
          params: {seed: 2}
""",
    )
    assert summary.total_configs == 2


def test_none_mode_duplicate_packs_error(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigError, match="Duplicate pack.*not allowed.*merge.mode=none"
    ):
        _plan(
            tmp_path=tmp_path,
            experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: none
select:
  packs:
    - name: seed
      options:
        - id: s1
          params: {seed: 1}
    - name: seed
      options:
        - id: s2
          params: {seed: 2}
""",
        )


def test_strategies_with_priority(tmp_path: Path) -> None:
    summary = _plan(
        tmp_path=tmp_path,
        run_id="strategy_priority",
        dry_run=False,
        experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
merge:
  mode: merge
  strategies:
    model: deep_merge
    callbacks: append_unique
    tags: set_union
    logging.level: replace
select:
  packs:
    - name: base
      priority: 0
      options:
        - id: base
          params:
            model:
              hidden: 256
              dropout: 0.1
            callbacks: [checkpoint, lr_monitor]
            tags: [baseline, vision]
            logging:
              level: INFO
    - name: tweak
      priority: 10
      options:
        - id: tweak
          params:
            model:
              dropout: 0.2
            callbacks: [checkpoint, grad_norm]
            tags: [vision, ablation]
            logging:
              level: DEBUG
""",
    )
    store = ArtifactStore(summary.run_root)
    cfg = store.read_planned_configs()[0]
    assert cfg.params["model"]["hidden"] == 256
    assert cfg.params["model"]["dropout"] == 0.2
    assert cfg.params["callbacks"] == ["checkpoint", "lr_monitor", "grad_norm"]
    assert set(cfg.params["tags"]) == {"baseline", "vision", "ablation"}
    assert cfg.params["logging"]["level"] == "DEBUG"


def test_delete_sentinel_with_priority(tmp_path: Path) -> None:
    summary = _plan(
        tmp_path=tmp_path,
        run_id="delete_sentinel",
        dry_run=False,
        experiment_text="""
schema:
  version: 4
command:
  program: python
  args: ["-c", "print('x')"]
defaults:
  params:
    train:
      lr: 0.1
      scheduler: cosine
merge:
  mode: merge
  delete_sentinel: "__drop__"
select:
  packs:
    - name: base
      priority: 0
      options:
        - id: base
          params:
            train:
              lr: 0.001
              scheduler: "__drop__"
""",
    )
    store = ArtifactStore(summary.run_root)
    cfg = store.read_planned_configs()[0]
    assert cfg.params["train"]["lr"] == 0.001
    assert "scheduler" not in cfg.params["train"]
