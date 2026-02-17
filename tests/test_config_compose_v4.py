from __future__ import annotations

from pathlib import Path

import pytest

from geryon.config_compose import compose_experiment_data
from geryon.models import ConfigError


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_flat_defs_expand_with_import_package_scope(tmp_path: Path) -> None:
    defs = tmp_path / "defs"
    _write(
        defs / "options.yaml",
        """
option_sets:
  arch:
    - id: resnet18
      params:
        model:
          name: resnet18
  optim:
    - id: adam
      params:
        train:
          optimizer: adam
  seeds:
    - id: s1
      params:
        seed: 1
""",
    )
    _write(
        defs / "packs.yaml",
        """
packs:
  architecture_pack:
    name: architecture
    options_from:
      - ref: arch
  optimizer_pack:
    name: optimizer
    options_from:
      - ref: optim
  seed_pack:
    name: seed
    options_from:
      - ref: seeds
""",
    )
    _write(
        defs / "groups.yaml",
        """
groups:
  core:
    - ref: optimizer_pack
    - ref: seed_pack
""",
    )
    _write(
        tmp_path / "experiment.yaml",
        """
schema:
  version: 4
imports:
  - path: defs/options.yaml
    package: common
  - path: defs/packs.yaml
    package: common
  - path: defs/groups.yaml
    package: common
select:
  groups: [common.core]
  packs:
    - ref: common.architecture_pack
""",
    )

    composed, diagnostics = compose_experiment_data(tmp_path / "experiment.yaml")
    assert [pack["name"] for pack in composed["packs"]] == [
        "optimizer",
        "seed",
        "architecture",
    ]
    assert diagnostics["registry"]["option_sets"] == [
        "common.arch",
        "common.optim",
        "common.seeds",
    ]
    assert diagnostics["registry"]["packs"] == [
        "common.architecture_pack",
        "common.optimizer_pack",
        "common.seed_pack",
    ]
    assert diagnostics["registry"]["groups"] == ["common.core"]


def test_same_import_file_can_be_reused_under_different_packages(
    tmp_path: Path,
) -> None:
    defs = tmp_path / "defs"
    _write(
        defs / "shared.yaml",
        """
option_sets:
  seeds:
    - id: s1
      params:
        seed: 1
packs:
  seed_pack:
    name: seed
    options_from:
      - ref: seeds
""",
    )
    _write(
        tmp_path / "experiment.yaml",
        """
schema:
  version: 4
imports:
  - path: defs/shared.yaml
    package: a
  - path: defs/shared.yaml
    package: b
select:
  packs:
    - ref: a.seed_pack
      name: seed_a
    - ref: b.seed_pack
      name: seed_b
""",
    )

    composed, diagnostics = compose_experiment_data(tmp_path / "experiment.yaml")
    assert [pack["name"] for pack in composed["packs"]] == ["seed_a", "seed_b"]
    assert diagnostics["registry"]["option_sets"] == ["a.seeds", "b.seeds"]
    assert diagnostics["registry"]["packs"] == ["a.seed_pack", "b.seed_pack"]


def test_duplicate_qualified_definition_names_fail(tmp_path: Path) -> None:
    defs = tmp_path / "defs"
    _write(
        defs / "a.yaml",
        """
option_sets:
  arch:
    - id: a
      params:
        model:
          name: a
""",
    )
    _write(
        defs / "b.yaml",
        """
option_sets:
  arch:
    - id: b
      params:
        model:
          name: b
""",
    )
    _write(
        tmp_path / "experiment.yaml",
        """
schema:
  version: 4
imports:
  - path: defs/a.yaml
    package: common
  - path: defs/b.yaml
    package: common
select:
  packs: []
""",
    )

    with pytest.raises(
        ConfigError, match="Duplicate option_set definition 'common.arch'"
    ):
        compose_experiment_data(tmp_path / "experiment.yaml")


def test_import_cycle_is_reported(tmp_path: Path) -> None:
    defs = tmp_path / "defs"
    _write(
        defs / "a.yaml",
        """
imports:
  - path: b.yaml
    package: common
option_sets: {}
""",
    )
    _write(
        defs / "b.yaml",
        """
imports:
  - path: a.yaml
    package: common
option_sets: {}
""",
    )
    _write(
        tmp_path / "experiment.yaml",
        """
schema:
  version: 4
imports:
  - path: defs/a.yaml
    package: common
select:
  packs: []
""",
    )

    with pytest.raises(ConfigError, match="Import cycle detected"):
        compose_experiment_data(tmp_path / "experiment.yaml")


@pytest.mark.parametrize(
    ("content", "pattern"),
    [
        (
            """
schema:
  version: 4
registry:
  option_sets: {}
select:
  packs: []
""",
            "'registry' is removed in schema v4",
        ),
        (
            """
schema:
  version: 4
package: common
select:
  packs: []
""",
            "'package' is removed in schema v4",
        ),
        (
            """
schema:
  version: 4
imports:
  - defs/options.yaml
select:
  packs: []
""",
            r"imports\[0\] must be a mapping",
        ),
    ],
)
def test_migration_and_import_shape_errors(
    tmp_path: Path, content: str, pattern: str
) -> None:
    _write(tmp_path / "experiment.yaml", content)
    with pytest.raises(ConfigError, match=pattern):
        compose_experiment_data(tmp_path / "experiment.yaml")
