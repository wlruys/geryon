from __future__ import annotations

from pathlib import Path

import pytest

from geryon.interface import Experiment, Option, Pack, Predicate, PredicateArg, require
from geryon.models import ConfigError
from geryon.planner import plan_experiment
from geryon.store import ArtifactStore


def test_option_id_alias_and_legacy_field() -> None:
    option_alias = Option(id="resnet18", params={"model": {"name": "resnet18"}})
    assert option_alias.id == "resnet18"
    assert option_alias.option_id == "resnet18"

    option_legacy = Option(option_id="vit_tiny", params={"model": {"name": "vit_tiny"}})
    assert option_legacy.id == "vit_tiny"
    assert option_legacy.option_id == "vit_tiny"

    with pytest.raises(ConfigError, match="exactly one of 'id' or 'option_id'"):
        Option(id="x", option_id="y", params={"a": 1})


def test_predicate_id_alias_and_legacy_field() -> None:
    pred_alias = Predicate(
        id="lr_guard",
        args={"arch": PredicateArg.pack_id("architecture")},
        expr={"eq": ["$arch", "resnet18"]},
    )
    assert pred_alias.id == "lr_guard"
    assert pred_alias.predicate_id == "lr_guard"

    pred_legacy = Predicate(
        predicate_id="legacy_guard",
        args={"arch": PredicateArg.pack_id("architecture")},
        expr={"eq": ["$arch", "resnet18"]},
    )
    assert pred_legacy.id == "legacy_guard"
    assert pred_legacy.predicate_id == "legacy_guard"

    with pytest.raises(ConfigError, match="exactly one of 'id' or 'predicate_id'"):
        Predicate(
            id="a",
            predicate_id="b",
            args={"arch": PredicateArg.pack_id("architecture")},
            expr={"eq": ["$arch", "resnet18"]},
        )


def test_require_shorthand_builds_predicate_and_plans(tmp_path: Path) -> None:
    lr_guardrails = require(
        rules=[
            ({"architecture": "resnet18"}, {"train.lr": ("between", 0.0005, 0.002)}),
            ({"architecture": "vit_tiny"}, {"train.lr": ("between", 0.0001, 0.0005)}),
        ],
        predicate_id="lr_guardrails",
    )
    pred_doc = lr_guardrails.to_dict()
    assert pred_doc["id"] == "lr_guardrails"
    assert set(pred_doc["args"].keys()) == {"pack_architecture", "param_train_lr"}
    assert pred_doc["expr"]["or"][0]["and"][0] == {
        "eq": ["$pack_architecture", "resnet18"]
    }

    experiment = (
        Experiment()
        .command("python3", "dummy.py")
        .add_pack(
            Pack(
                name="architecture",
                options=(
                    Option(id="resnet18", params={"model": {"name": "resnet18"}}),
                    Option(id="vit_tiny", params={"model": {"name": "vit_tiny"}}),
                ),
            )
        )
        .add_pack(
            Pack(
                name="profile",
                options=(
                    Option(id="lr_high", params={"train": {"lr": 0.001}}),
                    Option(id="lr_low", params={"train": {"lr": 0.0003}}),
                ),
            )
        )
        .add_predicate(lr_guardrails)
    )

    experiment_path = tmp_path / "experiment.yaml"
    experiment.to_yaml_file(experiment_path, validate_with_geryon=True)

    summary = plan_experiment(
        experiment_path=experiment_path,
        out_dir=tmp_path / "out",
        run_id="interface_require",
        batch_size=8,
    )
    assert summary.total_configs == 2

    store = ArtifactStore(summary.run_root)
    configs = store.read_planned_configs()
    pairs = {
        (cfg.selected_options["architecture"], cfg.selected_options["profile"])
        for cfg in configs
    }
    assert pairs == {("resnet18", "lr_high"), ("vit_tiny", "lr_low")}


def test_variant_patch_packs_replaces_only_target_pack(tmp_path: Path) -> None:
    experiment = (
        Experiment()
        .command("python3", "dummy.py")
        .add_pack(
            Pack(
                name="dataset",
                options=(
                    Option(
                        id="cifar10_aug",
                        params={"data": {"name": "cifar10", "augmentation": "strong"}},
                    ),
                    Option(
                        id="cifar10_light",
                        params={"data": {"name": "cifar10", "augmentation": "light"}},
                    ),
                ),
            )
        )
        .add_pack(
            Pack(
                name="architecture",
                options=(
                    Option(id="resnet18", params={"model": {"name": "resnet18"}}),
                ),
            )
        )
        .add_pack(
            Pack(
                name="regularization",
                options=(Option(id="dropout_on", params={"train": {"dropout": 0.2}}),),
            )
        )
        .variant(
            "no_augmentation",
            patch_packs={
                "dataset": {
                    "options": [
                        {
                            "id": "cifar10_noaug",
                            "params": {
                                "data": {"name": "cifar10", "augmentation": "none"}
                            },
                        }
                    ]
                }
            },
        )
    )

    experiment_path = tmp_path / "experiment.yaml"
    experiment.to_yaml_file(experiment_path, validate_with_geryon=True)

    baseline = plan_experiment(
        experiment_path=experiment_path,
        out_dir=tmp_path / "out",
        run_id="interface_patch_base",
        batch_size=8,
    )
    assert baseline.total_configs == 2

    no_aug = plan_experiment(
        experiment_path=experiment_path,
        out_dir=tmp_path / "out",
        run_id="interface_patch_variant",
        run_set="no_augmentation",
        batch_size=8,
    )
    assert no_aug.total_configs == 1

    store = ArtifactStore(no_aug.run_root)
    configs = store.read_planned_configs()
    assert {cfg.selected_options["dataset"] for cfg in configs} == {"cifar10_noaug"}
    assert {cfg.selected_options["architecture"] for cfg in configs} == {"resnet18"}
    assert {cfg.selected_options["regularization"] for cfg in configs} == {"dropout_on"}


def test_yaml_patch_packs_plans_end_to_end(tmp_path: Path) -> None:
    experiment_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "usage"
        / "02_ablation_study"
        / "experiment.yaml"
    )
    summary = plan_experiment(
        experiment_path=experiment_path,
        out_dir=tmp_path / "out",
        run_id="yaml_patch",
        run_set="no_augmentation",
        batch_size=8,
    )
    assert summary.total_configs == 1

    store = ArtifactStore(summary.run_root)
    configs = store.read_planned_configs()
    assert {cfg.selected_options["dataset"] for cfg in configs} == {"cifar10_noaug"}
    assert {cfg.selected_options["regularization"] for cfg in configs} == {"dropout_on"}


def test_patch_packs_missing_pack_raises(tmp_path: Path) -> None:
    experiment = (
        Experiment()
        .command("python3", "dummy.py")
        .add_pack(
            Pack(
                name="dataset",
                options=(Option(id="cifar10", params={"data": {"name": "cifar10"}}),),
            )
        )
        .variant(
            "bad_patch",
            patch_packs={
                "missing_pack": {
                    "options": [{"id": "x", "params": {"data": {"name": "x"}}}],
                }
            },
        )
    )

    experiment_path = tmp_path / "experiment.yaml"
    experiment.to_yaml_file(experiment_path)

    with pytest.raises(
        ConfigError, match="does not match any pack name in select.packs"
    ):
        plan_experiment(
            experiment_path=experiment_path,
            out_dir=tmp_path / "out",
            run_id="interface_bad_patch",
            run_set="bad_patch",
            batch_size=8,
        )


def test_interface_v4_import_and_definition_maps() -> None:
    exp = (
        Experiment()
        .add_import("defs/shared.yaml", package="common")
        .add_option_set(
            "seeds",
            [
                Option(id="s1", params={"seed": 1}),
            ],
        )
        .add_pack_def(
            "seed_pack",
            Pack(
                name="seed",
                options=(Option(id="s1", params={"seed": 1}),),
            ),
        )
        .add_group_def(
            "core",
            [
                {"ref": "seed_pack"},
            ],
        )
        .command("python3", "dummy.py")
        .select_group("common.core")
    )

    doc = exp.to_dict(strict=True)
    assert doc["schema"]["version"] == 4
    assert doc["imports"] == [{"path": "defs/shared.yaml", "package": "common"}]
    assert "option_sets" in doc
    assert "packs" in doc
    assert "groups" in doc
    assert "registry" not in doc
    assert "package" not in doc


def test_interface_rejects_removed_registry_and_package_keys() -> None:
    with pytest.raises(ConfigError, match="registry is removed in schema v4"):
        Experiment.from_dict(
            {
                "schema": {"version": 4},
                "registry": {"option_sets": {}},
                "command": {"program": "python3", "args": []},
                "select": {"packs": []},
            }
        )

    with pytest.raises(ConfigError, match="package is removed in schema v4"):
        Experiment.from_dict(
            {
                "schema": {"version": 4},
                "package": "common",
                "command": {"program": "python3", "args": []},
                "select": {"packs": []},
            }
        )
