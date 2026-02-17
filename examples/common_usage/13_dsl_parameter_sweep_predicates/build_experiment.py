#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from geryon.dsl import Experiment, Predicate, PredicateArg, pack_map, pack_param_values


def build_experiment() -> Experiment:
    # Base sweep size before predicates:
    # model (2) * train_lr (4) * batch_size (2) * seed (2) = 32 configs.
    # Predicate guardrails reduce that to 8 valid configs.
    return (
        Experiment()
        .command("python3", "../../common/dummy_hydra_app.py")
        .defaults(
            params={"data": {"dataset": "cifar100"}, "train": {"epochs": 50}},
            tags=["demo:dsl", "demo:predicate_sweep"],
        )
        .add_pack(
            pack_map(
                pack_name="model",
                options={
                    "resnet18": {"model": {"family": "cnn", "name": "resnet18"}},
                    "vit_tiny": {
                        "model": {"family": "transformer", "name": "vit_tiny"}
                    },
                },
            )
        )
        .add_pack(
            pack_param_values(
                pack_name="train_lr",
                param_path="train.lr",
                values=[0.0001, 0.0003, 0.001, 0.003],
                id_prefix="lr",
            )
        )
        .add_pack(
            pack_param_values(
                pack_name="batch_size",
                param_path="train.batch_size",
                values=[64, 128],
                id_prefix="bs",
            )
        )
        .add_pack(
            pack_param_values(
                pack_name="seed",
                param_path="seed",
                values=[1, 2],
                id_prefix="seed",
            )
        )
        .add_predicate(
            Predicate(
                predicate_id="model_lr_batch_guardrails",
                args={
                    "model": PredicateArg.pack_id("model"),
                    "lr": PredicateArg.param("train.lr"),
                    "batch": PredicateArg.param("train.batch_size"),
                },
                expr={
                    "or": [
                        {
                            "and": [
                                {"eq": ["$model", "resnet18"]},
                                {"in": ["$lr", [0.001, 0.003]]},
                                {"eq": ["$batch", 128]},
                            ]
                        },
                        {
                            "and": [
                                {"eq": ["$model", "vit_tiny"]},
                                {"in": ["$lr", [0.0001, 0.0003]]},
                                {"eq": ["$batch", 64]},
                            ]
                        },
                    ]
                },
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate experiment.yaml for the DSL predicate sweep example"
    )
    parser.add_argument("--out", default="experiment.yaml", help="Output YAML path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    experiment = build_experiment()
    experiment.to_yaml_file(out_path, validate_with_geryon=True, max_configs=64)

    print(f"wrote DSL experiment to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
