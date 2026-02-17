from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import yaml

from geryon.dsl.emit import dump_yaml, write_yaml
from geryon.dsl.specs import (
    ConstraintRule,
    DefaultsSpec,
    MergeSpec,
    Option,
    Pack,
    PackSelector,
    Predicate,
    RunSetSpec,
)
from geryon.dsl.validate import validate_document
from geryon.models import ConfigError

DSL_API_VERSION = "0.1.0"
SUPPORTED_SCHEMA_VERSION = 4


class Experiment:
    """Builder for geryon experiment documents.

    This DSL always lowers into canonical schema-v4 YAML.
    High-level helpers should compile into explicit select/pack/options structures.
    """

    def __init__(
        self,
        *,
        unknown_key_behavior: str = "error",
        document: Mapping[str, Any] | None = None,
    ) -> None:
        if document is None:
            if unknown_key_behavior not in {"error", "warn"}:
                raise ConfigError("unknown_key_behavior must be 'error' or 'warn'")
            self._doc: dict[str, Any] = {
                "schema": {
                    "version": SUPPORTED_SCHEMA_VERSION,
                    "unknown_key_behavior": unknown_key_behavior,
                }
            }
            return

        self._doc = validate_document(document)
        schema = dict(self._doc.get("schema", {}))
        schema.setdefault("version", SUPPORTED_SCHEMA_VERSION)
        schema.setdefault("unknown_key_behavior", "error")
        self._doc["schema"] = schema

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Experiment":
        return cls(document=data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Experiment":
        yaml_path = Path(path)
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{yaml_path}: YAML root must be a mapping")
        return cls(document=raw)

    @property
    def document(self) -> dict[str, Any]:
        return deepcopy(self._doc)

    def schema_unknown_key_behavior(self, behavior: str) -> "Experiment":
        if behavior not in {"error", "warn"}:
            raise ConfigError("unknown_key_behavior must be 'error' or 'warn'")
        schema = dict(self._doc.get("schema", {}))
        schema["version"] = SUPPORTED_SCHEMA_VERSION
        schema["unknown_key_behavior"] = behavior
        self._doc["schema"] = schema
        return self

    def add_import(
        self, import_path: str, *, package: str | None = None
    ) -> "Experiment":
        if not isinstance(import_path, str) or not import_path.strip():
            raise ConfigError("import_path must be a non-empty string")
        imports = list(self._doc.get("imports", []))
        import_item: dict[str, Any] = {"path": import_path.strip()}
        if package is not None:
            if not isinstance(package, str) or not package.strip():
                raise ConfigError("package must be a non-empty string when provided")
            import_item["package"] = package.strip()
        imports.append(import_item)
        self._doc["imports"] = imports
        return self

    def command(self, program: str, *args: str) -> "Experiment":
        if not isinstance(program, str) or not program.strip():
            raise ConfigError("program must be a non-empty string")
        argv: list[str] = []
        for idx, item in enumerate(args):
            if not isinstance(item, str):
                raise ConfigError(f"args[{idx}] must be a string")
            argv.append(item)
        self._doc["command"] = {"program": program.strip(), "args": argv}
        return self

    def defaults(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
    ) -> "Experiment":
        spec = DefaultsSpec(params=dict(params or {}), tags=tuple(tags or ()))
        section = spec.to_dict()
        if section:
            self._doc["defaults"] = section
        else:
            self._doc.pop("defaults", None)
        return self

    def merge(self, spec: MergeSpec | None = None, **kwargs: Any) -> "Experiment":
        merge_spec = spec if spec is not None else MergeSpec(**kwargs)
        section = merge_spec.to_dict()
        if section:
            self._doc["merge"] = section
        else:
            self._doc.pop("merge", None)
        return self

    def _ensure_select(self) -> dict[str, Any]:
        select = dict(self._doc.get("select", {}))
        select.setdefault("groups", [])
        select.setdefault("packs", [])
        self._doc["select"] = select
        return select

    def select_group(self, group_ref: str) -> "Experiment":
        if not isinstance(group_ref, str) or not group_ref.strip():
            raise ConfigError("group_ref must be a non-empty string")
        select = self._ensure_select()
        groups = list(select.get("groups", []))
        groups.append(group_ref.strip())
        select["groups"] = groups
        self._doc["select"] = select
        return self

    def add_pack(self, pack: Pack) -> "Experiment":
        if not isinstance(pack, Pack):
            raise ConfigError("add_pack expects a Pack")
        select = self._ensure_select()
        packs = list(select.get("packs", []))
        packs.append(pack.to_dict())
        select["packs"] = packs
        self._doc["select"] = select
        return self

    def add_pack_selector(self, selector: PackSelector) -> "Experiment":
        if not isinstance(selector, PackSelector):
            raise ConfigError("add_pack_selector expects a PackSelector")
        select = self._ensure_select()
        packs = list(select.get("packs", []))
        packs.append(selector.to_dict())
        select["packs"] = packs
        self._doc["select"] = select
        return self

    def add_pack_ref(
        self,
        ref: str,
        *,
        name: str | None = None,
        priority: int | None = None,
        replace_options: bool | None = None,
        options_from: tuple[Any, ...] = (),
        options: tuple[Any, ...] = (),
        include_ids: tuple[str, ...] = (),
        exclude_ids: tuple[str, ...] = (),
    ) -> "Experiment":
        selector = PackSelector(
            ref=ref,
            name=name,
            priority=priority,
            replace_options=replace_options,
            options_from=options_from,
            options=options,
            include_ids=include_ids,
            exclude_ids=exclude_ids,
        )
        return self.add_pack_selector(selector)

    def _ensure_constraints(self) -> dict[str, Any]:
        constraints = dict(self._doc.get("constraints", {}))
        constraints.setdefault("include", [])
        constraints.setdefault("exclude", [])
        constraints.setdefault("predicates", [])
        self._doc["constraints"] = constraints
        return constraints

    def include(
        self, when: Mapping[str, str | list[str] | tuple[str, ...]]
    ) -> "Experiment":
        rule = ConstraintRule(
            mode="include",
            when={
                str(k): tuple(v if isinstance(v, (list, tuple)) else (v,))
                for k, v in when.items()
            },
        )
        constraints = self._ensure_constraints()
        rules = list(constraints.get("include", []))
        rules.append(rule.to_dict())
        constraints["include"] = rules
        self._doc["constraints"] = constraints
        return self

    def exclude(
        self, when: Mapping[str, str | list[str] | tuple[str, ...]]
    ) -> "Experiment":
        rule = ConstraintRule(
            mode="exclude",
            when={
                str(k): tuple(v if isinstance(v, (list, tuple)) else (v,))
                for k, v in when.items()
            },
        )
        constraints = self._ensure_constraints()
        rules = list(constraints.get("exclude", []))
        rules.append(rule.to_dict())
        constraints["exclude"] = rules
        self._doc["constraints"] = constraints
        return self

    def add_predicate(self, predicate: Predicate) -> "Experiment":
        if not isinstance(predicate, Predicate):
            raise ConfigError("add_predicate expects a Predicate")
        constraints = self._ensure_constraints()
        preds = list(constraints.get("predicates", []))
        preds.append(predicate.to_dict())
        constraints["predicates"] = preds
        self._doc["constraints"] = constraints
        return self

    def _ensure_definitions(self) -> dict[str, Any]:
        option_sets = dict(self._doc.get("option_sets", {}))
        packs = dict(self._doc.get("packs", {}))
        groups = dict(self._doc.get("groups", {}))
        self._doc["option_sets"] = option_sets
        self._doc["packs"] = packs
        self._doc["groups"] = groups
        return {"option_sets": option_sets, "packs": packs, "groups": groups}

    def add_option_set(self, name: str, options: list[Option]) -> "Experiment":
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("option_set name must be a non-empty string")
        if not isinstance(options, list) or not options:
            raise ConfigError("options must be a non-empty list")
        serialized: list[dict[str, Any]] = []
        for index, option in enumerate(options):
            if not isinstance(option, Option):
                raise ConfigError(f"options[{index}] must be Option")
            serialized.append(option.to_dict())
        definitions = self._ensure_definitions()
        option_sets = dict(definitions.get("option_sets", {}))
        option_sets[name.strip()] = serialized
        self._doc["option_sets"] = option_sets
        return self

    def add_pack_def(self, name: str, pack: Pack | Mapping[str, Any]) -> "Experiment":
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("pack definition name must be a non-empty string")
        if isinstance(pack, Pack):
            serialized = pack.to_dict()
        elif isinstance(pack, Mapping):
            serialized = deepcopy(dict(pack))
        else:
            raise ConfigError("pack must be Pack or mapping")

        definitions = self._ensure_definitions()
        packs = dict(definitions.get("packs", {}))
        packs[name.strip()] = serialized
        self._doc["packs"] = packs
        return self

    def add_group_def(
        self, name: str, selectors: list[PackSelector | Mapping[str, Any]]
    ) -> "Experiment":
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("group definition name must be a non-empty string")
        if not isinstance(selectors, list) or not selectors:
            raise ConfigError("selectors must be a non-empty list")
        serialized: list[dict[str, Any]] = []
        for idx, selector in enumerate(selectors):
            if isinstance(selector, PackSelector):
                serialized.append(selector.to_dict())
            elif isinstance(selector, Mapping):
                serialized.append(deepcopy(dict(selector)))
            else:
                raise ConfigError(f"selectors[{idx}] must be PackSelector or mapping")

        definitions = self._ensure_definitions()
        groups = dict(definitions.get("groups", {}))
        groups[name.strip()] = serialized
        self._doc["groups"] = groups
        return self

    def add_run_set(self, name: str, run_set: RunSetSpec) -> "Experiment":
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("run_set name must be a non-empty string")
        if not isinstance(run_set, RunSetSpec):
            raise ConfigError("run_set must be RunSetSpec")
        run_sets = dict(self._doc.get("run_sets", {}))
        run_sets[name.strip()] = run_set.to_dict()
        self._doc["run_sets"] = run_sets
        return self

    @staticmethod
    def _serialize_pack_entry(
        entry: Pack | PackSelector | Mapping[str, Any],
    ) -> dict[str, Any]:
        if isinstance(entry, Pack):
            return entry.to_dict()
        if isinstance(entry, PackSelector):
            return entry.to_dict()
        if isinstance(entry, Mapping):
            return deepcopy(dict(entry))
        raise ConfigError("pack entry must be Pack, PackSelector, or mapping")

    def run_set(
        self,
        name: str,
        *,
        extends: tuple[str, ...] = (),
        append: Mapping[str, Any] | None = None,
        replace: Mapping[str, Any] | None = None,
        patch_packs: Mapping[str, Pack | PackSelector | Mapping[str, Any]]
        | None = None,
    ) -> "Experiment":
        serialized_patch_packs: dict[str, dict[str, Any]] = {}
        if patch_packs is not None:
            for raw_name, replacement in patch_packs.items():
                pack_name = str(raw_name).strip()
                if not pack_name:
                    raise ConfigError("patch_packs keys must be non-empty strings")
                serialized_patch_packs[pack_name] = self._serialize_pack_entry(
                    replacement
                )
        return self.add_run_set(
            name,
            RunSetSpec(
                extends=extends,
                append=dict(append or {}),
                replace=dict(replace or {}),
                patch_packs=serialized_patch_packs,
            ),
        )

    def variant(
        self,
        name: str,
        *,
        extends: tuple[str, ...] = (),
        append_packs: list[Pack | PackSelector | Mapping[str, Any]] | None = None,
        replace_packs: list[Pack | PackSelector | Mapping[str, Any]] | None = None,
        patch_packs: Mapping[str, Pack | PackSelector | Mapping[str, Any]]
        | None = None,
        append_groups: list[str] | None = None,
        replace_groups: list[str] | None = None,
        append: Mapping[str, Any] | None = None,
        replace: Mapping[str, Any] | None = None,
    ) -> "Experiment":
        """Convenience helper for ablation run_sets.

        This method only builds canonical run_set payloads under:
        run_sets.<name>.{extends, append, replace}
        """

        append_payload = deepcopy(dict(append or {}))
        replace_payload = deepcopy(dict(replace or {}))

        if append_groups:
            select_map = dict(append_payload.get("select", {}))
            groups = list(select_map.get("groups", []))
            groups.extend(
                str(item).strip() for item in append_groups if str(item).strip()
            )
            select_map["groups"] = groups
            append_payload["select"] = select_map

        if append_packs:
            select_map = dict(append_payload.get("select", {}))
            packs = list(select_map.get("packs", []))
            packs.extend(self._serialize_pack_entry(entry) for entry in append_packs)
            select_map["packs"] = packs
            append_payload["select"] = select_map

        if replace_groups is not None:
            select_map = dict(replace_payload.get("select", {}))
            select_map["groups"] = [
                str(item).strip() for item in replace_groups if str(item).strip()
            ]
            replace_payload["select"] = select_map

        if replace_packs is not None:
            select_map = dict(replace_payload.get("select", {}))
            select_map["packs"] = [
                self._serialize_pack_entry(entry) for entry in replace_packs
            ]
            replace_payload["select"] = select_map

        serialized_patch_packs: dict[str, dict[str, Any]] = {}
        if patch_packs is not None:
            for raw_name, replacement in patch_packs.items():
                pack_name = str(raw_name).strip()
                if not pack_name:
                    raise ConfigError("patch_packs keys must be non-empty strings")
                serialized_patch_packs[pack_name] = self._serialize_pack_entry(
                    replacement
                )

        return self.add_run_set(
            name,
            RunSetSpec(
                extends=extends,
                append=append_payload,
                replace=replace_payload,
                patch_packs=serialized_patch_packs,
            ),
        )

    def to_dict(self, *, strict: bool = True) -> dict[str, Any]:
        doc = deepcopy(self._doc)
        if strict:
            doc = validate_document(doc)
        return doc

    def to_yaml(self, *, strict: bool = True, sort_keys: bool = False) -> str:
        return dump_yaml(self.to_dict(strict=strict), sort_keys=sort_keys)

    def _semantic_validate_via_geryon(
        self, experiment_path: Path, *, run_set: str | None = None
    ) -> dict[str, Any]:
        from geryon.planner import parse_experiment_yaml_with_diagnostics

        _, diagnostics = parse_experiment_yaml_with_diagnostics(
            experiment_path, run_set=run_set
        )
        return diagnostics

    def planned_config_count(self, *, run_set: str | None = None) -> int:
        from geryon.planner import plan_experiment

        yaml_text = self.to_yaml(strict=True)
        with TemporaryDirectory(prefix="geryon_dsl_count_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            experiment_path = tmp_path / "experiment.yaml"
            experiment_path.write_text(yaml_text, encoding="utf-8")
            summary = plan_experiment(
                experiment_path=experiment_path,
                out_dir=tmp_path / "out",
                batch_size=1024,
                run_set=run_set,
                dry_run=True,
            )
            return int(summary.total_configs)

    def to_yaml_file(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        sort_keys: bool = False,
        validate_with_geryon: bool = False,
        validate_run_set: str | None = None,
        max_configs: int | None = None,
        config_budget_run_set: str | None = None,
    ) -> Path:
        if max_configs is not None and max_configs <= 0:
            raise ConfigError("max_configs must be positive")

        data = self.to_dict(strict=strict)
        yaml_text = dump_yaml(data, sort_keys=sort_keys)

        if validate_with_geryon or max_configs is not None:
            with TemporaryDirectory(prefix="geryon_dsl_validate_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                tmp_experiment = tmp_path / "experiment.yaml"
                tmp_experiment.write_text(yaml_text, encoding="utf-8")

                if validate_with_geryon:
                    self._semantic_validate_via_geryon(
                        tmp_experiment, run_set=validate_run_set
                    )

                if max_configs is not None:
                    from geryon.planner import plan_experiment

                    summary = plan_experiment(
                        experiment_path=tmp_experiment,
                        out_dir=tmp_path / "out",
                        batch_size=1024,
                        run_set=config_budget_run_set,
                        dry_run=True,
                    )
                    if int(summary.total_configs) > int(max_configs):
                        raise ConfigError(
                            "DSL config budget exceeded: "
                            f"planned {summary.total_configs} configs (max_configs={max_configs})"
                        )

        return write_yaml(path, data, sort_keys=sort_keys)
