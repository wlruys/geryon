from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples" / "merge_policies"


def _run_example(example_name: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    example_dir = EXAMPLES_ROOT / example_name
    launch_script = example_dir / "launch.sh"

    env = os.environ.copy()
    env["GERYON_EXAMPLE_OUT_DIR"] = str(tmp_path / "outputs" / example_name)
    env["GERYON_EXAMPLE_RUN_ID"] = f"pytest_{example_name}"
    env["PYTHON_BIN"] = sys.executable

    return subprocess.run(
        ["bash", str(launch_script)],
        cwd=str(example_dir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(dict(json.loads(stripped)))
    return records


def _run_root(example_name: str, tmp_path: Path) -> Path:
    out_dir = tmp_path / "outputs" / example_name
    run_id = f"pytest_{example_name}"
    return out_dir / "runs" / run_id


@pytest.mark.parametrize(
    ("example_name", "expect_plan", "expected_failure_text"),
    [
        ("01_conflict_behavior", True, None),
        ("02_key_strategies", True, None),
        ("03_intentional_overrides", True, None),
        ("04_duplicate_pack_extension", True, None),
        ("05_delete_sentinel", True, None),
        ("06_policy_gates", True, None),
    ],
)
def test_merge_policy_examples(
    example_name: str,
    expect_plan: bool,
    expected_failure_text: str | None,
    tmp_path: Path,
) -> None:
    proc = _run_example(example_name, tmp_path)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    run_root = _run_root(example_name, tmp_path)
    if not expect_plan:
        combined_output = f"{proc.stdout}\n{proc.stderr}"
        assert expected_failure_text is not None
        assert expected_failure_text in combined_output
        assert not (run_root / "plan").exists()
        return

    assert run_root.exists(), f"missing run root: {run_root}"
    assert (run_root / "plan" / "configs.jsonl").exists()
    assert (run_root / "plan" / "batches.jsonl").exists()

    planned_configs = _read_jsonl(run_root / "plan" / "configs.jsonl")
    planned_batches = _read_jsonl(run_root / "plan" / "batches.jsonl")
    assert len(planned_configs) >= 1
    assert len(planned_batches) >= 1

    result_files = sorted((run_root / "exec" / "results").glob("task_local_*.jsonl"))
    assert result_files, f"missing result files under {run_root / 'exec' / 'results'}"

    result_records: list[dict[str, object]] = []
    for result_file in result_files:
        result_records.extend(_read_jsonl(result_file))

    success_records = [
        record for record in result_records if record.get("status") == "success"
    ]
    assert success_records, "expected at least one successful local execution"

    for record in success_records:
        stdout_path = Path(str(record.get("stdout_path", "")))
        assert stdout_path.exists(), f"missing command stdout log: {stdout_path}"
        stdout_text = stdout_path.read_text(encoding="utf-8")
        assert "RESOLVED_CONFIG_JSON=" in stdout_text
