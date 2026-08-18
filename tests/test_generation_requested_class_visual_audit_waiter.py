from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RESULT_ROLE,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.inference_replay import file_identity
from scripts import wait_for_generation_requested_class_visual_audit as waiter
from scripts.build_generation_requested_class_visual_audit import (
    CLAIM_BOUNDARY as VISUAL_AUDIT_CLAIM_BOUNDARY,
    REPORT_FILENAME as VISUAL_AUDIT_REPORT_FILENAME,
    REPORT_ROLE as VISUAL_AUDIT_REPORT_ROLE,
)
from scripts.wait_for_generation_requested_class_visual_audit import (
    AUTHORIZATION_BOUNDARY,
    FIXED_INDICES,
    build_visual_audit_command,
    validate_quality_bridge_result,
    validate_visual_audit_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_quality_result_requires_exact_nonauthorizing_terminal_contract(
    tmp_path: Path,
) -> None:
    result = tmp_path / "quality_bridge_result.json"
    payload = {
        "schema_version": 1,
        "status": "completed",
        "role": QUALITY_BRIDGE_RESULT_ROLE,
        "authorization_boundary": RESULT_AUTHORIZATION_BOUNDARY,
        "quality_screen": {"status": "hold"},
    }
    _write_json(result, payload)
    validated = validate_quality_bridge_result(result)
    assert validated["identity"] == file_identity(result)
    assert validated["quality_screen"] == {"status": "hold"}

    widened = copy.deepcopy(payload)
    widened["authorization_boundary"]["full_300k_launch_allowed"] = True
    _write_json(result, widened)
    with pytest.raises(ValueError, match="contract differs"):
        validate_quality_bridge_result(result)


def test_visual_audit_command_is_fixed_cpu_diagnostic_selection(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    args = argparse.Namespace(
        python=str(python),
        project=str(project),
        cofitok_sampling_report=str(tmp_path / "cofitok.json"),
        dense_sampling_report=str(tmp_path / "dense.json"),
        cofitok_dir=str(tmp_path / "cofitok"),
        dense_dir=str(tmp_path / "dense"),
        classifier_calibration_report=str(tmp_path / "calibration.json"),
        real_dir=str(tmp_path / "real"),
        output_dir=str(tmp_path / "output"),
        expected_revision="a" * 40,
        expected_branch="visual-audit",
    )
    command = build_visual_audit_command(args)

    assert Path(command[0]) == python.resolve()
    assert command[command.index("--indices") + 1] == ",".join(
        str(index) for index in FIXED_INDICES
    )
    assert command[command.index("--panel-columns") + 1] == "8"
    assert command[-1] == "--resume"
    assert "train_generation.py" not in " ".join(command)
    assert AUTHORIZATION_BOUNDARY["gpu_use_allowed"] is False
    assert AUTHORIZATION_BOUNDARY["training_launch_allowed"] is False


def test_visual_report_revalidates_fixed_indices_and_panel_identities(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    panels = []
    for panel_index, indices in enumerate((range(0, 8), range(8, 16))):
        panel_path = output / f"requested_class_panel_{panel_index:02d}.png"
        panel_path.write_bytes(f"panel-{panel_index}".encode("ascii"))
        panels.append(
            {
                **file_identity(panel_path),
                "indices": list(indices),
                "row_order": ["real_validation", "cofitok", "dense_identity"],
                "columns": 8,
                "grid_shape": [3, 1, 1],
            }
        )
    report_path = output / VISUAL_AUDIT_REPORT_FILENAME
    _write_json(
        report_path,
        {
            "schema_version": 1,
            "status": "completed",
            "role": VISUAL_AUDIT_REPORT_ROLE,
            "claim_boundary": VISUAL_AUDIT_CLAIM_BOUNDARY,
            "indices": list(FIXED_INDICES),
            "panels": panels,
        },
    )

    assert validate_visual_audit_report(output) == file_identity(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["indices"][-1] = 99
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="report differs"):
        validate_visual_audit_report(output)


def test_waiter_lock_fails_closed_without_posix_fcntl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(waiter, "fcntl", None)
    with (tmp_path / "waiter.lock").open("a+", encoding="utf-8") as lock_handle:
        with pytest.raises(RuntimeError, match="fcntl locking is unavailable"):
            waiter.acquire_exclusive_waiter_lock(lock_handle)
