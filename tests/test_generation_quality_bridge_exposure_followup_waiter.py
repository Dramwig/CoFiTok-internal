from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from scripts import wait_for_generation_quality_bridge_exposure_followup as waiter


def _args(tmp_path: Path, *, max_polls: int) -> argparse.Namespace:
    script = Path(waiter.__file__).resolve()
    return argparse.Namespace(
        project=tmp_path / "project",
        quality_bridge_root=tmp_path / "quality",
        expected_revision="a" * 40,
        expected_tree="b" * 40,
        expected_branch="analysis/exposure-aware",
        expected_self_sha256=waiter._sha256(script),
        python=tmp_path / "python",
        status=tmp_path / "status.json",
        log=tmp_path / "waiter.log",
        poll_seconds=1,
        max_polls=max_polls,
    )


def _decision() -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "completed",
        "recommended_next_stage": {
            "id": "prepare_matched_250m_capacity_qualification_probe",
            "category": "capacity_qualification",
            "execution_ready": False,
        },
        "training_exposure": {
            "full_data_equivalent_epochs": 4.995,
            "insufficient_exposure_is_live_hypothesis": True,
            "terminal_result_content_bound": True,
            "terminal_checkpoint_binding_verified": True,
        },
        "authorization_boundary": {
            "recommended_stage_execution_allowed": False,
            "quality_bridge_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
            "release_authorization_allowed": False,
        },
    }


def test_postcondition_requires_fail_closed_exposure_aware_decision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_decision()), encoding="utf-8")
    result = waiter._postcondition(path)
    assert result["insufficient_exposure_is_live_hypothesis"] is True

    payload = _decision()
    payload["authorization_boundary"]["full_300k_launch_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="postcondition differs"):
        waiter._postcondition(path)


def test_waiter_bounded_wait_is_non_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, max_polls=1)
    monkeypatch.setattr(waiter, "parse_args", lambda: args)
    monkeypatch.setattr(
        waiter,
        "_validate_checkout",
        lambda unused: {
            "revision": args.expected_revision,
            "tree": args.expected_tree,
            "branch": args.expected_branch,
            "tracked_dirty": False,
            "runbook_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(waiter.time, "sleep", lambda seconds: None)

    assert waiter.main() == 78
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["status"] == "stopped"
    assert status["scope"]["gpu_use_allowed"] is False
    assert status["scope"]["process_signals_allowed"] is False


def test_waiter_builds_once_after_both_sources_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, max_polls=0)
    reports = args.quality_bridge_root / "reports"
    reports.mkdir(parents=True)
    (reports / "quality_bridge_result.json").write_text("{}", encoding="utf-8")
    exposure = args.quality_bridge_root / waiter.EXPOSURE_RELATIVE_PATH
    exposure.parent.mkdir(parents=True)
    exposure.write_text("{}", encoding="utf-8")
    args.project.mkdir()
    args.python.write_text("", encoding="utf-8")
    monkeypatch.setattr(waiter, "parse_args", lambda: args)
    monkeypatch.setattr(
        waiter,
        "_validate_checkout",
        lambda unused: {
            "revision": args.expected_revision,
            "tree": args.expected_tree,
            "branch": args.expected_branch,
            "tracked_dirty": False,
            "runbook_sha256": "c" * 64,
        },
    )

    def run_once(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        decision = reports / waiter.DECISION_NAME
        decision.write_text(json.dumps(_decision()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(waiter.subprocess, "run", run_once)

    assert waiter.main() == 0
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["recommended_next_stage"]["execution_ready"] is False
