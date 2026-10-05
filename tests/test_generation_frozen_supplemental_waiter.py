from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_generation_stability_frozen_supplemental_waiter as waiter


TRAINING_REVISION = "1" * 40
TRAINING_BRANCH = "scale/generation-stability-50k-preflight"
FROZEN_REVISION = "2" * 40
FROZEN_BRANCH = "scale/generation-stability-50k-posteval-v4"
READINESS_REVISION = "3" * 40
READINESS_BRANCH = "scale/generation-large-capacity"
SUPPLEMENTAL_REVISION = "4" * 40
SUPPLEMENTAL_BRANCH = "scale/generation-large-capacity"


def _posteval(*, status: str = "pass") -> dict:
    return {
        "schema_version": 1,
        "role": "generation_stability_50k_posteval_waiter",
        "status": status,
        "detail": (
            "formal_ema_postevaluation_completed"
            if status == "pass"
            else "waiting_for_completed_training"
        ),
        "child_exit_code": 0 if status == "pass" else None,
        "expected": {
            "training_revision": TRAINING_REVISION,
            "training_branch": TRAINING_BRANCH,
            "evaluation_revision": FROZEN_REVISION,
            "evaluation_branch": FROZEN_BRANCH,
            "formal_300k_allowed": False,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _readiness(*, status: str = "pass") -> dict:
    return {
        "schema_version": 1,
        "role": "generation_stability_full_readiness_waiter",
        "status": status,
        "detail": f"readiness_{status}",
        "child_pid": 900 if status == "running" else None,
        "child_exit_code": 0 if status == "pass" else None,
        "expected": {
            "training_revision": TRAINING_REVISION,
            "training_branch": TRAINING_BRANCH,
            "evaluation_revision": FROZEN_REVISION,
            "evaluation_branch": FROZEN_BRANCH,
            "readiness_revision": READINESS_REVISION,
            "readiness_branch": READINESS_BRANCH,
            "full_training_launch_allowed": False,
        },
        "full_training_launch_allowed": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _supplemental(*, status: str = "hold") -> dict:
    passed = status == "pass"
    return {
        "schema_version": 1,
        "role": "generation_stability_frozen_supplemental_qualification",
        "status": status,
        "decision": "supplemental_quality_complete" if passed else "hold",
        "sources": {},
        "builder_git": {
            "revision": SUPPLEMENTAL_REVISION,
            "branch": SUPPLEMENTAL_BRANCH,
            "tracked_dirty": False,
        },
        "provenance_contract": {
            "training_revision": TRAINING_REVISION,
            "training_branch": TRAINING_BRANCH,
            "evaluation_revision": FROZEN_REVISION,
            "evaluation_branch": FROZEN_BRANCH,
            "supplemental_revision": SUPPLEMENTAL_REVISION,
            "supplemental_branch": SUPPLEMENTAL_BRANCH,
        },
        "checks": {
            "base_gate_passed": True,
            "distribution_support_passed": passed,
            "ema_rollout_stability_passed": passed,
            "all_supplemental_quality_checks_passed": passed,
        },
        "claim_boundary": {
            "supplemental_non_authorizing": True,
            "replaces_generation_gate": False,
            "replaces_readiness": False,
            "scaling_authorization_evaluated": False,
            "full_training_launch_allowed": False,
        },
    }


def _args(tmp_path: Path) -> SimpleNamespace:
    project = tmp_path / "checkout"
    project.mkdir()
    runbook = project / "supplemental.sh"
    runbook.write_text("#!/usr/bin/env bash\n", encoding="ascii")
    posteval = tmp_path / "posteval.json"
    posteval.write_text(json.dumps(_posteval()), encoding="utf-8")
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps(_readiness()), encoding="utf-8")
    return SimpleNamespace(
        project=project,
        posteval_status=posteval,
        readiness_status=readiness,
        supplemental_report=tmp_path / "supplemental.json",
        status_output=tmp_path / "supplemental_waiter.json",
        supplemental_runbook=runbook,
        checkpoint_root=tmp_path / "generation",
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_frozen_evaluation_revision=FROZEN_REVISION,
        expected_frozen_evaluation_branch=FROZEN_BRANCH,
        expected_readiness_revision=READINESS_REVISION,
        expected_readiness_branch=READINESS_BRANCH,
        expected_supplemental_revision=SUPPLEMENTAL_REVISION,
        expected_supplemental_branch=SUPPLEMENTAL_BRANCH,
        timeout_seconds=60.0,
        poll_seconds=1.0,
        status_silence_seconds=900.0,
    )


def test_waiter_requires_fresh_source_bound_postevaluation() -> None:
    observation = waiter.observe_posteval_status(
        _posteval(),
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision=FROZEN_REVISION,
        expected_evaluation_branch=FROZEN_BRANCH,
        silence_seconds=900.0,
    )
    assert observation["complete"] is True

    running = _posteval(status="waiting")
    now = datetime.now(timezone.utc)
    running["updated_at"] = (now - timedelta(seconds=901)).isoformat()
    with pytest.raises(RuntimeError, match="post-evaluation status is stale"):
        waiter.observe_posteval_status(
            running,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            expected_evaluation_revision=FROZEN_REVISION,
            expected_evaluation_branch=FROZEN_BRANCH,
            silence_seconds=900.0,
            now=now,
        )


@pytest.mark.parametrize("status", ["pass", "failed"])
def test_readiness_coordination_must_finish_before_supplemental(status: str) -> None:
    observation = waiter.observe_readiness_status(
        _readiness(status=status),
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision=FROZEN_REVISION,
        expected_evaluation_branch=FROZEN_BRANCH,
        expected_readiness_revision=READINESS_REVISION,
        expected_readiness_branch=READINESS_BRANCH,
        silence_seconds=900.0,
    )
    assert observation["terminal"] is True
    assert observation["full_training_launch_allowed"] is False


def test_readiness_coordination_rejects_authorization_or_identity_drift() -> None:
    payload = _readiness()
    payload["full_training_launch_allowed"] = True
    with pytest.raises(ValueError, match="coordination contract mismatch"):
        waiter.observe_readiness_status(
            payload,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            expected_evaluation_revision=FROZEN_REVISION,
            expected_evaluation_branch=FROZEN_BRANCH,
            expected_readiness_revision=READINESS_REVISION,
            expected_readiness_branch=READINESS_BRANCH,
            silence_seconds=900.0,
        )


def test_terminal_supplemental_hold_is_execution_complete_but_non_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "supplemental.json"
    path.write_text(json.dumps(_supplemental()), encoding="utf-8")
    monkeypatch.setattr(
        waiter,
        "verify_frozen_supplemental_report",
        lambda *args, **kwargs: pytest.fail("a scientific hold is not a pass replay"),
    )
    result = waiter.validate_supplemental_result(
        _supplemental(),
        report_path=path,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_frozen_evaluation_revision=FROZEN_REVISION,
        expected_frozen_evaluation_branch=FROZEN_BRANCH,
        expected_supplemental_revision=SUPPLEMENTAL_REVISION,
        expected_supplemental_branch=SUPPLEMENTAL_BRANCH,
    )
    assert result["status"] == "hold"
    assert result["pass_replay"] is None
    assert result["full_training_launch_allowed"] is False

    incoherent = _supplemental()
    incoherent["checks"]["all_supplemental_quality_checks_passed"] = True
    path.write_text(json.dumps(incoherent), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal result contract mismatch"):
        waiter.validate_supplemental_result(
            incoherent,
            report_path=path,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            expected_frozen_evaluation_revision=FROZEN_REVISION,
            expected_frozen_evaluation_branch=FROZEN_BRANCH,
            expected_supplemental_revision=SUPPLEMENTAL_REVISION,
            expected_supplemental_branch=SUPPLEMENTAL_BRANCH,
        )


def test_waiter_launches_only_supplemental_after_readiness_terminal_and_idle_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        waiter,
        "verify_supplemental_checkout",
        lambda *args, **kwargs: {
            "revision": SUPPLEMENTAL_REVISION,
            "branch": SUPPLEMENTAL_BRANCH,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(waiter, "_gpu_compute_pids", lambda: [])
    captured: dict[str, object] = {}

    class Child:
        pid = 902

        def wait(self, *, timeout: float) -> int:
            assert timeout == 1.0
            args.supplemental_report.write_text(
                json.dumps(_supplemental()), encoding="utf-8"
            )
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> Child:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Child()

    monkeypatch.setattr(waiter.subprocess, "Popen", fake_popen)
    assert waiter.run_waiter(args) == 0

    assert captured["command"] == [
        "bash",
        str(args.supplemental_runbook.resolve()),
    ]
    environment = captured["kwargs"]["env"]
    assert isinstance(environment, dict)
    assert environment["EXPECTED_SUPPLEMENTAL_REVISION"] == SUPPLEMENTAL_REVISION
    assert "EXPECTED_FULL_LAUNCH_RECEIPT_SHA256" not in environment
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "pass"
    assert status["detail"] == "frozen_supplemental_execution_completed"
    assert status["supplemental"]["status"] == "hold"
    assert status["supplemental_non_authorizing"] is True
    assert status["full_training_launch_allowed"] is False


def test_waiter_does_not_race_the_queued_readiness_gpu_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.readiness_status.write_text(
        json.dumps(_readiness(status="running")), encoding="utf-8"
    )
    monkeypatch.setattr(
        waiter,
        "verify_supplemental_checkout",
        lambda *args, **kwargs: {
            "revision": SUPPLEMENTAL_REVISION,
            "branch": SUPPLEMENTAL_BRANCH,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(waiter, "_gpu_compute_pids", lambda: [])
    sleeps = 0
    launch_observation: dict[str, object] = {}

    def fake_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        assert seconds == 1.0
        waiting = json.loads(args.status_output.read_text(encoding="utf-8"))
        assert waiting["detail"] == "waiting_for_readiness_gpu_stage_to_finish"
        args.readiness_status.write_text(
            json.dumps(_readiness(status="pass")), encoding="utf-8"
        )

    class Child:
        pid = 903

        def wait(self, *, timeout: float) -> int:
            assert timeout == 1.0
            args.supplemental_report.write_text(
                json.dumps(_supplemental()), encoding="utf-8"
            )
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> Child:
        launch_observation["readiness"] = json.loads(
            args.readiness_status.read_text(encoding="utf-8")
        )["status"]
        return Child()

    monkeypatch.setattr(waiter.time, "sleep", fake_sleep)
    monkeypatch.setattr(waiter.subprocess, "Popen", fake_popen)

    assert waiter.run_waiter(args) == 0
    assert sleeps == 1
    assert launch_observation["readiness"] == "pass"


def test_waiter_runbook_is_quality_only_and_readiness_serialized() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/generation_stability_frozen_50k_supplemental_waiter.sh"
    ).read_text(encoding="utf-8")

    assert "run_generation_stability_frozen_supplemental_waiter.py" in source
    assert "generation_stability_frozen_50k_supplemental_after_posteval.sh" in source
    assert "--readiness-status" in source
    assert "EXPECTED_READINESS_REVISION" in source
    assert "EXPECTED_SUPPLEMENTAL_REVISION" in source
    assert "train_generation.py" not in source
    assert "full_matched_300k" not in source
    assert "full_training_launch_allowed=true" not in source
