from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cofitok.reporting import file_sha256
from scripts import run_generation_stability_full_readiness_waiter as waiter


TRAINING_REVISION = "1" * 40
TRAINING_BRANCH = "scale/generation-stability-50k-preflight"
EVALUATION_REVISION = "2" * 40
EVALUATION_BRANCH = "scale/generation-stability-50k-posteval-v4"
READINESS_REVISION = "3" * 40
READINESS_BRANCH = "scale/generation-large-capacity"


def _posteval(*, status: str = "pass") -> dict:
    return {
        "schema_version": 1,
        "role": "generation_stability_50k_posteval_waiter",
        "status": status,
        "detail": (
            "formal_ema_postevaluation_completed"
            if status == "pass"
            else "formal_ema_postevaluation_running"
        ),
        "child_exit_code": 0 if status == "pass" else None,
        "expected": {
            "training_revision": TRAINING_REVISION,
            "training_branch": TRAINING_BRANCH,
            "evaluation_revision": EVALUATION_REVISION,
            "evaluation_branch": EVALUATION_BRANCH,
            "formal_300k_allowed": False,
        },
        "updated_at": "2026-07-31T00:00:00+00:00",
    }


def test_posteval_status_requires_exact_completed_identity() -> None:
    observation = waiter.validate_posteval_status(
        _posteval(),
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision=EVALUATION_REVISION,
        expected_evaluation_branch=EVALUATION_BRANCH,
    )
    assert observation["complete"] is True

    replaced = _posteval()
    replaced["expected"]["evaluation_revision"] = "f" * 40
    with pytest.raises(ValueError, match="identity mismatch"):
        waiter.validate_posteval_status(
            replaced,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            expected_evaluation_revision=EVALUATION_REVISION,
            expected_evaluation_branch=EVALUATION_BRANCH,
        )


def test_scaling_gate_requires_exact_provenance_and_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "promotion_gate.json"
    path.write_text("{}", encoding="ascii")
    gate = {
        "provenance_contract": {
            "training_revision": TRAINING_REVISION,
            "training_branch": TRAINING_BRANCH,
            "evaluation_revision": EVALUATION_REVISION,
            "evaluation_branch": EVALUATION_BRANCH,
        }
    }
    monkeypatch.setattr(
        waiter,
        "validate_generation_gate_authorization",
        lambda *args, **kwargs: {"stage": "scaling"},
    )
    monkeypatch.setattr(
        waiter,
        "verify_generation_gate_source_reports",
        lambda *args, **kwargs: {
            "source_profile": "stability_scaling",
            "source_reports": {str(index): {} for index in range(6)},
        },
    )

    evidence = waiter.validate_scaling_gate(
        gate,
        gate_path=path,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision=EVALUATION_REVISION,
        expected_evaluation_branch=EVALUATION_BRANCH,
    )
    assert evidence["sha256"] == file_sha256(path)
    assert evidence["source_report_count"] == 6

    gate["provenance_contract"]["training_revision"] = "f" * 40
    with pytest.raises(ValueError, match="provenance contract mismatch"):
        waiter.validate_scaling_gate(
            gate,
            gate_path=path,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            expected_evaluation_revision=EVALUATION_REVISION,
            expected_evaluation_branch=EVALUATION_BRANCH,
        )


def test_deployment_must_bind_the_waiter_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "checkout"
    project.mkdir()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        waiter,
        "verify_deployment_receipt",
        lambda *args, **kwargs: {
            "checkout": {
                "path": project.resolve().as_posix(),
                "git": {
                    "revision": READINESS_REVISION,
                    "branch": READINESS_BRANCH,
                    "tracked_dirty": False,
                },
            },
            "readiness_execution_allowed": True,
            "readiness_executed": False,
            "full_training_launch_allowed": False,
        },
    )
    evidence = waiter.validate_deployment(
        project=project,
        receipt_path=receipt_path,
        expected_receipt_sha256="a" * 64,
        expected_revision=READINESS_REVISION,
        expected_branch=READINESS_BRANCH,
    )
    assert evidence["full_training_launch_allowed"] is False


def test_existing_readiness_replay_is_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "checkout"
    checkpoint_root = tmp_path / "generation"
    paths = waiter._full_paths(project, checkpoint_root)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="ascii")
        else:
            path.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(waiter.subprocess, "run", fake_run)
    evidence = waiter.validate_existing_readiness(
        project=project,
        checkpoint_root=checkpoint_root,
        promotion_gate=tmp_path / "gate.json",
        deployment_receipt=tmp_path / "receipt.json",
        expected_revision=READINESS_REVISION,
        expected_branch=READINESS_BRANCH,
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert "--allow-later-formal-repository" in command
    assert "--require-current-runtime-environment" in command
    assert command[command.index("--expected-revision") + 1] == READINESS_REVISION
    assert evidence["sha256"] == file_sha256(paths["readiness"])


def test_readiness_waiter_runbook_never_launches_training() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/"
        "generation_stability_ema_teacher_full_readiness_waiter.sh"
    ).read_text(encoding="utf-8")

    assert "run_generation_stability_full_readiness_waiter.py" in source
    assert "generation_stability_ema_teacher_full_readiness_after_gate.sh" in source
    assert "deployments/$EXPECTED_READINESS_REVISION/deployment_receipt.json" in source
    assert "EXPECTED_TRAINING_REVISION" in source
    assert "EXPECTED_EVALUATION_REVISION" in source
    assert "EXPECTED_READINESS_REVISION" in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source
    assert "train_generation.py" not in source

def test_waiter_main_launches_only_readiness_after_pass_gate_and_idle_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "checkout"
    project.mkdir()
    checkpoint_root = tmp_path / "generation"
    promotion_gate = tmp_path / "promotion_gate.json"
    posteval_status = tmp_path / "posteval_waiter.json"
    deployment_receipt = tmp_path / "deployment_receipt.json"
    readiness_runbook = tmp_path / "readiness.sh"
    status_output = tmp_path / "readiness_waiter.json"
    for path in (
        promotion_gate,
        posteval_status,
        deployment_receipt,
        readiness_runbook,
    ):
        path.write_text("{}", encoding="ascii")
    args = SimpleNamespace(
        project=project,
        promotion_gate=promotion_gate,
        posteval_status=posteval_status,
        deployment_receipt=deployment_receipt,
        expected_deployment_receipt_sha256="a" * 64,
        status_output=status_output,
        readiness_runbook=readiness_runbook,
        checkpoint_root=checkpoint_root,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision=EVALUATION_REVISION,
        expected_evaluation_branch=EVALUATION_BRANCH,
        expected_readiness_revision=READINESS_REVISION,
        expected_readiness_branch=READINESS_BRANCH,
        timeout_seconds=60.0,
        poll_seconds=1.0,
    )
    monkeypatch.setattr(waiter, "_parse_args", lambda: args)
    monkeypatch.setattr(waiter, "verify_readiness_checkout", lambda *a, **k: {})
    monkeypatch.setattr(
        waiter,
        "validate_deployment",
        lambda **kwargs: {
            "receipt_sha256": "a" * 64,
            "readiness_execution_allowed": True,
            "full_training_launch_allowed": False,
        },
    )
    monkeypatch.setattr(waiter, "_read_object", lambda path: {})
    monkeypatch.setattr(
        waiter,
        "validate_posteval_status",
        lambda *a, **k: {"status": "pass", "complete": True},
    )
    monkeypatch.setattr(
        waiter,
        "validate_scaling_gate",
        lambda *a, **k: {"status": "verified", "sha256": "b" * 64},
    )
    monkeypatch.setattr(waiter, "_gpu_compute_pids", lambda: [])
    monkeypatch.setattr(
        waiter,
        "validate_existing_readiness",
        lambda **kwargs: {
            "status": "verified",
            "path": "full_training_readiness.json",
            "sha256": "c" * 64,
        },
    )
    captured: dict[str, object] = {}

    class Child:
        pid = 901

        @staticmethod
        def wait() -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> Child:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Child()

    monkeypatch.setattr(waiter.subprocess, "Popen", fake_popen)

    assert waiter.main() == 0
    assert captured["command"] == ["bash", str(readiness_runbook.resolve())]
    environment = captured["kwargs"]["env"]
    assert isinstance(environment, dict)
    assert environment["EXPECTED_SCALING_GATE_SHA256"] == "b" * 64
    assert "EXPECTED_FULL_LAUNCH_RECEIPT_SHA256" not in environment
    status = json.loads(status_output.read_text(encoding="utf-8"))
    assert status["status"] == "pass"
    assert status["detail"] == "stability_full_readiness_completed"
    assert status["full_training_launch_allowed"] is False
