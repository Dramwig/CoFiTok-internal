from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cofitok.generation_gate_sources import (
    GATE_DIAGNOSTIC_SUFFIXES,
    GATE_SOURCE_PROFILE_STAGES,
    GATE_SOURCE_SUFFIXES,
)
from scripts import build_large_scale_generation_comparison as comparison
from scripts import audit_generation_capacity_full_completion as completion
from scripts import run_generation_capacity_full_300k_posteval_supervisor as posteval
from scripts import run_generation_capacity_full_300k_finalization_supervisor as finalizer
from cofitok.generation.release import _COMPLETION_PROFILES


ROOT = Path(__file__).resolve().parents[1]
POSTEVAL_RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_capacity_full_300k_posteval_50k.sh"
)
SUPERVISOR_RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_capacity_full_300k_posteval_supervisor.sh"
)
BASE_RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_stability_ema_teacher_full_posteval_50k.sh"
)
SUPERVISOR = ROOT / "scripts/run_generation_capacity_full_300k_posteval_supervisor.py"
EXPORT_RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_capacity_full_300k_export_inference_artifacts.sh"
)
COMPLETION_RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_capacity_full_300k_completion_audit.sh"
)
FINALIZE_RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_capacity_full_300k_finalize_after_gate.sh"
)
FINALIZATION_SUPERVISOR_RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_capacity_full_300k_finalization_supervisor.sh"
)
FINALIZATION_SUPERVISOR = (
    ROOT / "scripts/run_generation_capacity_full_300k_finalization_supervisor.py"
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_capacity_full_profiles_bind_the_new_output_lineage() -> None:
    assert GATE_SOURCE_PROFILE_STAGES["capacity_full"] == "full"
    assert set(GATE_SOURCE_SUFFIXES["capacity_full"]) == {
        "cofitok_training",
        "dense_training",
        "cofitok_generation",
        "dense_generation",
        "cofitok_checkpoint_eval",
        "dense_checkpoint_eval",
    }
    assert all(
        "stability_capacity_full_300k_v1" in suffix
        for suffix in GATE_SOURCE_SUFFIXES["capacity_full"].values()
    )
    assert set(GATE_DIAGNOSTIC_SUFFIXES["capacity_full"]) == {
        "rollout_stability_qualification",
        "cofitok_class_fidelity",
        "dense_class_fidelity",
        "class_fidelity_qualification",
    }
    assert "capacity_full" in comparison.CLASS_FIDELITY_SOURCE_PROFILES
    assert all(
        "stability_capacity_full_300k_v1" in suffix
        for suffix in comparison.SOURCE_REPORT_PROFILES["capacity_full"].values()
        if "official_related" not in suffix
    )


def test_capacity_full_posteval_reuses_protocol_but_not_failed_scaling_gate() -> None:
    wrapper = POSTEVAL_RUNBOOK.read_text(encoding="utf-8")
    source = BASE_RUNBOOK.read_text(encoding="utf-8")
    assert "stability_capacity_full_300k_v1" in wrapper
    assert "capacity_full_experimental" in wrapper
    assert "authorize_fresh_matched_300k_training" in wrapper
    assert "capacity_full_300k_training_launch_receipt.json" in wrapper
    assert "EXPECTED_SCALING_GATE_SHA256" not in wrapper
    assert "promotion_gate.json" not in wrapper
    assert "SOURCE_PROFILE_ARGS=(--source-profile capacity_full)" in source
    assert '--authorization-gate "$TRAINING_AUTHORIZATION"' in source
    assert "scripts/verify_generation_training_authorization.py" in source
    assert source.count("--num-samples 50000") == 2
    assert source.count("--sample-steps 250") == 5
    assert "--max-absolute-fid 20.0" in source
    assert "--min-precision 0.30" in source
    assert "--min-recall 0.30" in source


def test_capacity_full_posteval_supervisor_is_fail_closed_and_non_signaling() -> None:
    wrapper = SUPERVISOR_RUNBOOK.read_text(encoding="utf-8")
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "flock -n" in wrapper
    assert "nice -n 19" in wrapper
    assert "run_generation_capacity_full_300k_posteval_supervisor.py" in wrapper
    assert "required_idle_polls" in source
    assert "start_new_session=True" in source
    assert "os.kill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert "pkill" not in source
    assert posteval.SUPERVISOR_BOUNDARY["training_launch_allowed"] is False
    assert posteval.SUPERVISOR_BOUNDARY[
        "posteval_50k_allowed_after_exact_training_pass"
    ] is True
    assert posteval.SUPERVISOR_BOUNDARY["final_quality_gate_build_allowed"] is True
    assert posteval.SUPERVISOR_BOUNDARY["inference_export_allowed"] is False
    assert posteval.SUPERVISOR_BOUNDARY["release_receipt_allowed"] is False
    assert posteval.SUPERVISOR_BOUNDARY[
        "formal_generation_completion_claim_allowed"
    ] is False


@pytest.mark.parametrize(
    ("passed", "expected_status", "expected_decision"),
    ((True, "pass", "large_scale_generation_ready"), (False, "hold", "hold")),
)
def test_capacity_full_posteval_result_distinguishes_pass_from_hold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    passed: bool,
    expected_status: str,
    expected_decision: str,
) -> None:
    training_revision = "1" * 40
    evaluation_revision = "2" * 40
    training_branch = "scale/training"
    evaluation_branch = "scale/evaluation"
    gate_status = "pass" if passed else "fail"
    gate_decision = "large_scale_generation_ready" if passed else "hold"
    gate = {
        "stage": "full",
        "source_profile": "capacity_full",
        "status": gate_status,
        "decision": gate_decision,
        "provenance_contract": {
            "training_revision": training_revision,
            "training_branch": training_branch,
            "evaluation_revision": evaluation_revision,
            "evaluation_branch": evaluation_branch,
        },
        "gates": [
            {"name": "quality", "passed": passed},
            {"name": "provenance", "passed": True},
        ],
    }
    comparison_report = {
        "source_profile": "capacity_full",
        "status": "ready" if passed else "hold",
        "final_gate": {"status": gate_status, "decision": gate_decision},
    }
    gate_path = tmp_path / "final_generation_gate.json"
    comparison_path = tmp_path / "comparison.json"
    _write(gate_path, gate)
    _write(comparison_path, comparison_report)
    monkeypatch.setattr(
        posteval,
        "verify_generation_gate_source_reports",
        lambda value: {
            "source_profile": "capacity_full",
            "source_reports": {"source": {}},
            "diagnostic_reports": {"diagnostic": {}},
        },
    )
    monkeypatch.setattr(
        posteval,
        "verify_comparison_source_reports",
        lambda value: {"source_profile": "capacity_full"},
    )
    monkeypatch.setattr(
        posteval,
        "validate_generation_gate_authorization",
        lambda value, expected_stage: {"stage": expected_stage},
    )
    result = posteval.validate_posteval_result(
        final_gate_path=gate_path,
        comparison_path=comparison_path,
        expected_training_revision=training_revision,
        expected_training_branch=training_branch,
        expected_evaluation_revision=evaluation_revision,
        expected_evaluation_branch=evaluation_branch,
    )
    assert result["status"] == expected_status
    assert result["decision"] == expected_decision
    assert (result["authorization"] is not None) is passed


def test_capacity_full_training_completion_replays_receipt_and_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "3" * 40
    tree = "4" * 40
    branch = "scale/training"
    full_root = tmp_path / "stability_capacity_full_300k_v1"
    receipt = full_root / "reports/capacity_full_300k_training_launch_receipt.json"
    status_path = tmp_path / "training_supervisor_status.json"
    cofitok_path = full_root / "cofitok/training_report.json"
    dense_path = full_root / "dense_identity/training_report.json"
    monitor_path = full_root / "pair_monitor.json"
    _write(receipt, {"role": "receipt"})
    _write(cofitok_path, {"method": "cofitok"})
    _write(dense_path, {"method": "dense"})
    _write(
        monitor_path,
        {
            "schema_version": 2,
            "monitor": "generation_capacity_full_300k_v1",
            "status": "pass",
            "stage": "complete",
            "issues": [],
            "git": {"revision": revision, "branch": branch, "tracked_dirty": False},
        },
    )
    receipt_identity = posteval.file_identity(receipt)
    status = {
        "schema_version": 1,
        "role": posteval.TRAINING_SUPERVISOR_ROLE,
        "status": "pass",
        "detail": "capacity_full_fresh_matched_300k_training_completed",
        "training_launch_performed": True,
        "formal_generation_completion_claimed": False,
        "child_exit_code": 0,
        "authorization_boundary": posteval.TRAINING_SUPERVISOR_BOUNDARY,
        "expected": {
            "training_git": {
                "revision": revision,
                "tree": tree,
                "branch": branch,
                "tracked_dirty": False,
            },
            "full_output_root": full_root.resolve().as_posix(),
        },
        "training_launch_receipt": receipt_identity,
    }
    _write(status_path, status)
    monkeypatch.setattr(
        posteval,
        "capture_generation_training_authorization",
        lambda path: {
            "stage": posteval.TRAINING_AUTHORIZATION_STAGE,
            "decision": posteval.TRAINING_AUTHORIZATION_DECISION,
            "gate_sha256": receipt_identity["sha256"],
        },
    )
    monkeypatch.setattr(
        posteval,
        "validate_training_pair",
        lambda *args, **kwargs: {
            "status": "pass",
            "expected_steps": 300_000,
            "expected_revision": revision,
            "authorization_gate_identity_sha256": "5" * 64,
        },
    )
    evidence = posteval.validate_training_completion(
        status,
        status_path=status_path,
        training_receipt_path=receipt,
        cofitok_training_path=cofitok_path,
        dense_training_path=dense_path,
        pair_monitor_path=monitor_path,
        expected_training_revision=revision,
        expected_training_branch=branch,
        expected_training_tree=tree,
        full_output_root=full_root,
    )
    assert evidence["training_launch_receipt"] == receipt_identity
    assert evidence["training_pair"]["status"] == "pass"


def test_capacity_full_export_and_completion_are_final_gate_bound() -> None:
    export = EXPORT_RUNBOOK.read_text(encoding="utf-8")
    audit = COMPLETION_RUNBOOK.read_text(encoding="utf-8")
    base_export = (
        ROOT
        / "artifacts/runbooks/"
        "generation_stability_ema_teacher_export_inference_artifacts.sh"
    ).read_text(encoding="utf-8")
    assert "stability_capacity_full_300k_v1" in export
    assert "generation_stability_ema_teacher_export_inference_artifacts.sh" in export
    assert "scripts/validate_generation_gate_report.py" in base_export
    assert "--stage full" in base_export
    assert "scripts/audit_generation_capacity_full_completion.py" in audit
    assert audit.index("audit_generation_capacity_full_completion.py") < audit.index(
        "build_generation_release_receipt.py"
    )
    assert "EXPECTED_FINAL_GATE_SHA256=${" in audit
    assert "EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256=${" in audit
    assert "scripts/train_generation.py" not in audit
    assert "scripts/generate_samples.py" not in audit
    assert _COMPLETION_PROFILES[completion.PROFILE] == {
        "status": "pass",
        "check": completion.RELEASE_CHECK,
    }


def test_capacity_full_finalization_supervisor_is_gate_bound_and_non_signaling() -> None:
    wrapper = FINALIZATION_SUPERVISOR_RUNBOOK.read_text(encoding="utf-8")
    source = FINALIZATION_SUPERVISOR.read_text(encoding="utf-8")
    finalize = FINALIZE_RUNBOOK.read_text(encoding="utf-8")
    assert "flock -n" in wrapper
    assert "nice -n 19" in wrapper
    assert "run_generation_capacity_full_300k_finalization_supervisor.py" in wrapper
    assert finalizer.finalization_action({"status": "pass"}) == "finalize"
    assert finalizer.finalization_action({"status": "hold"}) == "hold"
    with pytest.raises(ValueError, match="not terminal"):
        finalizer.finalization_action({"status": "waiting"})
    assert "capacity_full_final_gate_held_no_export" in source
    assert "os.kill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert "pkill" not in source
    assert finalize.index(
        "generation_capacity_full_300k_export_inference_artifacts.sh"
    ) < finalize.index("generation_capacity_full_300k_completion_audit.sh")
    assert finalizer.SUPERVISOR_BOUNDARY["training_launch_allowed"] is False
    assert finalizer.SUPERVISOR_BOUNDARY["posteval_launch_allowed"] is False
    assert finalizer.SUPERVISOR_BOUNDARY[
        "inference_export_allowed_after_passed_full_gate"
    ] is True
    assert finalizer.SUPERVISOR_BOUNDARY["process_signaling_allowed"] is False


def test_capacity_full_finalization_requires_passed_audit_and_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = {
        "training_revision": "1" * 40,
        "training_tree": "2" * 40,
        "training_branch": "train",
        "evaluation_revision": "3" * 40,
        "evaluation_tree": "4" * 40,
        "evaluation_branch": "eval",
        "export_revision": "3" * 40,
        "export_branch": "eval",
        "training_supervisor_deployment_sha256": "a" * 64,
        "posteval_supervisor_deployment_sha256": "b" * 64,
        "training_launch_receipt_sha256": "c" * 64,
        "final_gate_sha256": "d" * 64,
    }
    audit_path = tmp_path / "completion.json"
    receipt_path = tmp_path / "release.json"
    cofitok_artifact = tmp_path / "cofitok.pt"
    dense_artifact = tmp_path / "dense.pt"
    _write(
        audit_path,
        {
            "schema_version": 1,
            "profile": completion.PROFILE,
            "status": "pass",
            "complete": True,
            "failed_checks": [],
            "missing_checks": [],
            "expectations": expected,
            "claim_boundary": {
                "training_authorization_is_quality_promotion_gate": False,
                "final_gate_required_for_release": True,
                "release_receipt_requires_complete_audit": True,
            },
            "checks": [{"name": completion.RELEASE_CHECK, "status": "pass"}],
        },
    )
    _write(receipt_path, {"profile": completion.PROFILE})
    cofitok_artifact.write_bytes(b"cofitok")
    dense_artifact.write_bytes(b"dense")
    monkeypatch.setattr(
        finalizer,
        "verify_generation_release_receipt",
        lambda receipt, artifact: {"completion_profile": completion.PROFILE},
    )
    evidence = finalizer.validate_completion_result(
        completion_audit_path=audit_path,
        release_receipt_path=receipt_path,
        cofitok_artifact=cofitok_artifact,
        dense_artifact=dense_artifact,
        expected=expected,
    )
    assert evidence["completion_audit"]["sha256"]
    assert set(evidence["artifacts"]) == {"cofitok", "dense_identity"}


def test_capacity_full_completion_empty_workspace_is_incomplete(
    tmp_path: Path,
) -> None:
    output = tmp_path / "completion.json"
    missing = tmp_path / "missing.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    command = [
        sys.executable,
        str(ROOT / "scripts/audit_generation_capacity_full_completion.py"),
        "--project-root",
        str(ROOT),
        "--output-root",
        str(tmp_path / "empty"),
        "--training-project",
        str(ROOT),
        "--formal-project",
        str(ROOT),
        "--training-supervisor-status",
        str(missing),
        "--training-supervisor-deployment",
        str(missing),
        "--posteval-supervisor-status",
        str(missing),
        "--posteval-supervisor-deployment",
        str(missing),
        "--expected-training-supervisor-deployment-sha256",
        "a" * 64,
        "--expected-posteval-supervisor-deployment-sha256",
        "b" * 64,
        "--expected-training-launch-receipt-sha256",
        "c" * 64,
        "--expected-final-gate-sha256",
        "d" * 64,
        "--expected-training-revision",
        "1" * 40,
        "--expected-training-tree",
        "2" * 40,
        "--expected-training-branch",
        "train",
        "--expected-evaluation-revision",
        "3" * 40,
        "--expected-evaluation-tree",
        "4" * 40,
        "--expected-evaluation-branch",
        "eval",
        "--expected-export-revision",
        "3" * 40,
        "--expected-export-branch",
        "eval",
        "--output",
        str(output),
        "--allow-incomplete",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["profile"] == completion.PROFILE
    assert report["status"] == "incomplete"
    assert report["complete"] is False
    assert report["failed_checks"] == []
    assert completion.RELEASE_CHECK in report["missing_checks"]
