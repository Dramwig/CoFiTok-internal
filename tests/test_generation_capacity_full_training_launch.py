from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.generation import capacity_full_training_launch as launch
from cofitok.generation.capacity_full_readiness import (
    CAPACITY_FULL_READINESS_BOUNDARY,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
)
from cofitok.training.authorization import (
    build_capacity_full_training_authorization,
    capture_generation_training_authorization,
    validate_checkpoint_training_authorization,
    validate_generation_training_authorization,
)
from cofitok.reporting import file_sha256


EXECUTION_REVISION = "1" * 40
EXECUTION_TREE = "2" * 40
EXECUTION_BRANCH = "scale/execution"
TRAINING_REVISION = "3" * 40
TRAINING_TREE = "4" * 40
TRAINING_BRANCH = "scale/training"
READINESS_REVISION = "5" * 40
READINESS_TREE = "6" * 40
READINESS_BRANCH = "scale/readiness"
FULL_ROOT = "/checkpoints/stability_capacity_full_300k_v1"


def _identity(path: str, digit: str = "7") -> dict[str, object]:
    return {"path": path, "bytes": 10, "sha256": digit * 64}


def _git(revision: str, tree: str, branch: str) -> dict[str, object]:
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def _standing() -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "cofitok_standing_experiment_authorization_record",
        "status": "active",
        "instruction": {},
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def _receipt(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    runtime = {
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "effective_batch_size": 64,
        "baseline": {"micro_batch_size": 1, "gradient_accumulation_steps": 64},
        "candidate_count": 5,
        "runtime_environment_sha256": "8" * 64,
        "dataset_identity_sha256": "9" * 64,
        "estimated_speedup_over_baseline": 1.2,
    }
    run_dirs = {
        "cofitok": f"{FULL_ROOT}/cofitok",
        "dense_identity": f"{FULL_ROOT}/dense_identity",
    }
    normalized_storage = {
        "reference_checkpoint_bytes_each": 1_000_000_000,
        "checkpoint_size_multiplier": 1.0,
        "checkpoint_bytes_each": 1_000_000_000,
        "checkpoint_count": 16,
        "sample_count": 116_640,
        "required_free_bytes": 100,
        "free_bytes": 200,
        "headroom_bytes": 100,
    }
    monkeypatch.setattr(
        launch,
        "validate_capacity_full_readiness",
        lambda *args, **kwargs: {
            "status": "pass",
            "full_output_root": FULL_ROOT,
            "training_run_dirs": copy.deepcopy(run_dirs),
            "runtime_selection": copy.deepcopy(runtime),
            "storage_capacity": copy.deepcopy(normalized_storage),
            "full_training_launch_allowed": True,
            "authorization_boundary": copy.deepcopy(
                CAPACITY_FULL_READINESS_BOUNDARY
            ),
        },
    )
    monkeypatch.setattr(
        launch,
        "validate_capacity_full_storage",
        lambda *args, **kwargs: copy.deepcopy(normalized_storage),
    )
    monkeypatch.setattr(
        launch,
        "validate_standing_experiment_authorization",
        lambda value: copy.deepcopy(value),
    )
    readiness_identity = _identity("/evidence/readiness.json", "a")
    standing_identity = _identity("/evidence/standing.json", "b")
    cofitok_config = _identity("/target/cofitok.json", "c")
    dense_config = _identity("/target/dense.json", "d")
    readiness_git = _git(READINESS_REVISION, READINESS_TREE, READINESS_BRANCH)
    readiness = {
        "authorization_boundary": copy.deepcopy(CAPACITY_FULL_READINESS_BOUNDARY),
        "source_reports": {
            "target_cofitok_config": _identity("/source/cofitok.json", "c"),
            "target_dense_identity_config": _identity("/source/dense.json", "d"),
        },
    }
    status = {
        "schema_version": 1,
        "role": "capacity_full_300k_readiness_supervisor",
        "status": "pass",
        "detail": "capacity_full_readiness_completed_without_training_launch",
        "expected": {
            "readiness_git": copy.deepcopy(readiness_git),
            "standing_authorization": copy.deepcopy(standing_identity),
            "full_output_root": FULL_ROOT,
            "capacity_100k_checkpoint_resume_allowed": False,
            "training_launch_allowed_by_supervisor": False,
            "full_300k_launch_allowed_by_supervisor": False,
        },
        "readiness": {
            "identity": copy.deepcopy(readiness_identity),
            "full_training_launch_allowed_by_artifact": True,
            "training_launch_performed_by_supervisor": False,
            "full_300k_launch_performed_by_supervisor": False,
        },
        "readiness_execution_only": True,
        "training_launch_performed": False,
        "full_300k_launch_performed": False,
        "authorization_boundary": copy.deepcopy(
            launch.READINESS_SUPERVISOR_BOUNDARY
        ),
    }
    deployment_identity = _identity("/evidence/deployment.json", "e")
    deployment = {
        "schema_version": 1,
        "role": "capacity_full_300k_readiness_supervisor_deployment",
        "status": "active",
        "git": copy.deepcopy(readiness_git),
        "checkout": {"git": copy.deepcopy(readiness_git)},
        "standing_authorization": copy.deepcopy(standing_identity),
        "full_output_root": FULL_ROOT,
        "authorization_boundary": copy.deepcopy(
            launch.READINESS_DEPLOYMENT_BOUNDARY
        ),
        "target_state_at_deployment": {
            "capacity_100k_checkpoint_resume_allowed": False,
            "cofitok_training_state_absent": True,
            "dense_identity_training_state_absent": True,
            "full_output_root_absent": True,
        },
    }
    clarification = {
        "schema_version": 1,
        "role": (
            "capacity_full_300k_readiness_supervisor_deployment_validation_"
            "clarification"
        ),
        "status": "clarified",
        "deployment_receipt": copy.deepcopy(deployment_identity),
        "deployed_git": copy.deepcopy(readiness_git),
        "authorization_effect": {
            "changes_authorization": False,
            "training_launch_allowed_by_supervisor": False,
            "full_300k_launch_allowed_by_supervisor": False,
        },
    }
    return launch.build_capacity_full_training_launch(
        readiness=readiness,
        readiness_identity=readiness_identity,
        readiness_supervisor_status=status,
        readiness_supervisor_status_identity=_identity("/evidence/status.json", "f"),
        readiness_deployment_receipt=deployment,
        readiness_deployment_receipt_identity=deployment_identity,
        readiness_deployment_clarification=clarification,
        readiness_deployment_clarification_identity=_identity(
            "/evidence/clarification.json", "0"
        ),
        standing_authorization=_standing(),
        standing_authorization_identity=standing_identity,
        target_config_identities={
            "cofitok": cofitok_config,
            "dense_identity": dense_config,
        },
        launch_storage={"status": "pass"},
        launch_storage_identity=_identity("/evidence/storage.json", "1"),
        readiness_to_training_transition={
            "source_revision": READINESS_REVISION,
            "target_revision": TRAINING_REVISION,
            "source_is_ancestor": True,
            "changed_paths": sorted(
                launch.ALLOWED_READINESS_TO_TRAINING_TRANSITION_PATHS
            ),
            "target_configs_byte_identical": True,
            "model_data_diffusion_compute_paths_changed": [],
            "runtime_requalification_required": False,
            "authorization_metadata_only_runtime_change": True,
        },
        execution_git=_git(EXECUTION_REVISION, EXECUTION_TREE, EXECUTION_BRANCH),
        training_git=_git(TRAINING_REVISION, TRAINING_TREE, TRAINING_BRANCH),
        expected_execution_revision=EXECUTION_REVISION,
        expected_execution_tree=EXECUTION_TREE,
        expected_execution_branch=EXECUTION_BRANCH,
        expected_training_revision=TRAINING_REVISION,
        expected_training_tree=TRAINING_TREE,
        expected_training_branch=TRAINING_BRANCH,
        expected_readiness_revision=READINESS_REVISION,
        expected_readiness_tree=READINESS_TREE,
        expected_readiness_branch=READINESS_BRANCH,
    )


def _validate(report: dict[str, object]) -> dict[str, object]:
    return launch.validate_capacity_full_training_launch(
        report,
        expected_execution_revision=EXECUTION_REVISION,
        expected_execution_tree=EXECUTION_TREE,
        expected_execution_branch=EXECUTION_BRANCH,
        expected_training_revision=TRAINING_REVISION,
        expected_training_tree=TRAINING_TREE,
        expected_training_branch=TRAINING_BRANCH,
        expected_readiness_revision=READINESS_REVISION,
        expected_readiness_tree=READINESS_TREE,
        expected_readiness_branch=READINESS_BRANCH,
    )


def test_capacity_full_launch_authorizes_fresh_training_without_quality_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _receipt(monkeypatch)
    evidence = _validate(report)
    assert evidence["status"] == "authorized"
    assert report["training_plan"]["start_step"] == 0
    assert report["training_plan"]["milestone_steps"] == [
        50_000,
        100_000,
        200_000,
        300_000,
    ]
    assert report["authorization_boundary"][
        "training_authorization_is_quality_promotion_gate"
    ] is False
    assert report["authorization_boundary"]["formal_generation_claim_allowed"] is False
    assert report["authorization_boundary"]["release_authorization_allowed"] is False


def test_capacity_full_launch_rejects_transition_or_claim_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _receipt(monkeypatch)
    escalated = copy.deepcopy(report)
    escalated["authorization_boundary"]["formal_generation_claim_allowed"] = True
    with pytest.raises(ValueError, match="launch contract differs"):
        _validate(escalated)
    changed = copy.deepcopy(report)
    changed["readiness_to_training_transition"]["changed_paths"].append(
        "src/cofitok/models/generation.py"
    )
    with pytest.raises(ValueError, match="transition differs"):
        _validate(changed)


def test_capacity_full_training_authorization_binds_receipt_and_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _receipt(monkeypatch)
    authorization = build_capacity_full_training_authorization(
        report,
        receipt_path="/evidence/capacity_full_training_launch.json",
        receipt_bytes=100,
        receipt_sha256="2" * 64,
    )
    evidence = validate_generation_training_authorization(
        authorization,
        expected_gate=report,
    )
    assert evidence["stage"] == launch.CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE
    assert evidence["decision"] == launch.CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION
    integrity = {
        "authorization_stage": evidence["stage"],
        "authorization_decision": evidence["decision"],
        "authorization_gate_bytes": evidence["gate_bytes"],
        "authorization_gate_sha256": evidence["gate_sha256"],
        "authorization_gate_identity_sha256": evidence["gate_identity_sha256"],
    }
    checkpoint = {
        "config": {
            "data": {"dataset": "imagenet_256"},
            "runtime": {"steps": 300_000},
        },
        "extra_state": {"training_authorization": authorization},
    }
    assert validate_checkpoint_training_authorization(checkpoint, integrity) == authorization
    drifted = copy.deepcopy(authorization)
    drifted["validated_thresholds"]["formal_generation_claim_allowed"] = True
    with pytest.raises(ValueError, match="boundary differs"):
        validate_generation_training_authorization(drifted)


def test_capacity_full_training_authorization_capture_physically_replays_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _receipt(monkeypatch)
    identities: dict[str, dict[str, object]] = {}
    for index, name in enumerate(sorted(report["source_reports"])):
        path = tmp_path / f"source_{index}_{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        identities[name] = {
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    report["source_reports"] = copy.deepcopy(identities)
    report["standing_authorization"]["source"] = copy.deepcopy(
        identities["standing_authorization"]
    )
    report["training_plan"]["target_configs"] = {
        "cofitok": copy.deepcopy(identities["target_cofitok_config"]),
        "dense_identity": copy.deepcopy(
            identities["target_dense_identity_config"]
        ),
    }
    receipt = tmp_path / "training_launch_receipt.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    authorization = capture_generation_training_authorization(receipt)
    assert authorization["stage"] == launch.CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE
    assert authorization["gate_sha256"] == file_sha256(receipt)
    changed = Path(str(identities["launch_storage_capacity"]["path"]))
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        capture_generation_training_authorization(receipt)
