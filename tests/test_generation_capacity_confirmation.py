from __future__ import annotations

import copy
from hashlib import sha256

import pytest

from cofitok.generation.capacity_confirmation import (
    CONFIRMATION_DIRNAME,
    PREPARATION_BOUNDARY,
    build_capacity_confirmation_preparation,
    validate_capacity_confirmation_preparation_contract,
)
from cofitok.generation.capacity_confirmation_execution import (
    EXECUTION_BOUNDARY,
    LAUNCH_BOUNDARY,
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    STAGE_AUTHORIZATION_ROLE,
    STAGE_AUTHORIZATION_SCHEMA,
    STAGE_AUTHORIZATION_SCOPE,
    STAGE_BOUNDARY,
    build_capacity_confirmation_execution_authorization,
    build_capacity_confirmation_launch_receipt,
    capacity_confirmation_execution_lock_path,
    validate_capacity_confirmation_launch_receipt_contract,
)
from cofitok.generation.capacity_screen import ARM_NAMES, ARM_SPECS


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}
SCREEN_ROOT = "/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1"
ROOT = f"{SCREEN_ROOT}/{CONFIRMATION_DIRNAME}"
LOCK = capacity_confirmation_execution_lock_path(ROOT)


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": len(name) + 10,
        "sha256": sha256(name.encode("utf-8")).hexdigest(),
    }


def _screen_sources() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    launch_id = _identity("screen_launch")
    arm_ids = {arm: _identity(f"screen_{arm}") for arm in ARM_NAMES}
    result = {
        "scientific_status": "screen_pass",
        "failed_checks": [],
        "next_stage": {"capacity_confirmation_preparation_allowed": True},
        "source_evidence": {
            "launch_receipt": launch_id,
            "arm_validations": arm_ids,
        },
    }
    launch = {"execution_checkout": GIT, "output_root": SCREEN_ROOT}
    arms: dict[str, dict[str, object]] = {}
    for index, arm in enumerate(ARM_NAMES):
        checkpoint = {
            "path": f"{SCREEN_ROOT}/{arm}/checkpoint_step_00010000.pt",
            "bytes": 1000 + index,
            "sha256": sha256(f"checkpoint-{arm}".encode()).hexdigest(),
            "step": 10_000,
            "integrity_manifest": _identity(f"sidecar_{arm}"),
        }
        arms[arm] = {
            "execution_git": GIT,
            "sources": {
                "config": _identity(f"config_{arm}"),
                "training_report": _identity(f"training_{arm}"),
            },
            "training": {"checkpoint": checkpoint},
            "checkpoint_evaluation": {
                "summary": {
                    "ordered_endpoint_clean_mse": 0.03,
                    "order_count": 6 if arm.endswith("cofitok") else 1,
                }
            },
            "rollout": {
                "summary": {
                    "final_reconstruction_x0_mse": 0.03,
                    "predicted_x0_high_frequency_ratio": {"91": 0.2},
                }
            },
        }
    return result, launch, arms, arm_ids


def _preparation(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    result, launch, arms, arm_ids = _screen_sources()
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation.validate_capacity_screen_result_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation.validate_capacity_screen_launch_receipt_contract",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation.validate_capacity_screen_arm_validation",
        lambda value: value,
    )
    return build_capacity_confirmation_preparation(
        screen_result=result,
        screen_result_identity=_identity("screen_result"),
        screen_launch_receipt=launch,
        screen_launch_receipt_identity=_identity("screen_launch"),
        screen_arm_validations=arms,
        screen_arm_validation_identities=arm_ids,
        preparation_git=GIT,
        output_root=ROOT,
    )


def _stage(preparation_id: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": STAGE_AUTHORIZATION_SCHEMA,
        "role": STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "selection": {
            "preparation": preparation_id,
            "execution_checkout": GIT,
            "output_root": ROOT,
            "frozen_arms": list(ARM_NAMES),
            "checkpoint_step": 10_000,
            "samples_per_arm": 10_000,
            "sample_steps": 100,
            "sampling_batch_size": 4,
        },
        "approval_record": {
            "approved_by": "user",
            "approved_at": "2026-09-03T00:00:00+08:00",
            "source_instruction": "complete the capacity qualification",
        },
        "authorization_boundary": copy.deepcopy(STAGE_BOUNDARY),
    }


def test_prepares_exact_frozen_four_arm_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _preparation(monkeypatch)
    assert report["evaluation_contract"]["samples_per_arm"] == 10_000
    assert report["selection"]["fresh_training_allowed"] is False
    assert report["authorization_boundary"] == PREPARATION_BOUNDARY
    assert validate_capacity_confirmation_preparation_contract(report) == report


def test_confirmation_preparation_rejects_wrong_output_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, launch, arms, arm_ids = _screen_sources()
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation.validate_capacity_screen_result_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation.validate_capacity_screen_launch_receipt_contract",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation.validate_capacity_screen_arm_validation",
        lambda value: value,
    )
    with pytest.raises(ValueError, match="output root differs"):
        build_capacity_confirmation_preparation(
            screen_result=result,
            screen_result_identity=_identity("screen_result"),
            screen_launch_receipt=launch,
            screen_launch_receipt_identity=_identity("screen_launch"),
            screen_arm_validations=arms,
            screen_arm_validation_identities=arm_ids,
            preparation_git=GIT,
            output_root=f"{SCREEN_ROOT}/other",
        )


def test_builds_evaluation_only_launch_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _preparation(monkeypatch)
    preparation_id = _identity("confirmation_preparation")
    authorization = build_capacity_confirmation_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_id,
        stage_authorization=_stage(preparation_id),
        stage_authorization_identity=_identity("confirmation_stage"),
        execution_checkout=GIT,
        output_root=ROOT,
    )
    assert authorization["authorization_boundary"] == EXECUTION_BOUNDARY
    assert authorization["authorization_boundary"]["training_launch_allowed"] is False
    storage = {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "status": "pass",
        "filesystem": {
            "path": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
            "free_bytes": 500 * 1024**3,
        },
        "plan": {"sample_count": 40_000},
        "headroom_bytes": 400 * 1024**3,
    }
    live = {
        "schema_version": LIVE_SNAPSHOT_SCHEMA,
        "role": LIVE_SNAPSHOT_ROLE,
        "status": "pass",
        "execution_checkout": GIT,
        "gpu_inventory": [
            {
                "memory_used_mib": 0,
                "memory_total_mib": 97_887,
                "utilization_percent": 0,
            }
        ],
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root": ROOT,
        "execution_lock": LOCK,
        "output_root_absent": True,
        "execution_lock_free": True,
        "free_bytes": 500 * 1024**3,
        "runtime_environment_sha256": "c" * 64,
        "dataset_identity_sha256": "d" * 64,
        "captured_at": "2026-09-03T00:00:00+00:00",
    }
    output_dirs = {arm: f"{ROOT}/{arm}" for arm in ARM_NAMES}
    launch = build_capacity_confirmation_launch_receipt(
        preparation=preparation,
        preparation_identity=preparation_id,
        execution_authorization=authorization,
        execution_authorization_identity=_identity("confirmation_authorization"),
        storage_capacity=storage,
        storage_capacity_identity=_identity("confirmation_storage"),
        live_snapshot=live,
        live_snapshot_identity=_identity("confirmation_live"),
        execution_checkout=GIT,
        output_root=ROOT,
        output_dirs=output_dirs,
        execution_lock=LOCK,
        evaluation_state_absent_at_launch=True,
    )
    assert launch["authorization_boundary"] == LAUNCH_BOUNDARY
    assert launch["authorization_boundary"]["full_300k_launch_allowed"] is False
    assert validate_capacity_confirmation_launch_receipt_contract(
        launch, expected_execution_checkout=GIT
    ) == launch


def test_confirmation_stage_cannot_authorize_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _preparation(monkeypatch)
    preparation_id = _identity("confirmation_preparation")
    stage = _stage(preparation_id)
    stage["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="stage authorization differs"):
        build_capacity_confirmation_execution_authorization(
            preparation=preparation,
            preparation_identity=preparation_id,
            stage_authorization=stage,
            stage_authorization_identity=_identity("confirmation_stage"),
            execution_checkout=GIT,
            output_root=ROOT,
        )


def test_parameter_counts_remain_authoritative() -> None:
    assert ARM_SPECS["base128_cofitok"]["parameter_count"] == 62_834_083
    assert ARM_SPECS["base256_cofitok"]["parameter_count"] == 250_153_763


def test_confirmation_lock_does_not_mutate_the_locked_screen_root() -> None:
    assert not LOCK.startswith(f"{SCREEN_ROOT}/")
