from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    dataset_provenance_identity_sha256,
)
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.terminal_snr_confirmation import (
    CONFIRMATION_DIRNAME,
    PREPARATION_BOUNDARY,
    SAMPLE_SEED,
    build_terminal_snr_confirmation_preparation,
    validate_terminal_snr_confirmation_preparation_contract,
)
from cofitok.generation.terminal_snr_confirmation_execution import (
    EXECUTION_BOUNDARY,
    LAUNCH_BOUNDARY,
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    build_terminal_snr_confirmation_execution_authorization,
    build_terminal_snr_confirmation_launch_receipt,
    build_terminal_snr_confirmation_stage_authorization,
    terminal_snr_confirmation_execution_lock_path,
    validate_terminal_snr_confirmation_execution_authorization_physical,
    validate_terminal_snr_confirmation_launch_receipt_contract,
)
from cofitok.generation.terminal_snr_screen import (
    ARM_NAMES,
    ARM_SPECS,
    SAMPLE_SEED as SCREEN_SEED,
)
from cofitok.inference_replay import file_identity


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/generation-terminal-snr-endpoint-screen-v1-20260907",
    "tracked_dirty": False,
}
PREPARATION_GIT = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "analysis/generation-terminal-snr-confirmation-v1-20260908",
    "tracked_dirty": False,
}
SCREEN_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1"
)
ROOT = f"{SCREEN_ROOT}/{CONFIRMATION_DIRNAME}"
LOCK = terminal_snr_confirmation_execution_lock_path(ROOT)


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": len(name) + 10,
        "sha256": sha256(name.encode()).hexdigest(),
    }


def _dataset_provenance() -> dict[str, object]:
    spec = FORMAL_GENERATION_DATASETS["imagenet_256"]
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "formal": True,
        "dataset": "imagenet_256",
        "dataset_root": "/root/autodl-tmp/CoFiTok/datasets/imagenet_256",
        "manifest": {
            "relative_path": "metadata/image_manifest.jsonl",
            "bytes": spec.manifest_bytes,
            "sha256": spec.manifest_sha256,
        },
        "splits": {"train": spec.train_images, "val": spec.val_images},
        "issues": [],
    }
    report["identity_sha256"] = dataset_provenance_identity_sha256(report)
    return report


def _sources(*, passed: bool = True, duplicate_checkpoint: bool = False):
    launch_id = _identity("terminal_launch")
    arm_ids = {arm: _identity(f"screen_{arm}") for arm in ARM_NAMES}
    result = {
        "scientific_status": "pass" if passed else "hold",
        "screen_pass": passed,
        "failed_checks": [] if passed else ["cofitok.relative_fid_improvement"],
        "result_git": GIT,
        "next_stage": {
            "route": "frozen_10k_confirmation_preparation" if passed else "hold",
            "frozen_confirmation_preparation_allowed": passed,
            "frozen_confirmation_launch_allowed": False,
        },
        "source_evidence": {
            "launch_receipt": launch_id,
            "arm_validations": arm_ids,
        },
    }
    launch = {"execution_checkout": GIT, "output_root": SCREEN_ROOT}
    arms = {}
    for index, arm in enumerate(ARM_NAMES):
        checkpoint_name = "shared" if duplicate_checkpoint else arm
        checkpoint = {
            "path": f"{SCREEN_ROOT}/training/{arm}/checkpoint_step_00010000.pt",
            "bytes": 1000 + index,
            "sha256": sha256(f"checkpoint-{checkpoint_name}".encode()).hexdigest(),
            "integrity_manifest": _identity(f"sidecar_{arm}"),
        }
        spec = ARM_SPECS[arm]
        arms[arm] = {
            "arm": arm,
            "condition": spec["condition"],
            "method": spec["method"],
            "endpoint_fraction": spec["endpoint_fraction"],
            "execution_git": GIT,
            "sources": {
                "config": _identity(f"config_{arm}"),
                "training_report": _identity(f"training_{arm}"),
            },
            "training": {
                "checkpoint": checkpoint,
                "validation": {"completed_steps": 10_000},
            },
            "checkpoint_evaluation": {"summary": {"ordered_rank_by_path_auc": 1}},
            "rollout": {
                "summary": {"terminal_raw_x0_clip_fraction": 0.9},
                "terminal_raw_x0_clipping": {"raw_x0_clip_fraction": 0.9},
            },
        }
    return result, launch, arms, arm_ids


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation.validate_terminal_snr_screen_result_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation.validate_terminal_snr_screen_launch_receipt_physical",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation.replay_terminal_snr_screen_result",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation.validate_terminal_snr_screen_validation_receipt",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation.validate_terminal_snr_screen_arm_validation",
        lambda value: value,
    )


def _build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_pass: bool = True,
    **source_options,
):
    _patch(monkeypatch)
    result, launch, arms, arm_ids = _sources(**source_options)
    result_id = _identity("terminal_result")
    result_validation = {
        "validator_git": PREPARATION_GIT,
        "result": result_id,
        "screen_pass": validation_pass,
        "failed_checks": [] if validation_pass else ["screen.failed"],
    }
    return build_terminal_snr_confirmation_preparation(
        screen_result=result,
        screen_result_identity=result_id,
        screen_result_validation=result_validation,
        screen_result_validation_identity=_identity("terminal_result_validation"),
        screen_launch_receipt=launch,
        screen_launch_receipt_identity=_identity("terminal_launch"),
        screen_arm_validations=arms,
        screen_arm_validation_identities=arm_ids,
        preparation_git=PREPARATION_GIT,
        output_root=ROOT,
    )


def test_prepares_disjoint_frozen_10k_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _build(monkeypatch)
    assert report["evaluation_contract"]["samples_per_arm"] == 10_000
    assert report["evaluation_contract"]["seed"] == SAMPLE_SEED
    assert SAMPLE_SEED != SCREEN_SEED
    assert report["selection"]["fresh_training_allowed"] is False
    assert report["authorization_boundary"] == PREPARATION_BOUNDARY
    assert validate_terminal_snr_confirmation_preparation_contract(report) == report


def test_rejects_nonpassing_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="requires a passing"):
        _build(monkeypatch, passed=False)


def test_rejects_nonpassing_screen_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="result validation differs"):
        _build(monkeypatch, validation_pass=False)


def test_rejects_wrong_output_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    result, launch, arms, arm_ids = _sources()
    result_id = _identity("terminal_result")
    with pytest.raises(ValueError, match="output root differs"):
        build_terminal_snr_confirmation_preparation(
            screen_result=result,
            screen_result_identity=result_id,
            screen_result_validation={
                "validator_git": PREPARATION_GIT,
                "result": result_id,
                "screen_pass": True,
                "failed_checks": [],
            },
            screen_result_validation_identity=_identity(
                "terminal_result_validation"
            ),
            screen_launch_receipt=launch,
            screen_launch_receipt_identity=_identity("terminal_launch"),
            screen_arm_validations=arms,
            screen_arm_validation_identities=arm_ids,
            preparation_git=PREPARATION_GIT,
            output_root=f"{SCREEN_ROOT}/other",
        )


def test_rejects_duplicate_frozen_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="checkpoints are duplicated"):
        _build(monkeypatch, duplicate_checkpoint=True)


def test_contract_rejects_training_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_terminal_snr_confirmation_preparation_contract(altered)


def test_contract_rejects_reused_screen_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered["evaluation_contract"]["seed"] = SCREEN_SEED
    with pytest.raises(ValueError, match="contract differs"):
        validate_terminal_snr_confirmation_preparation_contract(altered)


def _stage(preparation_id: dict[str, object]) -> dict[str, object]:
    return build_terminal_snr_confirmation_stage_authorization(
        preparation_identity=preparation_id,
        execution_checkout=PREPARATION_GIT,
        output_root=ROOT,
        approved_by="user",
        approved_at="2026-09-08T03:00:00+08:00",
        source_instruction="complete exposure/capacity qualification first",
    )


def test_builds_evaluation_only_authorization_and_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _build(monkeypatch)
    preparation_id = _identity("confirmation_preparation")
    authorization = build_terminal_snr_confirmation_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_id,
        stage_authorization=_stage(preparation_id),
        stage_authorization_identity=_identity("confirmation_stage"),
        execution_checkout=PREPARATION_GIT,
        output_root=ROOT,
    )
    assert authorization["authorization_boundary"] == EXECUTION_BOUNDARY
    assert authorization["authorization_boundary"]["training_launch_allowed"] is False
    storage = {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "status": "pass",
        "filesystem": {"path": SCREEN_ROOT, "free_bytes": 500 * 1024**3},
        "plan": {"sample_count": 40_000},
        "headroom_bytes": 300 * 1024**3,
    }
    runtime_environment = {"schema_version": 1, "device": {"type": "cuda"}}
    dataset_provenance = _dataset_provenance()
    live = {
        "schema_version": LIVE_SNAPSHOT_SCHEMA,
        "role": LIVE_SNAPSHOT_ROLE,
        "status": "pass",
        "execution_checkout": PREPARATION_GIT,
        "source_evidence": {
            "preparation": preparation_id,
            "execution_authorization": _identity("confirmation_authorization"),
            "reference_config": preparation["frozen_arms"]["control_cofitok"][
                "config"
            ],
            "frozen_checkpoint_verification": {
                arm: {
                    "checkpoint": {
                        key: preparation["frozen_arms"][arm]["checkpoint"][key]
                        for key in ("path", "bytes", "sha256")
                    },
                    "integrity_manifest": preparation["frozen_arms"][arm][
                        "checkpoint"
                    ]["integrity_manifest"],
                    "step": 10_000,
                }
                for arm in ARM_NAMES
            },
        },
        "output_root": ROOT,
        "execution_lock": LOCK,
        "captured_at": "2026-09-08T03:01:00+08:00",
        "gpu_inventory": [
            {
                "memory_used_mib": 0,
                "memory_total_mib": 97_887,
                "utilization_percent": 0,
            }
        ],
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root_absent": True,
        "execution_lock_free": True,
        "free_bytes": 500 * 1024**3,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256(
            runtime_environment
        ),
        "dataset_provenance": dataset_provenance,
        "dataset_identity_sha256": dataset_provenance["identity_sha256"],
    }
    launch = build_terminal_snr_confirmation_launch_receipt(
        preparation=preparation,
        preparation_identity=preparation_id,
        execution_authorization=authorization,
        execution_authorization_identity=_identity("confirmation_authorization"),
        storage_capacity=storage,
        storage_capacity_identity=_identity("confirmation_storage"),
        live_snapshot=live,
        live_snapshot_identity=_identity("confirmation_live"),
        execution_checkout=PREPARATION_GIT,
        output_root=ROOT,
        output_dirs={arm: f"{ROOT}/{arm}" for arm in ARM_NAMES},
        execution_lock=LOCK,
        evaluation_state_absent_at_launch=True,
    )
    assert launch["authorization_boundary"] == LAUNCH_BOUNDARY
    assert launch["authorization_boundary"]["full_300k_launch_allowed"] is False
    assert (
        validate_terminal_snr_confirmation_launch_receipt_contract(
            launch, expected_execution_checkout=PREPARATION_GIT
        )
        == launch
    )


def test_stage_cannot_authorize_training(monkeypatch: pytest.MonkeyPatch) -> None:
    preparation = _build(monkeypatch)
    preparation_id = _identity("confirmation_preparation")
    stage = _stage(preparation_id)
    stage["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="stage authorization differs"):
        build_terminal_snr_confirmation_execution_authorization(
            preparation=preparation,
            preparation_identity=preparation_id,
            stage_authorization=stage,
            stage_authorization_identity=_identity("confirmation_stage"),
            execution_checkout=PREPARATION_GIT,
            output_root=ROOT,
        )


def test_execution_authorization_physically_binds_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparation = _build(monkeypatch)
    preparation_id = _identity("confirmation_preparation")
    stage = _stage(preparation_id)
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    stage_id = file_identity(stage_path)
    authorization = build_terminal_snr_confirmation_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_id,
        stage_authorization=stage,
        stage_authorization_identity=stage_id,
        execution_checkout=PREPARATION_GIT,
        output_root=ROOT,
    )
    assert (
        validate_terminal_snr_confirmation_execution_authorization_physical(
            authorization,
            preparation=preparation,
            preparation_identity=preparation_id,
            expected_execution_checkout=PREPARATION_GIT,
            expected_output_root=ROOT,
        )
        == authorization
    )
    stage_path.write_text(json.dumps({**stage, "status": "revoked"}), encoding="utf-8")
    with pytest.raises(ValueError, match="physical identity differs"):
        validate_terminal_snr_confirmation_execution_authorization_physical(
            authorization,
            preparation=preparation,
            preparation_identity=preparation_id,
            expected_execution_checkout=PREPARATION_GIT,
            expected_output_root=ROOT,
        )


def test_lock_is_adjacent_to_screen_output() -> None:
    assert not LOCK.startswith(f"{ROOT}/")
