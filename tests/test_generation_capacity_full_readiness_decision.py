from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.configs import load_config
from cofitok.generation import capacity_full_readiness_decision as decision
from cofitok.generation.capacity_completion_result import _policy
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_EXACT_TEXT,
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
)
from cofitok.reporting import to_jsonable
from scripts.validate_generation_configs import validate_pair


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIGS = {
    "cofitok": ROOT
    / "configs/generation/imagenet256_stability_capacity_probe_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json",
    "dense_identity": ROOT
    / "configs/generation/imagenet256_stability_capacity_probe_"
    "rollout_x0_u2_ema_teacher_dense_100k.json",
}
TARGET_CONFIGS = {
    "cofitok": ROOT
    / "configs/generation/imagenet256_stability_rgbtail3_"
    "rollout_x0_u2_ema_teacher_k8_300k.json",
    "dense_identity": ROOT
    / "configs/generation/imagenet256_stability_rollout_x0_u2_"
    "ema_teacher_dense_300k.json",
}
RESULT_REVISION = "1" * 40
RESULT_TREE = "2" * 40
DECISION_REVISION = "3" * 40
DECISION_TREE = "4" * 40
EXECUTION_REVISION = "5" * 40
EXECUTION_TREE = "6" * 40
TRAINING_REVISION = "7" * 40
TRAINING_TREE = "8" * 40
SHA256 = "9" * 64
RESULT_BRANCH = "scale/result"
DECISION_BRANCH = "scale/decision"
EXECUTION_BRANCH = "scale/execution"
TRAINING_BRANCH = "scale/training"


def _identity(name: str) -> dict[str, object]:
    return {"path": f"/tmp/{name}.json", "bytes": 1, "sha256": SHA256}


def _configs(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    return {
        method: json.loads(path.read_text(encoding="utf-8"))
        for method, path in paths.items()
    }


def _pair(paths: dict[str, Path], *, stage: str) -> dict[str, object]:
    return to_jsonable(
        validate_pair(
            load_config(paths["cofitok"]),
            load_config(paths["dense_identity"]),
            max_parameter_gap=0.02,
            stage=stage,
        )
    )


def _standing() -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_EXACT_TEXT,
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
            "received_at": "2026-08-14T00:00:00+00:00",
        },
        "preserved_safety_boundaries": STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    }


def _kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    recommendation = _policy(
        quality_status="hold",
        failed_checks=["cofitok_absolute_fid"],
        milestone_alerts=[],
        shared_strict_fid_improvement=True,
    )
    result_identity = _identity("result")
    launch_identity = _identity("launch")
    source_identities = {
        "cofitok": _identity("source_cofitok"),
        "dense_identity": _identity("source_dense"),
    }
    target_identities = {
        "cofitok": _identity("target_cofitok"),
        "dense_identity": _identity("target_dense"),
    }
    result_git = {
        "revision": RESULT_REVISION,
        "tree": RESULT_TREE,
        "branch": RESULT_BRANCH,
        "tracked_dirty": False,
    }
    monkeypatch.setattr(
        decision,
        "validate_capacity_completion_100k_result",
        lambda *args, **kwargs: {
            "recommended_next_stage": copy.deepcopy(recommendation)
        },
    )
    result = {
        "quality_screen": {
            "status": "hold",
            "failed_checks": ["cofitok_absolute_fid"],
        },
        "decision_support": {
            "step_50000_to_100000_shared_strict_fid_improvement": True,
            "recommended_next_stage": recommendation,
        },
        "source_reports": {
            "capacity_completion_launch_receipt": launch_identity
        },
    }
    waiter_identity = _identity("waiter")
    waiter_status = {
        "schema_version": 1,
        "role": "capacity_completion_100k_result_source_replay_waiter",
        "status": "completed",
        "detail": "source_replayed_capacity_completion_100k_result_emitted",
        "git": result_git,
        "capacity_completion_100k_result": result_identity,
        "recommended_next_stage": recommendation,
        "authorization_boundary": decision.RESULT_WAITER_BOUNDARY,
    }
    deployment = {
        "schema_version": 1,
        "role": "capacity_completion_100k_result_waiter_deployment",
        "status": "active",
        "git": result_git,
        "authorization_boundary": decision.RESULT_WAITER_BOUNDARY,
        "incremental_bundle": {"advertised_revision": RESULT_REVISION},
        "waiter": {
            "status_path": waiter_identity["path"],
            "result_path": result_identity["path"],
        },
    }
    launch = {
        "source_reports": {
            "cofitok_config": source_identities["cofitok"],
            "dense_config": source_identities["dense_identity"],
        }
    }
    return {
        "capacity_completion_result": result,
        "capacity_completion_result_identity": result_identity,
        "result_waiter_status": waiter_status,
        "result_waiter_status_identity": waiter_identity,
        "result_waiter_deployment_receipt": deployment,
        "result_waiter_deployment_receipt_identity": _identity("deployment"),
        "capacity_completion_launch_receipt": launch,
        "capacity_completion_launch_receipt_identity": launch_identity,
        "standing_authorization": _standing(),
        "standing_authorization_identity": _identity("standing"),
        "source_configs": _configs(SOURCE_CONFIGS),
        "source_config_identities": source_identities,
        "target_configs": _configs(TARGET_CONFIGS),
        "target_config_identities": target_identities,
        "source_pair_validation": _pair(
            SOURCE_CONFIGS,
            stage="stability_capacity_probe",
        ),
        "target_pair_validation": _pair(TARGET_CONFIGS, stage="stability_full"),
        "training_run_dirs": {
            "cofitok": "/checkpoints/stability_capacity_full_300k_v1/cofitok",
            "dense_identity": (
                "/checkpoints/stability_capacity_full_300k_v1/dense_identity"
            ),
        },
        "training_state_absent": True,
        "full_output_root": "/checkpoints/stability_capacity_full_300k_v1",
        "benchmark_root": (
            "/checkpoints/stability_capacity_full_300k_v1/"
            "reports/runtime_benchmark"
        ),
        "storage_path": "/checkpoints",
        "decision_builder_git": {
            "revision": DECISION_REVISION,
            "tree": DECISION_TREE,
            "branch": DECISION_BRANCH,
            "tracked_dirty": False,
        },
        "expected_decision_revision": DECISION_REVISION,
        "expected_decision_tree": DECISION_TREE,
        "expected_decision_branch": DECISION_BRANCH,
        "expected_result_execution_revision": EXECUTION_REVISION,
        "expected_result_execution_tree": EXECUTION_TREE,
        "expected_result_execution_branch": EXECUTION_BRANCH,
        "expected_training_revision": TRAINING_REVISION,
        "expected_training_tree": TRAINING_TREE,
        "expected_training_branch": TRAINING_BRANCH,
        "expected_result_revision": RESULT_REVISION,
        "expected_result_tree": RESULT_TREE,
        "expected_result_branch": RESULT_BRANCH,
    }


def test_capacity_full_config_bridge_is_exact_and_fresh_only() -> None:
    report = decision.validate_capacity_to_full_config_bridge(
        source_configs=_configs(SOURCE_CONFIGS),
        target_configs=_configs(TARGET_CONFIGS),
        source_pair_validation=_pair(
            SOURCE_CONFIGS,
            stage="stability_capacity_probe",
        ),
        target_pair_validation=_pair(TARGET_CONFIGS, stage="stability_full"),
    )
    assert report["status"] == "pass"
    assert report["methods"]["cofitok"]["fresh_start_required"] is True
    assert set(report["methods"]["cofitok"]["changes"]) == {
        "name",
        *decision.EXPECTED_BRIDGE_CHANGES,
    }

    target = _configs(TARGET_CONFIGS)
    target["cofitok"]["loss"]["epsilon_weight"] = 0.5
    with pytest.raises(ValueError, match="config bridge differs"):
        decision.validate_capacity_to_full_config_bridge(
            source_configs=_configs(SOURCE_CONFIGS),
            target_configs=target,
            source_pair_validation=_pair(
                SOURCE_CONFIGS,
                stage="stability_capacity_probe",
            ),
            target_pair_validation=_pair(TARGET_CONFIGS, stage="stability_full"),
        )


def test_capacity_full_readiness_decision_authorizes_benchmark_not_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = decision.build_capacity_full_readiness_decision(**_kwargs(monkeypatch))
    evidence = decision.validate_capacity_full_readiness_decision(
        report,
        expected_decision_revision=DECISION_REVISION,
        expected_decision_tree=DECISION_TREE,
        expected_decision_branch=DECISION_BRANCH,
        expected_result_revision=RESULT_REVISION,
        expected_result_tree=RESULT_TREE,
        expected_result_branch=RESULT_BRANCH,
    )
    authorization = evidence["execution_authorization"]
    assert authorization["readiness_gpu_runtime_benchmark_allowed"] is True
    assert authorization["training_launch_allowed"] is False
    assert authorization["full_300k_launch_allowed"] is False
    assert report["fresh_training_contract"][
        "capacity_100k_checkpoint_resume_allowed"
    ] is False


def test_capacity_full_readiness_decision_rejects_resume_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = decision.build_capacity_full_readiness_decision(**_kwargs(monkeypatch))
    report["fresh_training_contract"][
        "capacity_100k_checkpoint_resume_allowed"
    ] = True
    with pytest.raises(ValueError, match="fresh-training contract differs"):
        decision.validate_capacity_full_readiness_decision(
            report,
            expected_decision_revision=DECISION_REVISION,
            expected_decision_tree=DECISION_TREE,
            expected_decision_branch=DECISION_BRANCH,
            expected_result_revision=RESULT_REVISION,
            expected_result_tree=RESULT_TREE,
            expected_result_branch=RESULT_BRANCH,
        )


def test_capacity_full_readiness_branch_selection_is_exact() -> None:
    report = {
        "decision_support": {
            "recommended_next_stage": {"id": decision.SELECTED_RECOMMENDATION_ID}
        }
    }
    assert decision.readiness_branch_selected(report) is True
    report["decision_support"]["recommended_next_stage"]["id"] = (
        "build_source_compatible_formal_quality_gate"
    )
    assert decision.readiness_branch_selected(report) is False


def test_capacity_full_readiness_entrypoints_import() -> None:
    for module in (
        "scripts.build_generation_capacity_full_300k_readiness_decision",
        "scripts.verify_generation_capacity_full_300k_readiness_decision",
        "scripts.wait_for_generation_capacity_full_300k_readiness_decision",
    ):
        __import__(module)
