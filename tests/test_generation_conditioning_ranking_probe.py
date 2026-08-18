from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    EXECUTION_BOUNDARY,
    EXECUTION_AUTHORIZATION_MODE,
    EXECUTION_AUTHORIZATION_ROLE,
    PROBE_AUTHORIZATION_TEXT,
    PROBE_ROLE,
    PROBE_SCOPE,
    PROBE_STAGE,
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
    build_conditioning_ranking_probe_execution_authorization,
    conditioning_ranking_probe_contract,
    validate_conditioning_ranking_probe_approval,
    validate_conditioning_ranking_probe_execution_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "generation"


def _configs() -> dict[str, dict]:
    names = {
        "control_cofitok": (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_"
            "k8_probe1k.json"
        ),
        "control_dense": (
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_"
            "dense_probe1k.json"
        ),
        "ranked_cofitok": (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_"
            "classrank_k8_probe1k.json"
        ),
        "ranked_dense": (
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_"
            "classrank_dense_probe1k.json"
        ),
    }
    return {
        key: config_to_dict(load_config(CONFIG_ROOT / name))
        for key, name in names.items()
    }


def test_four_arm_probe_changes_only_the_shared_ranking_contract() -> None:
    configs = _configs()
    report = conditioning_ranking_probe_contract(**configs)

    assert report["valid"] is True, report["issues"]
    assert report["pair_contracts"]["control"]["valid"] is True
    assert report["pair_contracts"]["ranked"]["valid"] is True
    assert report["ranked_ranking_config"][
        "class_conditioning_ranking_weight"
    ] == 0.05
    assert report["execution_boundary"]["steps_per_run"] == 1_000
    assert report["gpu_execution_authorized"] is False


def test_four_arm_probe_rejects_nonranking_recipe_drift() -> None:
    configs = _configs()
    configs["ranked_dense"] = copy.deepcopy(configs["ranked_dense"])
    configs["ranked_dense"]["optimization"]["learning_rate"] = 2e-4

    report = conditioning_ranking_probe_contract(**configs)

    assert report["valid"] is False
    assert any("outside the ranking fields" in issue for issue in report["issues"])


def test_probe_approval_is_exact_and_non_authorizing_beyond_1k() -> None:
    revision = "a" * 40
    preparation_sha = "b" * 64
    output_root = (
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "conditioning_ranking_four_arm_probe1k_v1"
    )
    approval = {
        "schema_version": 1,
        "role": "generation_conditioning_ranking_probe_execution_approval",
        "status": "approved",
        "scope": PROBE_SCOPE,
        "user_authorization_text": PROBE_AUTHORIZATION_TEXT,
        "authorized_revision": revision,
        "preparation_report_sha256": preparation_sha,
        "output_root": output_root,
        "authorization_boundary": EXECUTION_BOUNDARY,
    }

    validated = validate_conditioning_ranking_probe_approval(
        approval,
        expected_revision=revision,
        expected_preparation_sha256=preparation_sha,
        expected_output_root=output_root,
    )

    assert validated["authorization_boundary"]["full_training_launch_allowed"] is False
    assert validated["authorization_boundary"]["followup_training_allowed"] is False

    approval["authorization_boundary"] = {
        **EXECUTION_BOUNDARY,
        "followup_training_allowed": True,
    }
    with pytest.raises(ValueError, match="exact scope"):
        validate_conditioning_ranking_probe_approval(
            approval,
            expected_revision=revision,
            expected_preparation_sha256=preparation_sha,
            expected_output_root=output_root,
        )


def test_probe_runbook_requires_fully_clean_idle_checkout() -> None:
    runbook = (
        ROOT
        / "artifacts"
        / "runbooks"
        / "generation_conditioning_ranking_four_arm_probe1k_v1.sh"
    ).read_text(encoding="utf-8")

    assert '[[ -z "$(git status --porcelain)" ]]' in runbook
    assert "--untracked-files=no" not in runbook
    assert "nvidia-smi --query-compute-apps=pid" in runbook
    assert 'test ! -e "$OUTPUT_ROOT"' in runbook
    assert "verify_generation_conditioning_ranking_probe_execution_authorization.py" in runbook
    assert "EXPECTED_STANDING_AUTHORIZATION_SHA256" in runbook
    assert "generation_conditioning_ranking_four_arm_posteval_v1.sh" in runbook
    assert "EXECUTION_APPROVAL" not in runbook


def _identity(name: str, character: str) -> dict:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 100 + len(name),
        "sha256": character * 64,
    }


def _standing_authorization() -> dict:
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_TEXT,
            "received_at": "2026-08-13T00:11:00+08:00",
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
        },
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def _preparation(*, revision: str, branch: str, output_root: str) -> dict:
    configs = _configs()
    report = conditioning_ranking_probe_contract(**configs)
    return {
        **report,
        "git": {
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
        "output_root": output_root,
        "configs": {},
        "parameter_counts": {},
        "claim_boundary": {},
    }


def _followup(quality_identity: dict) -> dict:
    failed_checks = [
        "cofitok_absolute_fid",
        "cofitok_recall_floor",
        "class_fidelity",
    ]
    checks = [
        {"name": name, "passed": name not in failed_checks}
        for name in (
            "cofitok_absolute_fid",
            "matched_fid_tolerance",
            "cofitok_precision_floor",
            "cofitok_recall_floor",
            "matched_precision_tolerance",
            "matched_recall_tolerance",
            "matched_endpoint_tolerance",
            "ordered_prefix_rank",
            "coarse_token_utilization",
            "restricted_synthesis_zero_token",
            "shuffle_mismatch",
            "class_fidelity",
        )
    ]
    return {
        "schema_version": 1,
        "status": "completed",
        "role": "stability_quality_bridge_followup_experiment_decision",
        "quality_bridge_execution_git": {
            "revision": "cf0e5faa94bf4ab38d947b921935b3b765b5537a",
            "branch": "scale/generation-stability-quality-bridge-100k",
            "tracked_dirty": False,
        },
        "source_reports": {"quality_bridge_result": quality_identity},
        "terminal_quality": {
            "failed_checks": failed_checks,
            "checks": checks,
        },
        "recommended_next_stage": {
            "id": "run_class_conditioning_fidelity_diagnostic",
            "category": "class_conditioning_recovery",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def _terminal_guard(quality_identity: dict, failed_checks: list[str]) -> dict:
    return {
        "schema_version": 1,
        "role": "generation_terminal_system_claim_guard",
        "status": "hold",
        "sources": {"quality_bridge_result": quality_identity},
        "evidence": {
            "quality_screen": {"failed_checks": failed_checks},
            "requested_class_visual_audit": {
                "status": "completed",
                "quantitative_quality_evidence": False,
                "panel_count": 2,
            },
        },
        "claim_policy": {
            "terminal_system_evidence_complete": True,
            "requested_class_visual_evidence_available": True,
            "requested_class_visual_evidence_is_quantitative": False,
            "larger_training_launch_allowed": False,
            "inference_export_authorization_allowed": False,
            "release_authorization_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
        },
        "claim_boundary": {
            "training_launch_allowed": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def test_standing_authorization_is_source_bound_to_terminal_class_failure() -> None:
    revision = "1" * 40
    branch = "scale/generation-label-ranking-standing-authorization-v1"
    output_root = (
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "conditioning_ranking_four_arm_probe1k_v1"
    )
    quality_identity = _identity("quality", "a")
    followup = _followup(quality_identity)
    terminal = _terminal_guard(
        quality_identity,
        followup["terminal_quality"]["failed_checks"],
    )
    kwargs = {
        "preparation": _preparation(
            revision=revision,
            branch=branch,
            output_root=output_root,
        ),
        "preparation_identity": _identity("preparation", "b"),
        "standing_authorization": _standing_authorization(),
        "standing_authorization_identity": _identity("standing", "c"),
        "followup_decision": followup,
        "followup_decision_identity": _identity("followup", "d"),
        "terminal_system_guard": terminal,
        "terminal_system_guard_identity": _identity("terminal", "e"),
        "authorization_git": {
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
        "expected_revision": revision,
        "expected_branch": branch,
        "expected_output_root": output_root,
    }
    report = build_conditioning_ranking_probe_execution_authorization(**kwargs)

    assert report["role"] == EXECUTION_AUTHORIZATION_ROLE
    assert report["authorization_mode"] == EXECUTION_AUTHORIZATION_MODE
    assert report["stage"] == PROBE_STAGE
    assert report["source_decision"]["requested_class_visual_evidence_complete"] is True
    assert report["authorization_boundary"] == EXECUTION_BOUNDARY
    assert report["claim_boundary"]["full_300k_launch_allowed"] is False
    assert (
        validate_conditioning_ranking_probe_execution_authorization(
            report,
            **kwargs,
        )
        == report
    )


def test_standing_authorization_rejects_missing_visual_or_wrong_route() -> None:
    revision = "1" * 40
    branch = "scale/generation-label-ranking-standing-authorization-v1"
    output_root = "/output/conditioning_ranking_four_arm_probe1k_v1"
    quality_identity = _identity("quality", "a")
    followup = _followup(quality_identity)
    terminal = _terminal_guard(
        quality_identity,
        followup["terminal_quality"]["failed_checks"],
    )
    kwargs = {
        "preparation": _preparation(
            revision=revision,
            branch=branch,
            output_root=output_root,
        ),
        "preparation_identity": _identity("preparation", "b"),
        "standing_authorization": _standing_authorization(),
        "standing_authorization_identity": _identity("standing", "c"),
        "followup_decision": followup,
        "followup_decision_identity": _identity("followup", "d"),
        "terminal_system_guard": terminal,
        "terminal_system_guard_identity": _identity("terminal", "e"),
        "authorization_git": {
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
        "expected_revision": revision,
        "expected_branch": branch,
        "expected_output_root": output_root,
    }
    kwargs["terminal_system_guard"] = copy.deepcopy(terminal)
    kwargs["terminal_system_guard"]["evidence"][
        "requested_class_visual_audit"
    ]["status"] = "waiting"
    with pytest.raises(ValueError, match="requested-class"):
        build_conditioning_ranking_probe_execution_authorization(**kwargs)

    kwargs["terminal_system_guard"] = terminal
    kwargs["followup_decision"] = copy.deepcopy(followup)
    kwargs["followup_decision"]["recommended_next_stage"]["id"] = (
        "prepare_matched_250m_capacity_qualification_probe"
    )
    with pytest.raises(ValueError, match="follow-up"):
        build_conditioning_ranking_probe_execution_authorization(**kwargs)
