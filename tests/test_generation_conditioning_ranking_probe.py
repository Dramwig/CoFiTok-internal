from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    EXECUTION_BOUNDARY,
    EXECUTION_AUTHORIZATION_MODE,
    EXECUTION_AUTHORIZATION_ROLE,
    FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_BUILDER_GIT,
    QUALITY_BRIDGE_EXECUTION_GIT,
    QUALITY_BRIDGE_OUTPUT_ROOT,
    PROBE_AUTHORIZATION_TEXT,
    PROBE_ROLE,
    PROBE_SCOPE,
    PROBE_STAGE,
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
    REQUESTED_CLASS_VISUAL_AUDIT_CLAIM_BOUNDARY,
    REQUESTED_CLASS_VISUAL_AUDIT_ROLE,
    REQUESTED_CLASS_VISUAL_AUDIT_WAITER_AUTHORIZATION_BOUNDARY,
    REQUESTED_CLASS_VISUAL_AUDIT_WAITER_DETAIL,
    REQUESTED_CLASS_VISUAL_AUDIT_WAITER_GIT,
    REQUESTED_CLASS_VISUAL_AUDIT_WAITER_ROLE,
    TERMINAL_SYSTEM_GUARD_CLAIM_BOUNDARY,
    TERMINAL_SYSTEM_GUARD_WAITER_AUTHORIZATION_BOUNDARY,
    TERMINAL_SYSTEM_GUARD_WAITER_DETAIL,
    TERMINAL_SYSTEM_GUARD_WAITER_GIT,
    TERMINAL_SYSTEM_GUARD_WAITER_ROLE,
    build_conditioning_ranking_probe_execution_authorization,
    conditioning_ranking_probe_contract,
    validate_class_conditioning_followup_decision,
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


CHECK_NAMES = (
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


def _checks(failed_checks: list[str]) -> list[dict]:
    return [
        {"name": name, "passed": name not in failed_checks}
        for name in CHECK_NAMES
    ]


def _quality_result(*additional_failed_checks: str) -> dict:
    failed_checks = [*additional_failed_checks, "class_fidelity"]
    return {
        "schema_version": 1,
        "status": "completed",
        "role": "stability_full_data_quality_bridge_result",
        "stage": "stability_quality_bridge",
        "git": copy.deepcopy(QUALITY_BRIDGE_EXECUTION_GIT),
        "quality_screen": {
            "status": "hold",
            "failed_checks": failed_checks,
            "checks": _checks(failed_checks),
        },
    }


def _followup(
    quality_identity: dict,
    *additional_failed_checks: str,
) -> dict:
    failed_checks = [*additional_failed_checks, "class_fidelity"]
    return {
        "schema_version": 2,
        "status": "completed",
        "role": "stability_quality_bridge_followup_experiment_decision",
        "decision_builder_git": copy.deepcopy(FOLLOWUP_DECISION_BUILDER_GIT),
        "quality_bridge_execution_git": copy.deepcopy(QUALITY_BRIDGE_EXECUTION_GIT),
        "source_reports": {
            "quality_bridge_result": quality_identity,
            "milestones": {
                "50000": _identity("milestone_50000", "1"),
                "100000": _identity("milestone_100000", "2"),
            },
            "terminal_training_exposure": _identity("exposure", "3"),
        },
        "terminal_quality": {
            "status": "hold",
            "failed_checks": failed_checks,
            "checks": _checks(failed_checks),
        },
        "recommended_next_stage": {
            "id": "run_class_conditioning_fidelity_diagnostic",
            "category": "class_conditioning_recovery",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
            "trigger": {"failed_checks": failed_checks},
        },
        "claim_policy": {
            "experiment_selection_only": True,
            "terminal_result_is_promotion_gate": False,
        },
        "authorization_boundary": copy.deepcopy(FOLLOWUP_AUTHORIZATION_BOUNDARY),
    }


@pytest.mark.parametrize(
    "additional_failure",
    (
        "matched_fid_tolerance",
        "cofitok_absolute_fid",
        "ordered_prefix_rank",
    ),
)
def test_class_conditioning_validator_rejects_mixed_quality_failure(
    additional_failure: str,
) -> None:
    quality_identity = _identity("quality", "a")
    assert validate_class_conditioning_followup_decision(
        _followup(quality_identity),
        expected_quality_bridge_result=quality_identity,
    )

    with pytest.raises(ValueError, match="class-conditioning follow-up"):
        validate_class_conditioning_followup_decision(
            _followup(quality_identity, additional_failure),
            expected_quality_bridge_result=quality_identity,
        )


def test_class_conditioning_validator_rejects_wrong_decision_builder_git() -> None:
    quality_identity = _identity("quality", "a")
    followup = _followup(quality_identity)
    followup["decision_builder_git"] = {
        **FOLLOWUP_DECISION_BUILDER_GIT,
        "revision": "0" * 40,
    }
    with pytest.raises(ValueError, match="class-conditioning follow-up"):
        validate_class_conditioning_followup_decision(followup)


def _visual_report() -> dict:
    panels = []
    for panel_index in range(2):
        panels.append(
            {
                **_identity(f"visual_panel_{panel_index}", str(4 + panel_index)),
                "indices": list(range(panel_index * 8, (panel_index + 1) * 8)),
                "row_order": ["real_validation", "cofitok", "dense_identity"],
                "columns": 8,
            }
        )
    return {
        "schema_version": 1,
        "role": REQUESTED_CLASS_VISUAL_AUDIT_ROLE,
        "status": "completed",
        "claim_boundary": copy.deepcopy(REQUESTED_CLASS_VISUAL_AUDIT_CLAIM_BOUNDARY),
        "indices": list(range(16)),
        "panels": panels,
        "sources": {
            "cofitok": {"sample_set": {}},
            "dense_identity": {"sample_set": {}},
        },
    }


def _visual_status(
    quality_identity: dict,
    visual_report_identity: dict,
) -> dict:
    return {
        "schema_version": 1,
        "role": REQUESTED_CLASS_VISUAL_AUDIT_WAITER_ROLE,
        "status": "completed",
        "detail": REQUESTED_CLASS_VISUAL_AUDIT_WAITER_DETAIL,
        "authorization_boundary": copy.deepcopy(
            REQUESTED_CLASS_VISUAL_AUDIT_WAITER_AUTHORIZATION_BOUNDARY
        ),
        "expected": {
            "git": copy.deepcopy(REQUESTED_CLASS_VISUAL_AUDIT_WAITER_GIT),
            "quality_result": quality_identity["path"],
        },
        "git": copy.deepcopy(REQUESTED_CLASS_VISUAL_AUDIT_WAITER_GIT),
        "quality_result": {
            "identity": quality_identity,
            "quality_screen": {},
        },
        "visual_audit": visual_report_identity,
    }


def _terminal_guard(
    quality_identity: dict,
    failed_checks: list[str],
    visual_status_identity: dict,
    visual_report_identity: dict,
) -> dict:
    decision = "terminal_system_evidence_complete_without_qualified_matched_advantage"
    return {
        "schema_version": 1,
        "role": "generation_terminal_system_claim_guard",
        "status": "hold",
        "decision": decision,
        "scope": {
            "dataset": "imagenet_256",
            "training_steps_per_method": 100_000,
            "quality_output_root": QUALITY_BRIDGE_OUTPUT_ROOT,
            "training_git": copy.deepcopy(QUALITY_BRIDGE_EXECUTION_GIT),
        },
        "sources": {
            "quality_bridge_result": quality_identity,
            "statistical_claim_language_guard": _identity("statistical", "6"),
            "requested_class_visual_audit_waiter_status": visual_status_identity,
            "runtime_compute_claim_guard": _identity("runtime", "7"),
        },
        "evidence": {
            "quality_screen": {
                "status": "hold",
                "absolute_quality_passed": True,
                "failed_checks": failed_checks,
                "check_count": len(CHECK_NAMES),
            },
            "requested_class_visual_audit": {
                "status": "completed",
                "quantitative_quality_evidence": False,
                "fixed_indices": list(range(16)),
                "panel_count": 2,
                "report": visual_report_identity,
                "waiter_status": visual_status_identity,
            },
            "runtime_compute": {"status": "verified"},
            "class_fidelity_classifier_integrity": {"status": "verified"},
        },
        "claim_policy": {
            "terminal_system_evidence_complete": True,
            "requested_class_visual_evidence_available": True,
            "requested_class_visual_evidence_is_quantitative": False,
            "class_fidelity_classifier_physical_integrity_verified": True,
            "larger_training_launch_allowed": False,
            "inference_export_authorization_allowed": False,
            "release_authorization_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
        },
        "claim_boundary": copy.deepcopy(TERMINAL_SYSTEM_GUARD_CLAIM_BOUNDARY),
    }


def _terminal_status(
    quality_identity: dict,
    visual_status_identity: dict,
    terminal_identity: dict,
    terminal: dict,
) -> dict:
    return {
        "schema_version": 1,
        "role": TERMINAL_SYSTEM_GUARD_WAITER_ROLE,
        "status": "completed",
        "detail": TERMINAL_SYSTEM_GUARD_WAITER_DETAIL,
        "authorization_boundary": copy.deepcopy(
            TERMINAL_SYSTEM_GUARD_WAITER_AUTHORIZATION_BOUNDARY
        ),
        "expected": {
            "git": copy.deepcopy(TERMINAL_SYSTEM_GUARD_WAITER_GIT),
            "quality_output_root": QUALITY_BRIDGE_OUTPUT_ROOT,
            "output": terminal_identity["path"],
        },
        "git": copy.deepcopy(TERMINAL_SYSTEM_GUARD_WAITER_GIT),
        "sources": {
            "quality_result": quality_identity,
            "visual_audit_waiter_status": visual_status_identity,
        },
        "guard": terminal_identity,
        "guard_status": terminal["status"],
        "guard_decision": terminal["decision"],
    }


def _authorization_kwargs() -> dict:
    revision = "1" * 40
    tree = "2" * 40
    branch = "analysis/generation-conditioning-exact-route-rebind-v2-20260822"
    output_root = "/output/conditioning_ranking_four_arm_probe1k_v1"
    quality_identity = _identity("quality", "a")
    quality = _quality_result()
    followup = _followup(quality_identity)
    visual_report_identity = _identity("visual_report", "8")
    visual_report = _visual_report()
    visual_status_identity = _identity("visual_status", "9")
    visual_status = _visual_status(quality_identity, visual_report_identity)
    terminal_identity = _identity("terminal", "e")
    terminal = _terminal_guard(
        quality_identity,
        followup["terminal_quality"]["failed_checks"],
        visual_status_identity,
        visual_report_identity,
    )
    terminal_status_identity = _identity("terminal_status", "f")
    terminal_status = _terminal_status(
        quality_identity,
        visual_status_identity,
        terminal_identity,
        terminal,
    )
    return {
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
        "quality_bridge_result": quality,
        "quality_bridge_result_identity": quality_identity,
        "terminal_system_guard": terminal,
        "terminal_system_guard_identity": terminal_identity,
        "terminal_system_guard_status": terminal_status,
        "terminal_system_guard_status_identity": terminal_status_identity,
        "requested_class_visual_audit_status": visual_status,
        "requested_class_visual_audit_status_identity": visual_status_identity,
        "requested_class_visual_audit_report": visual_report,
        "requested_class_visual_audit_report_identity": visual_report_identity,
        "authorization_git": {
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
        "authorization_tree": tree,
        "expected_revision": revision,
        "expected_tree": tree,
        "expected_branch": branch,
        "expected_output_root": output_root,
    }


def test_standing_authorization_is_source_bound_to_terminal_class_failure() -> None:
    kwargs = _authorization_kwargs()
    report = build_conditioning_ranking_probe_execution_authorization(**kwargs)

    assert report["role"] == EXECUTION_AUTHORIZATION_ROLE
    assert report["authorization_mode"] == EXECUTION_AUTHORIZATION_MODE
    assert report["stage"] == PROBE_STAGE
    assert report["source_decision"]["requested_class_visual_evidence_complete"] is True
    assert report["source_decision"]["decision_builder_git"] == (
        FOLLOWUP_DECISION_BUILDER_GIT
    )
    assert report["authorization_boundary"] == EXECUTION_BOUNDARY
    assert report["claim_boundary"]["full_300k_launch_allowed"] is False
    assert (
        validate_conditioning_ranking_probe_execution_authorization(
            report,
            **kwargs,
        )
        == report
    )


def test_standing_authorization_rejects_missing_or_mismatched_visual_audit() -> None:
    kwargs = _authorization_kwargs()
    kwargs["requested_class_visual_audit_status"] = copy.deepcopy(
        kwargs["requested_class_visual_audit_status"]
    )
    kwargs["requested_class_visual_audit_status"]["visual_audit"] = _identity(
        "another_visual_report",
        "0",
    )
    with pytest.raises(ValueError, match="visual-audit waiter binding"):
        build_conditioning_ranking_probe_execution_authorization(**kwargs)


def test_standing_authorization_rejects_wrong_terminal_guard_git() -> None:
    kwargs = _authorization_kwargs()
    kwargs["terminal_system_guard_status"] = copy.deepcopy(
        kwargs["terminal_system_guard_status"]
    )
    kwargs["terminal_system_guard_status"]["git"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="guard waiter binding"):
        build_conditioning_ranking_probe_execution_authorization(**kwargs)


def test_standing_authorization_requires_classifier_physical_integrity() -> None:
    kwargs = _authorization_kwargs()
    kwargs["terminal_system_guard"] = copy.deepcopy(
        kwargs["terminal_system_guard"]
    )
    kwargs["terminal_system_guard"]["claim_policy"].pop(
        "class_fidelity_classifier_physical_integrity_verified"
    )
    with pytest.raises(ValueError, match="terminal requested-class evidence differs"):
        build_conditioning_ranking_probe_execution_authorization(**kwargs)

    kwargs = _authorization_kwargs()
    kwargs["terminal_system_guard"] = copy.deepcopy(
        kwargs["terminal_system_guard"]
    )
    kwargs["terminal_system_guard"]["evidence"][
        "class_fidelity_classifier_integrity"
    ]["status"] = "unverified"
    with pytest.raises(ValueError, match="terminal requested-class evidence differs"):
        build_conditioning_ranking_probe_execution_authorization(**kwargs)


@pytest.mark.parametrize(
    "additional_failure",
    (
        "matched_fid_tolerance",
        "cofitok_absolute_fid",
        "ordered_prefix_rank",
    ),
)
def test_standing_authorization_rejects_non_class_only_routes(
    additional_failure: str,
) -> None:
    kwargs = _authorization_kwargs()
    kwargs["quality_bridge_result"] = _quality_result(additional_failure)
    kwargs["followup_decision"] = _followup(
        kwargs["quality_bridge_result_identity"],
        additional_failure,
    )
    with pytest.raises(ValueError, match="class-only|class-conditioning"):
        build_conditioning_ranking_probe_execution_authorization(**kwargs)
