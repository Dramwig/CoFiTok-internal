from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
    conditioning_ranking_probe_contract,
)
from cofitok.generation.conditioning_ranking_terminal_rebind import (
    AUTHORIZATION_ROLE,
    CLAIM_BOUNDARY,
    EPSILON_INPUT_NAMES,
    EPSILON_RESULT_AUTHORIZATION_BOUNDARY,
    EPSILON_RESULT_CLAIM_BOUNDARY,
    EXECUTION_BOUNDARY,
    EXPECTED_FAILED_CHECKS,
    EXPECTED_SOURCE_SHA256,
    GAIN_CLAIM_BOUNDARY,
    LEGACY_OUTPUT_ROOT,
    LEGACY_PREPARATION_GIT,
    OFFICIAL_EPSILON_VALIDATOR_GIT,
    OFFICIAL_EPSILON_VALIDATOR_SHA256,
    OUTPUT_ROOT,
    POST_RECONCILIATION_AUTHORIZATION_BOUNDARY,
    PREPARATION_CLAIM_BOUNDARY,
    RECOVERY_AUTHORIZED_ACTIONS,
    STAGE,
    SUPERSESSION_SCOPE,
    build_terminal_rebind_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
TREE = "b" * 40
BRANCH = "analysis/generation-conditioning-ranking-terminal-rebind-v2-20260824"


def _identity(name: str, *, character: str = "c", size: int = 10) -> dict:
    digest = EXPECTED_SOURCE_SHA256.get(name, character * 64)
    return {"path": f"/evidence/{name}.json", "bytes": size, "sha256": digest}


def _standing() -> dict:
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_TEXT,
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
            "received_at": "2026-08-18T00:00:00+00:00",
        },
        "preserved_safety_boundaries": STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    }


def _terminal_sources() -> tuple[dict, dict, dict]:
    quality = {
        "schema_version": 1,
        "role": "stability_full_data_quality_bridge_result",
        "status": "completed",
        "quality_screen": {
            "status": "hold",
            "failed_checks": list(EXPECTED_FAILED_CHECKS),
        },
    }
    failure = {
        "class_only_failure": False,
        "factorization_mechanism_failure": False,
        "matched_quality_only_failure": False,
        "mixed_shared_absolute_quality_support_and_class_failure": True,
    }
    terminal = {
        "status": "hold",
        "failed_checks": list(EXPECTED_FAILED_CHECKS),
        "both_methods_absolute_fid_above_threshold": True,
        "both_methods_recall_below_floor": True,
        "class_fidelity_pass": False,
    }
    resolution = {
        "failure_classification": failure,
        "terminal_quality": terminal,
    }
    next_stage = {
        "id": "prepare_matched_100k_epsilon_stability_sampling_diagnostic",
        "execution_ready": False,
        "gpu_execution_allowed": False,
    }
    source_replay = {
        "terminal_metrics_and_class_fidelity_replayed": True,
        "cross_protocol_reconciliation_replayed": True,
    }
    decision = {
        "schema_version": 1,
        "role": "generation_100k_post_reconciliation_experiment_decision",
        "status": "completed",
        "operational_status": "pass",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "authorization_boundary": POST_RECONCILIATION_AUTHORIZATION_BOUNDARY,
        "scientific_resolution": resolution,
        "legacy_route_disposition": {
            "conditioning_only_supervisor": {"eligible": False},
            "factorization_quality_regression_supervisor": {"eligible": False},
        },
        "source_replay": source_replay,
        "recommended_next_stage": next_stage,
        "source_evidence": {
            "quality_bridge_result": _identity("quality_bridge_result")
        },
    }
    verification = {
        "schema_version": 1,
        "role": "generation_100k_post_reconciliation_decision_verification",
        "status": "verified",
        "decision": _identity("post_reconciliation_decision"),
        "authorization_boundary": POST_RECONCILIATION_AUTHORIZATION_BOUNDARY,
        "scientific_resolution": resolution,
        "recommended_next_stage": next_stage,
        "source_replay": source_replay,
    }
    return quality, decision, verification


def _epsilon_sources() -> tuple[dict, dict, dict]:
    decision_id = _identity("post_reconciliation_decision")
    verification_id = _identity("post_reconciliation_verification")
    input_graph = {
        name: _identity(f"epsilon_{name}", character=str(index + 1))
        for index, name in enumerate(EPSILON_INPUT_NAMES)
    }
    result = {
        "status": "pass",
        "scientific_status": "screening_only",
        "selection_status": "no_shared_sampling_recovery_candidate",
        "selected_case_id": None,
        "generation_advantage_proven": False,
        "authorization_boundary": EPSILON_RESULT_AUTHORIZATION_BOUNDARY,
        "claim_boundary": EPSILON_RESULT_CLAIM_BOUNDARY,
        "candidates": [
            {"case_id": f"case_{index}", "passes_shared_recovery_screen": False}
            for index in range(7)
        ],
        "observations": [{"arm": index} for index in range(16)],
        "source_bindings": {
            **input_graph,
            "terminal_route_receipt": decision_id,
        },
        "execution": {
            "status": "approved",
            "scope": "matched_1000_sample_epsilon_stability_sampling_diagnostic_only",
            "output_root": (
                "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
                "stability_full_data_100k_epsilon_stability_sampling_recovery_v1"
            ),
            "output_root_non_overlapping": True,
            "git": OFFICIAL_EPSILON_VALIDATOR_GIT,
            "source_bindings": {
                "post_reconciliation_verification": verification_id,
            },
        },
    }
    result_id = _identity("epsilon_stability_result")
    recovery = {
        "schema_version": 1,
        "role": "generation_epsilon_stability_pre_gpu_recovery_controller",
        "status": "completed",
        "stage": "completed",
        "detail": "matched_1000_sample_sampling_recovery_diagnostic_completed",
        "completed_arms": 16,
        "total_arms": 16,
        "child_pid": None,
        "case_id": None,
        "method": None,
        "generation_advantage_proven": False,
        "authorized_actions": RECOVERY_AUTHORIZED_ACTIONS,
        "result_authorization_boundary": EPSILON_RESULT_AUTHORIZATION_BOUNDARY,
        "result": result_id,
    }
    replay = {
        "status": "verified",
        "checkout_git": OFFICIAL_EPSILON_VALIDATOR_GIT,
        "validator": {
            "path": "/official/validate_result.py",
            "bytes": 100,
            "sha256": OFFICIAL_EPSILON_VALIDATOR_SHA256,
        },
        "result": result_id,
        "input_graph": input_graph,
        "cuda_visible_devices": "-1",
        "returncode": 0,
    }
    return result, recovery, replay


def _gain() -> tuple[dict, dict]:
    sources = {
        "cofitok_k8": _identity("gain_cofitok", character="d"),
        "dense_identity": _identity("gain_dense", character="e"),
    }
    report = {
        "schema_version": 1,
        "role": "generation_conditioning_gain_sweep_comparison",
        "status": "completed",
        "claim_boundary": GAIN_CLAIM_BOUNDARY,
        "diagnostic_interpretation": {
            "shared_inference_gain_recovery_supported": False,
            "gain_only_amplifies_without_semantic_recovery": True,
            "recommended_next_action": (
                "develop_matched_training_time_label_ranking_or_contrastive_denoising_loss"
            ),
        },
        "sources": sources,
    }
    return report, sources


def _supersession() -> tuple[dict, dict]:
    marker_id = _identity("supersession_marker")
    receipt_id = _identity("supersession_receipt")
    receipt = {
        "schema_version": 1,
        "role": "generation_terminal_route_supersession_interlock",
        "status": "pass",
        "generation_advantage_proven": False,
        "scope": SUPERSESSION_SCOPE,
        "interlocks": {
            "conditioning_ranking_v1": {
                "active": True,
                "legacy_output_root": LEGACY_OUTPUT_ROOT,
                "interlock_kind": "directory_with_marker",
                "marker": marker_id,
            }
        },
        "guarantees": {
            "corrected_route_must_use_new_versioned_output_roots": True,
            "legacy_conditioning_child_launch_blocked_by_preexisting_lock_directory": True,
            "no_existing_process_was_signaled": True,
            "no_gpu_process_was_started": True,
        },
    }
    marker = {
        "schema_version": 1,
        "role": "generation_legacy_gpu_route_supersession_marker",
        "status": "active",
        "route": "conditioning_ranking_v1",
        "legacy_output_root": LEGACY_OUTPUT_ROOT,
        "reason": "legacy_consumer_binds_stale_runtime_v1_terminal_guard",
        "scope": SUPERSESSION_SCOPE,
        "canonical_receipt": receipt_id["path"],
    }
    return receipt, marker


def _preparations() -> tuple[dict, dict]:
    paths = {
        "control_cofitok": ROOT
        / "configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe1k.json",
        "control_dense": ROOT
        / "configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe1k.json",
        "ranked_cofitok": ROOT
        / "configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classrank_k8_probe1k.json",
        "ranked_dense": ROOT
        / "configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classrank_dense_probe1k.json",
    }
    configs = {name: config_to_dict(load_config(path)) for name, path in paths.items()}
    contract = conditioning_ranking_probe_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense=configs["ranked_dense"],
    )
    assert contract["valid"]
    descriptors = {
        name: {"path": path.as_posix(), "bytes": 100 + index, "sha256": str(index + 1) * 64}
        for index, (name, path) in enumerate(paths.items())
    }
    counts = {
        "control_cofitok": 62_834_083,
        "control_dense": 62_824_707,
        "ranked_cofitok": 62_834_083,
        "ranked_dense": 62_824_707,
    }
    legacy = {
        **contract,
        "git": {
            "revision": LEGACY_PREPARATION_GIT["revision"],
            "branch": LEGACY_PREPARATION_GIT["branch"],
            "tracked_dirty": False,
        },
        "output_root": LEGACY_OUTPUT_ROOT,
        "configs": descriptors,
        "parameter_counts": counts,
    }
    current = {
        **contract,
        "git": {"revision": REVISION, "branch": BRANCH, "tracked_dirty": False},
        "tree": TREE,
        "output_root": OUTPUT_ROOT,
        "configs": copy.deepcopy(descriptors),
        "parameter_counts": copy.deepcopy(counts),
        "legacy_candidate_source": _identity("legacy_preparation"),
        "claim_boundary": PREPARATION_CLAIM_BOUNDARY,
    }
    return current, legacy


def _arguments() -> dict:
    preparation, legacy = _preparations()
    quality, decision, verification = _terminal_sources()
    epsilon, recovery, replay = _epsilon_sources()
    gain, gain_sources = _gain()
    receipt, marker = _supersession()
    return {
        "preparation": preparation,
        "preparation_identity": _identity("new_preparation", character="f"),
        "legacy_preparation": legacy,
        "legacy_preparation_identity": _identity("legacy_preparation"),
        "standing_authorization": _standing(),
        "standing_authorization_identity": _identity("standing_authorization"),
        "quality_result": quality,
        "quality_result_identity": _identity("quality_bridge_result"),
        "post_decision": decision,
        "post_decision_identity": _identity("post_reconciliation_decision"),
        "post_verification": verification,
        "post_verification_identity": _identity("post_reconciliation_verification"),
        "epsilon_result": epsilon,
        "epsilon_result_identity": _identity("epsilon_stability_result"),
        "epsilon_recovery_status": recovery,
        "epsilon_recovery_status_identity": _identity("epsilon_recovery_status"),
        "gain_report": gain,
        "gain_report_identity": _identity("conditioning_gain_comparison"),
        "gain_replayed_sources": gain_sources,
        "supersession_receipt": receipt,
        "supersession_receipt_identity": _identity("supersession_receipt"),
        "supersession_marker": marker,
        "supersession_marker_identity": _identity("supersession_marker"),
        "official_epsilon_replay": replay,
        "runbook_identity": _identity("runbook", character="9"),
        "authorization_git": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "expected_revision": REVISION,
        "expected_tree": TREE,
        "expected_branch": BRANCH,
        "expected_output_root": OUTPUT_ROOT,
        "control_plane": {"test": True},
    }


def test_terminal_rebind_authorizes_only_bounded_semantic_probe() -> None:
    report = build_terminal_rebind_authorization(**_arguments())

    assert report["role"] == AUTHORIZATION_ROLE
    assert report["status"] == "authorized"
    assert report["stage"] == STAGE
    assert report["execution_boundary"] == EXECUTION_BOUNDARY
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["scientific_route"]["sampling_recovery_selection_status"] == (
        "no_shared_sampling_recovery_candidate"
    )
    assert report["scientific_route"]["absolute_fid_or_recall_recovery_tested"] is False
    assert report["generation_advantage_proven"] is False


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda args: args["post_decision"]["scientific_resolution"][
                "failure_classification"
            ].__setitem__("class_only_failure", True),
            "mixed-failure",
        ),
        (
            lambda args: args["epsilon_result"].__setitem__("selected_case_id", "case_0"),
            "no-candidate",
        ),
        (
            lambda args: args["gain_report"]["diagnostic_interpretation"].__setitem__(
                "shared_inference_gain_recovery_supported", True
            ),
            "gain diagnostic",
        ),
        (
            lambda args: args["supersession_marker"].__setitem__("status", "inactive"),
            "marker",
        ),
        (
            lambda args: args.__setitem__("expected_output_root", "/tmp/arbitrary"),
            "output root",
        ),
    ],
)
def test_terminal_rebind_rejects_route_or_scope_drift(mutator, match: str) -> None:
    arguments = _arguments()
    mutator(arguments)

    with pytest.raises(ValueError, match=match):
        build_terminal_rebind_authorization(**arguments)
