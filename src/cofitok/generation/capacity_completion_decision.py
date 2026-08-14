from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_CONFIGURED_STEPS,
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_PARAMETER_COUNTS,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)
from cofitok.generation.capacity_scaling_decision import (
    CAPACITY_SCALING_METHODS,
    CAPACITY_SCALING_TARGET_STEP,
)
from cofitok.generation.capacity_scaling_result import (
    CAPACITY_COMPLETION_MECHANISM_HOLD_ID,
    CAPACITY_COMPLETION_QUALITY_HOLD_ID,
    CAPACITY_SCALING_RESULT_BOUNDARY,
    validate_capacity_scaling_50k_result,
)


CAPACITY_COMPLETION_DECISION_SCHEMA_VERSION = 1
CAPACITY_COMPLETION_DECISION_ROLE = (
    "stability_full_data_capacity_completion_100k_decision"
)
CAPACITY_COMPLETION_TARGET_STEP = CAPACITY_PROBE_CONFIGURED_STEPS
CAPACITY_COMPLETION_RECOMMENDATION_ID = (
    "complete_matched_250m_capacity_bridge_to_100000"
)
CAPACITY_COMPLETION_DECISION_BOUNDARY = {
    "decision_is_source_replayed": True,
    "maximum_scope_is_matched_250m_step_50000_to_100000": True,
    "fresh_training_allowed": False,
    "resume_from_exact_step_50000_required": True,
    "stop_at_step_100000_required": True,
    "terminal_10000_sample_evaluation_required": True,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
}


def _absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or Path(value).is_absolute()


def _hex(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _hex(digest, length=64)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(value: Mapping[str, Any], *, with_tree: bool, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    result: dict[str, Any] = {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }
    if with_tree:
        result["tree"] = value.get("tree")
    if (
        not _hex(revision, length=40)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
        or (with_tree and not _hex(value.get("tree"), length=40))
        or dict(value) != result
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return result


def _terminal_evaluation_contract() -> dict[str, Any]:
    return {
        "sample_count_per_method": 10_000,
        "weights": "ema",
        "sampler": "ddim",
        "sample_steps": 100,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "precision": "bf16",
        "fixed_random_stream_across_methods": True,
        "precision_recall_required": True,
        "class_fidelity_required": True,
        "cofitok_mechanism_images": 256,
        "mechanism_timestep": 500,
    }


def build_capacity_completion_decision(
    *,
    capacity_scaling_result: Mapping[str, Any],
    capacity_scaling_result_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    decision_git: Mapping[str, Any],
    expected_execution_revision: str,
    expected_execution_tree: str,
    expected_execution_branch: str,
    expected_result_revision: str,
    expected_result_tree: str,
    expected_result_branch: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    result_identity = _identity(
        capacity_scaling_result_identity,
        label="capacity scaling 50K result",
    )
    standing_identity = _identity(
        standing_authorization_identity,
        label="standing experiment authorization",
    )
    standing = validate_standing_experiment_authorization(standing_authorization)
    if standing["preserved_safety_boundaries"] != (
        STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing experiment safety boundary differs")
    result = validate_capacity_scaling_50k_result(
        capacity_scaling_result,
        expected_execution_revision=expected_execution_revision,
        expected_execution_tree=expected_execution_tree,
        expected_execution_branch=expected_execution_branch,
        expected_result_revision=expected_result_revision,
        expected_result_tree=expected_result_tree,
        expected_result_branch=expected_result_branch,
    )
    if capacity_scaling_result.get("authorization_boundary") != (
        CAPACITY_SCALING_RESULT_BOUNDARY
    ):
        raise ValueError("capacity completion source boundary differs")
    training_git = _git(
        capacity_scaling_result.get("training_git", {}),
        with_tree=True,
        label="capacity completion training",
    )
    expected_training_git = {
        "revision": expected_training_revision,
        "tree": expected_training_tree,
        "branch": expected_training_branch,
        "tracked_dirty": False,
    }
    if training_git != expected_training_git:
        raise ValueError("capacity completion training Git identity differs")
    decision_provenance = _git(
        decision_git,
        with_tree=True,
        label="capacity completion decision",
    )
    supported = result["capacity_completion_supported"]
    mechanism_valid = result["cofitok_mechanism_invariants_valid"]
    if supported:
        recommendation = {
            "id": CAPACITY_COMPLETION_RECOMMENDATION_ID,
            "category": "capacity_completion_qualification",
            "objective": (
                "Resume the exact matched 250M step-50K pair to the configured "
                "step-100K horizon and produce terminal source-bound evidence."
            ),
            "execution_ready": True,
            "gpu_execution_allowed": True,
            "training_launch_allowed": True,
            "required_next_evidence": (
                "Exact step-100K training validation, a matched 2,048-sample "
                "DDIM-50 milestone, matched 10,000-sample EMA DDIM-100 quality "
                "and class-fidelity reports, and a 256-image mechanism audit."
            ),
        }
    elif not mechanism_valid:
        recommendation = {
            "id": CAPACITY_COMPLETION_MECHANISM_HOLD_ID,
            "category": "factorization_mechanism_recovery",
            "objective": (
                "Hold 250M completion and restore the failed ordered factorization "
                "invariant before further optimization."
            ),
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "training_launch_allowed": False,
            "required_next_evidence": (
                "A fresh source-bound mechanism recovery diagnostic."
            ),
        }
    else:
        recommendation = {
            "id": CAPACITY_COMPLETION_QUALITY_HOLD_ID,
            "category": "capacity_quality_not_supported",
            "objective": (
                "Hold 250M completion and revisit the matched training objective "
                "from the failed 10K-to-50K quality evidence."
            ),
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "training_launch_allowed": False,
            "required_next_evidence": (
                "A new bounded source-compatible quality diagnostic or recipe decision."
            ),
        }
    terminal = _terminal_evaluation_contract()
    authorization = {
        "status": "authorized" if supported else "not_authorized",
        "matched_250m_resume_allowed": supported,
        "gpu_execution_allowed": supported,
        "training_launch_allowed": supported,
        "fresh_training_allowed": False,
        "resume_from_step": CAPACITY_SCALING_TARGET_STEP,
        "stop_after_step": CAPACITY_COMPLETION_TARGET_STEP,
        "configured_training_horizon": CAPACITY_COMPLETION_TARGET_STEP,
        "terminal_evaluation": terminal,
        "full_300k_launch_allowed": False,
        "promotion_or_release_allowed": False,
    }
    return {
        "schema_version": CAPACITY_COMPLETION_DECISION_SCHEMA_VERSION,
        "status": "completed",
        "role": CAPACITY_COMPLETION_DECISION_ROLE,
        "git": decision_provenance,
        "source_evidence": {
            "capacity_scaling_50k_result": result_identity,
            "capacity_scaling_execution_git": result["git"],
            "capacity_scaling_result_builder_git": result["result_builder_git"],
            "capacity_scaling_source_reports": copy.deepcopy(
                capacity_scaling_result["source_reports"]
            ),
            "standing_authorization": {
                "source": standing_identity,
                "validated_record": standing,
            },
        },
        "selection": {
            "dataset": "imagenet_256",
            "methods": list(CAPACITY_SCALING_METHODS),
            "model_base_channels": 256,
            "parameter_counts": {
                method: CAPACITY_PROBE_PARAMETER_COUNTS["base256"][method]
                for method in CAPACITY_SCALING_METHODS
            },
            "effective_batch_size": CAPACITY_PROBE_EFFECTIVE_BATCH,
            "output_root": result["output_root"],
            "resume_sources": result["resume_sources"],
            "configured_training_horizon": CAPACITY_COMPLETION_TARGET_STEP,
            "resume_from_step": CAPACITY_SCALING_TARGET_STEP,
            "stop_after_step": CAPACITY_COMPLETION_TARGET_STEP,
            "milestone_evaluation": {
                "sample_count_per_method": 2_048,
                "weights": "ema",
                "sampler": "ddim",
                "sample_steps": 50,
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "fixed_random_stream_across_methods": True,
                "cofitok_mechanism_images": 256,
            },
            "terminal_evaluation": terminal,
        },
        "source_decision": {
            "capacity_completion_supported": supported,
            "shared_strict_fid_improvement": result[
                "shared_strict_fid_improvement"
            ],
            "cofitok_mechanism_invariants_valid": mechanism_valid,
            "step_50000_quality_alerts": result["quality_alerts"],
        },
        "recommended_next_stage": recommendation,
        "execution_authorization": authorization,
        "claim_policy": {
            "role": "non_claim_capacity_completion_qualification",
            "formal_generation_claim_allowed": False,
            "cross_stage_numeric_ranking_allowed": False,
            "terminal_100k_result_requires_new_source_compatible_decision": True,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_COMPLETION_DECISION_BOUNDARY
        ),
    }


def validate_capacity_completion_decision(
    report: Mapping[str, Any],
    *,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
) -> dict[str, Any]:
    recommendation = report.get("recommended_next_stage")
    authorization = report.get("execution_authorization")
    selection = report.get("selection")
    source = report.get("source_evidence")
    source_decision = report.get("source_decision")
    claim_policy = report.get("claim_policy")
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_COMPLETION_DECISION_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("role") != CAPACITY_COMPLETION_DECISION_ROLE
        or report.get("authorization_boundary")
        != CAPACITY_COMPLETION_DECISION_BOUNDARY
        or claim_policy
        != {
            "role": "non_claim_capacity_completion_qualification",
            "formal_generation_claim_allowed": False,
            "cross_stage_numeric_ranking_allowed": False,
            "terminal_100k_result_requires_new_source_compatible_decision": True,
        }
        or not isinstance(recommendation, Mapping)
        or not isinstance(authorization, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(source_decision, Mapping)
    ):
        raise ValueError("capacity completion decision contract differs")
    git = _git(
        report.get("git", {}),
        with_tree=True,
        label="capacity completion decision",
    )
    if git != {
        "revision": expected_decision_revision,
        "tree": expected_decision_tree,
        "branch": expected_decision_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity completion decision Git identity differs")
    supported = source_decision.get("capacity_completion_supported")
    mechanism_valid = source_decision.get("cofitok_mechanism_invariants_valid")
    shared = source_decision.get("shared_strict_fid_improvement")
    alerts = source_decision.get("step_50000_quality_alerts")
    allowed = authorization.get("matched_250m_resume_allowed")
    if (
        type(supported) is not bool
        or type(mechanism_valid) is not bool
        or type(shared) is not bool
        or not isinstance(alerts, list)
        or type(allowed) is not bool
        or allowed is not supported
        or supported is not (shared and mechanism_valid and not alerts)
        or authorization.get("status")
        != ("authorized" if supported else "not_authorized")
        or authorization.get("gpu_execution_allowed") is not supported
        or authorization.get("training_launch_allowed") is not supported
        or authorization.get("fresh_training_allowed") is not False
        or int(authorization.get("resume_from_step", -1))
        != CAPACITY_SCALING_TARGET_STEP
        or int(authorization.get("stop_after_step", -1))
        != CAPACITY_COMPLETION_TARGET_STEP
        or int(authorization.get("configured_training_horizon", -1))
        != CAPACITY_COMPLETION_TARGET_STEP
        or authorization.get("terminal_evaluation")
        != _terminal_evaluation_contract()
        or authorization.get("full_300k_launch_allowed") is not False
        or authorization.get("promotion_or_release_allowed") is not False
        or recommendation.get("execution_ready") is not supported
        or recommendation.get("gpu_execution_allowed") is not supported
        or recommendation.get("training_launch_allowed") is not supported
    ):
        raise ValueError("capacity completion execution scope differs")
    expected_recommendation = (
        CAPACITY_COMPLETION_RECOMMENDATION_ID
        if supported
        else (
            CAPACITY_COMPLETION_MECHANISM_HOLD_ID
            if not mechanism_valid
            else CAPACITY_COMPLETION_QUALITY_HOLD_ID
        )
    )
    if recommendation.get("id") != expected_recommendation:
        raise ValueError("capacity completion recommendation differs")
    output_root = selection.get("output_root")
    resume = selection.get("resume_sources")
    if (
        selection.get("dataset") != "imagenet_256"
        or selection.get("methods") != list(CAPACITY_SCALING_METHODS)
        or int(selection.get("model_base_channels", -1)) != 256
        or selection.get("parameter_counts")
        != CAPACITY_PROBE_PARAMETER_COUNTS["base256"]
        or int(selection.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or not isinstance(output_root, str)
        or not _absolute_path(output_root)
        or int(selection.get("resume_from_step", -1))
        != CAPACITY_SCALING_TARGET_STEP
        or int(selection.get("stop_after_step", -1))
        != CAPACITY_COMPLETION_TARGET_STEP
        or int(selection.get("configured_training_horizon", -1))
        != CAPACITY_COMPLETION_TARGET_STEP
        or selection.get("terminal_evaluation") != _terminal_evaluation_contract()
        or not isinstance(resume, Mapping)
        or set(resume) != set(CAPACITY_SCALING_METHODS)
    ):
        raise ValueError("capacity completion selection differs")
    normalized_resume = {}
    for method in CAPACITY_SCALING_METHODS:
        row = resume[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"capacity completion {method} resume source is malformed")
        checkpoint = _identity(
            row.get("checkpoint", {}),
            label=f"capacity completion {method} checkpoint",
        )
        integrity = _identity(
            row.get("checkpoint_integrity_manifest", {}),
            label=f"capacity completion {method} integrity",
        )
        if (
            not checkpoint["path"].startswith(f"{output_root}/")
            or not checkpoint["path"].endswith("checkpoint_step_00050000.pt")
            or integrity["path"] != f"{checkpoint['path']}.integrity.json"
        ):
            raise ValueError(f"capacity completion {method} resume source differs")
        normalized_resume[method] = {
            "checkpoint": checkpoint,
            "checkpoint_integrity_manifest": integrity,
        }
    _identity(
        source.get("capacity_scaling_50k_result", {}),
        label="capacity completion source result",
    )
    standing = source.get("standing_authorization")
    if not isinstance(standing, Mapping):
        raise ValueError("capacity completion standing authorization is missing")
    _identity(
        standing.get("source", {}),
        label="capacity completion standing authorization",
    )
    validate_standing_experiment_authorization(
        standing.get("validated_record", {})
    )
    return {
        "execution_authorized": allowed,
        "recommended_next_stage": copy.deepcopy(dict(recommendation)),
        "resume_sources": normalized_resume,
        "output_root": output_root,
        "terminal_evaluation": _terminal_evaluation_contract(),
        "git": git,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_COMPLETION_DECISION_BOUNDARY
        ),
    }
