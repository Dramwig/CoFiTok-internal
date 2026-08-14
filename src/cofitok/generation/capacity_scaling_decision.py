from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_CONFIGURED_STEPS,
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_PARAMETER_COUNTS,
    CAPACITY_PROBE_STOP_STEP,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)
from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_RESULT_BOUNDARY,
    CAPACITY_PROBE_RESULT_ROLE,
    CAPACITY_PROBE_RESULT_SOURCE_NAMES,
)


CAPACITY_SCALING_DECISION_SCHEMA_VERSION = 1
CAPACITY_SCALING_DECISION_ROLE = (
    "stability_full_data_capacity_scaling_decision"
)
CAPACITY_SCALING_RECOMMENDATION_ID = (
    "resume_matched_250m_capacity_bridge_to_50000"
)
CAPACITY_SCALING_HOLD_ID = (
    "hold_250m_capacity_scaling_and_prepare_recipe_intervention"
)
CAPACITY_SCALING_MECHANISM_HOLD_ID = (
    "hold_250m_scaling_and_run_factorization_mechanism_recovery"
)
CAPACITY_SCALING_TARGET_STEP = 50_000
CAPACITY_SCALING_METHODS = ("cofitok", "dense_identity")

CAPACITY_SCALING_DECISION_BOUNDARY = {
    "decision_is_source_replayed": True,
    "maximum_scope_is_matched_250m_step_10000_to_50000": True,
    "fresh_training_allowed": False,
    "resume_from_exact_step_10000_required": True,
    "stop_at_step_50000_required": True,
    "configured_100k_completion_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
}


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


def _git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not _hex(revision, length=40)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def _validate_capacity_result(
    result: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    decision = result.get("decision")
    training = result.get("partial_training")
    sources = result.get("source_reports")
    evaluation = result.get("evaluation")
    claim_policy = result.get("claim_policy")
    if (
        int(result.get("schema_version", -1)) != 1
        or result.get("status") != "completed"
        or result.get("role") != CAPACITY_PROBE_RESULT_ROLE
        or result.get("authorization_boundary") != CAPACITY_PROBE_RESULT_BOUNDARY
        or not isinstance(decision, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(sources, Mapping)
        or not isinstance(evaluation, Mapping)
        or not isinstance(claim_policy, Mapping)
    ):
        raise ValueError("capacity scaling requires the canonical capacity result")
    git = _git(result.get("git", {}), label="capacity probe result")
    if git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity scaling source Git identity differs")
    if set(sources) != CAPACITY_PROBE_RESULT_SOURCE_NAMES:
        raise ValueError("capacity scaling source report set differs")
    if (
        claim_policy.get("role") != "non_claim_capacity_causal_diagnostic"
        or claim_policy.get("formal_generation_claim_allowed") is not False
        or claim_policy.get("cross_stage_numeric_ranking_allowed") is not False
    ):
        raise ValueError("capacity scaling source claim policy differs")
    normalized_sources = {
        name: _identity(identity, label=f"capacity scaling source {name}")
        for name, identity in sources.items()
    }
    output_root = result.get("output_root")
    if (
        not isinstance(output_root, str)
        or not PurePosixPath(output_root).is_absolute()
    ):
        raise ValueError("capacity scaling output root must be absolute")
    if set(training) != set(CAPACITY_SCALING_METHODS):
        raise ValueError("capacity scaling partial training set differs")
    normalized_training: dict[str, Any] = {}
    for method in CAPACITY_SCALING_METHODS:
        row = training[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"capacity scaling {method} training row is malformed")
        checkpoint = _identity(
            row.get("checkpoint", {}),
            label=f"capacity scaling {method} checkpoint",
        )
        integrity = _identity(
            row.get("checkpoint_integrity_manifest", {}),
            label=f"capacity scaling {method} checkpoint integrity",
        )
        expected_parameters = CAPACITY_PROBE_PARAMETER_COUNTS["base256"][method]
        expected_suffix = f"checkpoint_step_{CAPACITY_PROBE_STOP_STEP:08d}.pt"
        if (
            not checkpoint["path"].startswith(f"{output_root}/")
            or not checkpoint["path"].endswith(expected_suffix)
            or integrity["path"] != f"{checkpoint['path']}.integrity.json"
            or int(row.get("images_seen", -1))
            != CAPACITY_PROBE_STOP_STEP * CAPACITY_PROBE_EFFECTIVE_BATCH
            or int(row.get("parameter_count", -1)) != expected_parameters
            or not _hex(row.get("runtime_environment_sha256"), length=64)
            or not _hex(row.get("dataset_identity_sha256"), length=64)
        ):
            raise ValueError(f"capacity scaling {method} resume source differs")
        normalized_training[method] = {
            "run_dir": str(PurePosixPath(checkpoint["path"]).parent),
            "checkpoint": checkpoint,
            "checkpoint_integrity_manifest": integrity,
            "images_seen": int(row["images_seen"]),
            "parameter_count": int(row["parameter_count"]),
            "runtime_environment_sha256": row["runtime_environment_sha256"],
            "dataset_identity_sha256": row["dataset_identity_sha256"],
        }
    if (
        normalized_training["cofitok"]["dataset_identity_sha256"]
        != normalized_training["dense_identity"]["dataset_identity_sha256"]
        or normalized_training["cofitok"]["runtime_environment_sha256"]
        != normalized_training["dense_identity"]["runtime_environment_sha256"]
    ):
        raise ValueError("capacity scaling matched resume identities differ")
    recommendation = decision.get("recommendation")
    if not isinstance(recommendation, Mapping):
        raise ValueError("capacity probe recommendation is malformed")
    capacity_supported = decision.get("capacity_supported")
    shared_improvement = decision.get("shared_strict_fid_improvement")
    mechanism_valid = decision.get("cofitok_mechanism_invariants_valid")
    if (
        type(capacity_supported) is not bool
        or type(shared_improvement) is not bool
        or type(mechanism_valid) is not bool
        or capacity_supported != (shared_improvement and mechanism_valid)
        or recommendation.get("execution_ready") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("capacity probe decision contract differs")
    if capacity_supported and (
        recommendation.get("id")
        != "prepare_source_compatible_capacity_scaling_decision"
        or recommendation.get("category") != "capacity_supported"
    ):
        raise ValueError("supported capacity result recommendation differs")
    if not capacity_supported and recommendation.get("category") not in {
        "capacity_not_qualified",
        "capacity_not_supported",
    }:
        raise ValueError("held capacity result recommendation differs")
    arms = evaluation.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {
        "base128_cofitok",
        "base128_dense_identity",
        "base256_cofitok",
        "base256_dense_identity",
    }:
        raise ValueError("capacity scaling four-arm evidence differs")
    return {
        "git": git,
        "output_root": output_root,
        "source_reports": normalized_sources,
        "partial_training": normalized_training,
        "capacity_supported": capacity_supported,
        "shared_strict_fid_improvement": shared_improvement,
        "cofitok_mechanism_invariants_valid": mechanism_valid,
        "source_recommendation": copy.deepcopy(dict(recommendation)),
    }


def build_capacity_scaling_decision(
    *,
    capacity_probe_result: Mapping[str, Any],
    capacity_probe_result_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    decision_git: Mapping[str, Any],
    expected_capacity_revision: str,
    expected_capacity_branch: str,
) -> dict[str, Any]:
    result_identity = _identity(
        capacity_probe_result_identity,
        label="capacity probe result",
    )
    standing_identity = _identity(
        standing_authorization_identity,
        label="standing experiment authorization",
    )
    normalized_standing = validate_standing_experiment_authorization(
        standing_authorization
    )
    if normalized_standing["preserved_safety_boundaries"] != (
        STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing experiment safety boundary differs")
    source = _validate_capacity_result(
        capacity_probe_result,
        expected_revision=expected_capacity_revision,
        expected_branch=expected_capacity_branch,
    )
    git = _git(decision_git, label="capacity scaling decision")
    supported = source["capacity_supported"]
    if supported:
        recommendation = {
            "id": CAPACITY_SCALING_RECOMMENDATION_ID,
            "category": "capacity_scaling_qualification",
            "objective": (
                "Resume both exact matched 250M step-10K runs to step 50K and "
                "produce a paired non-claim milestone evaluation."
            ),
            "execution_ready": True,
            "gpu_execution_allowed": True,
            "training_launch_allowed": True,
            "required_next_evidence": (
                "Exact-resume validation for both step-50K checkpoints plus a matched "
                "2,048-sample DDIM-50 milestone and CoFiTok mechanism audit."
            ),
        }
    elif source["shared_strict_fid_improvement"]:
        recommendation = {
            "id": CAPACITY_SCALING_MECHANISM_HOLD_ID,
            "category": "cofitok_mechanism_recovery",
            "objective": (
                "Hold additional 250M training and restore the failed ordered "
                "factorization invariant before any capacity extension."
            ),
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "training_launch_allowed": False,
            "required_next_evidence": (
                "A fresh source-bound matched mechanism recovery diagnostic that "
                "restores the failed CoFiTok invariant."
            ),
        }
    else:
        recommendation = {
            "id": CAPACITY_SCALING_HOLD_ID,
            "category": "recipe_or_objective_intervention",
            "objective": (
                "Hold additional 250M training and prepare a matched recipe or "
                "factorization intervention from the failed capacity evidence."
            ),
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "training_launch_allowed": False,
            "required_next_evidence": (
                "A new source-bound bounded diagnostic that addresses the failed "
                "capacity or mechanism condition."
            ),
        }
    execution_authorization = {
        "status": "authorized" if supported else "not_authorized",
        "matched_250m_resume_allowed": supported,
        "gpu_execution_allowed": supported,
        "training_launch_allowed": supported,
        "fresh_training_allowed": False,
        "resume_from_step": CAPACITY_PROBE_STOP_STEP,
        "stop_after_step": CAPACITY_SCALING_TARGET_STEP,
        "configured_training_horizon": CAPACITY_PROBE_CONFIGURED_STEPS,
        "configured_100k_completion_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_or_release_allowed": False,
    }
    return {
        "schema_version": CAPACITY_SCALING_DECISION_SCHEMA_VERSION,
        "status": "completed",
        "role": CAPACITY_SCALING_DECISION_ROLE,
        "git": git,
        "source_evidence": {
            "capacity_probe_result": result_identity,
            "capacity_probe_git": source["git"],
            "capacity_probe_source_reports": source["source_reports"],
            "standing_authorization": {
                "source": standing_identity,
                "validated_record": normalized_standing,
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
            "output_root": source["output_root"],
            "resume_sources": source["partial_training"],
            "configured_training_horizon": CAPACITY_PROBE_CONFIGURED_STEPS,
            "resume_from_step": CAPACITY_PROBE_STOP_STEP,
            "stop_after_step": CAPACITY_SCALING_TARGET_STEP,
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
        },
        "source_decision": {
            "capacity_supported": source["capacity_supported"],
            "shared_strict_fid_improvement": source[
                "shared_strict_fid_improvement"
            ],
            "cofitok_mechanism_invariants_valid": source[
                "cofitok_mechanism_invariants_valid"
            ],
            "recommendation": source["source_recommendation"],
        },
        "recommended_next_stage": recommendation,
        "execution_authorization": execution_authorization,
        "claim_policy": {
            "role": "non_claim_capacity_scaling_qualification",
            "formal_generation_claim_allowed": False,
            "cross_stage_numeric_ranking_allowed": False,
            "step_50000_result_requires_new_source_compatible_decision": True,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_DECISION_BOUNDARY
        ),
    }


def validate_capacity_scaling_decision(
    report: Mapping[str, Any],
    *,
    expected_decision_revision: str,
    expected_decision_branch: str,
) -> dict[str, Any]:
    recommendation = report.get("recommended_next_stage")
    authorization = report.get("execution_authorization")
    selection = report.get("selection")
    source = report.get("source_evidence")
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_SCALING_DECISION_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("role") != CAPACITY_SCALING_DECISION_ROLE
        or report.get("authorization_boundary")
        != CAPACITY_SCALING_DECISION_BOUNDARY
        or not isinstance(recommendation, Mapping)
        or not isinstance(authorization, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(source, Mapping)
    ):
        raise ValueError("capacity scaling decision contract differs")
    git = _git(report.get("git", {}), label="capacity scaling decision")
    if git != {
        "revision": expected_decision_revision,
        "branch": expected_decision_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity scaling decision Git identity differs")
    allowed = authorization.get("matched_250m_resume_allowed")
    if type(allowed) is not bool:
        raise ValueError("capacity scaling authorization is malformed")
    expected_status = "authorized" if allowed else "not_authorized"
    if (
        authorization.get("status") != expected_status
        or authorization.get("gpu_execution_allowed") is not allowed
        or authorization.get("training_launch_allowed") is not allowed
        or authorization.get("fresh_training_allowed") is not False
        or int(authorization.get("resume_from_step", -1))
        != CAPACITY_PROBE_STOP_STEP
        or int(authorization.get("stop_after_step", -1))
        != CAPACITY_SCALING_TARGET_STEP
        or int(authorization.get("configured_training_horizon", -1))
        != CAPACITY_PROBE_CONFIGURED_STEPS
        or authorization.get("configured_100k_completion_allowed") is not False
        or authorization.get("full_300k_launch_allowed") is not False
        or authorization.get("promotion_or_release_allowed") is not False
        or recommendation.get("execution_ready") is not allowed
        or recommendation.get("gpu_execution_allowed") is not allowed
        or recommendation.get("training_launch_allowed") is not allowed
    ):
        raise ValueError("capacity scaling execution scope differs")
    if allowed and recommendation.get("id") != CAPACITY_SCALING_RECOMMENDATION_ID:
        raise ValueError("capacity scaling recommendation differs")
    if not allowed and recommendation.get("id") not in {
        CAPACITY_SCALING_HOLD_ID,
        CAPACITY_SCALING_MECHANISM_HOLD_ID,
    }:
        raise ValueError("capacity scaling hold recommendation differs")
    if (
        selection.get("methods") != list(CAPACITY_SCALING_METHODS)
        or int(selection.get("resume_from_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or int(selection.get("stop_after_step", -1))
        != CAPACITY_SCALING_TARGET_STEP
        or int(selection.get("configured_training_horizon", -1))
        != CAPACITY_PROBE_CONFIGURED_STEPS
    ):
        raise ValueError("capacity scaling selection differs")
    _identity(
        source.get("capacity_probe_result", {}),
        label="capacity scaling source result",
    )
    standing = source.get("standing_authorization")
    if not isinstance(standing, Mapping):
        raise ValueError("capacity scaling standing authorization is missing")
    _identity(
        standing.get("source", {}),
        label="capacity scaling standing authorization",
    )
    validate_standing_experiment_authorization(
        standing.get("validated_record", {})
    )
    return {
        "git": git,
        "execution_authorized": allowed,
        "recommended_next_stage": copy.deepcopy(dict(recommendation)),
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_DECISION_BOUNDARY
        ),
    }
