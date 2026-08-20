from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from cofitok.generation.quality_bridge_followup import (
    ABSOLUTE_QUALITY_CHECKS,
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_ROLE,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
)
from cofitok.generation_recipe import (
    generation_training_recipe_contract,
    infer_generation_training_stage,
)


EXPOSURE_QUALIFICATION_SCHEMA_VERSION = 1
EXPOSURE_QUALIFICATION_ROLE = (
    "stability_full_data_base128_exposure_qualification_preparation"
)
EXPOSURE_QUALIFICATION_STAGE = "stability_exposure"
EXPOSURE_QUALIFICATION_RECOMMENDATION_ID = (
    "prepare_matched_training_exposure_qualification"
)
EXPOSURE_QUALIFICATION_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_300k_base128_exposure_qualification_v1"
)
CAPACITY_RECOMMENDATION_ID = "prepare_matched_250m_capacity_qualification_probe"
CAPACITY_RESULT_ROLE = "stability_full_data_capacity_probe_result"
CAPACITY_RESULT_SOURCE_NAMES = {
    "preparation",
    "launch_receipt",
    "base256_cofitok_training_validation",
    "base256_dense_identity_training_validation",
    *{
        f"{arm}_{kind}"
        for arm in (
            "base128_cofitok",
            "base128_dense_identity",
            "base256_cofitok",
            "base256_dense_identity",
        )
        for kind in ("sampling_preflight", "generation", "checkpoint_eval")
    },
}
CAPACITY_RESULT_BOUNDARY = {
    "capacity_probe_evidence_complete": True,
    "capacity_probe_execution_allowed": False,
    "additional_training_allowed": False,
    "configured_100k_completion_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "new_source_compatible_decision_required": True,
}
EXPOSURE_QUALIFICATION_BOUNDARY = {
    "exposure_qualification_prepared": True,
    "exposure_qualification_execution_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "exact_execution_authorization_required": True,
    "new_terminal_gate_required": True,
}

DATASET = "imagenet_256"
TRAIN_IMAGE_COUNT = 1_281_167
EFFECTIVE_BATCH_SIZE = 64
SOURCE_STEPS = 100_000
TARGET_STEPS = 300_000
TARGET_IMAGES_SEEN = TARGET_STEPS * EFFECTIVE_BATCH_SIZE
HISTORICAL_REFERENCE_TRAIN_IMAGES = 128_161
HISTORICAL_REFERENCE_STEPS = 50_000
HISTORICAL_REFERENCE_IMAGES_SEEN = (
    HISTORICAL_REFERENCE_STEPS * EFFECTIVE_BATCH_SIZE
)
HISTORICAL_EXPOSURE_MATCH_STEPS = math.ceil(
    (HISTORICAL_REFERENCE_IMAGES_SEEN / HISTORICAL_REFERENCE_TRAIN_IMAGES)
    * TRAIN_IMAGE_COUNT
    / EFFECTIVE_BATCH_SIZE
)
TARGET_MILESTONES = [50_000, 100_000, 150_000, 200_000, 300_000]
TREND_EVALUATION_MILESTONES = [50_000, 100_000, 150_000, 200_000, 300_000]
PARAMETER_COUNTS = {
    "cofitok": 62_834_083,
    "dense_identity": 62_824_707,
}
ALLOWED_SOURCE_TO_TARGET_CONFIG_DIFFERENCES = {
    "name",
    "runtime.steps",
    "runtime.protected_checkpoint_steps",
}
SOURCE_NAMES = {
    "followup_decision",
    "capacity_probe_result",
    "source_cofitok_config",
    "source_dense_identity_config",
    "target_cofitok_config",
    "target_dense_identity_config",
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


def _git(
    value: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    label: str,
) -> dict[str, Any]:
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if not _hex(expected_revision, length=40) or not expected_branch:
        raise ValueError(f"expected {label} Git identity is malformed")
    if dict(value) != expected:
        raise ValueError(f"{label} Git identity differs")
    return expected


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _differences(left: Any, right: Any, path: str = "") -> set[str]:
    if type(left) is not type(right):
        return {path}
    if isinstance(left, Mapping):
        result: set[str] = set()
        for key in set(left) | set(right):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                result.add(child)
            else:
                result.update(_differences(left[key], right[key], child))
        return result
    if isinstance(left, list):
        return set() if left == right else {path}
    return set() if left == right else {path}


def _validate_followup_decision(
    decision: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    recommendation = decision.get("recommended_next_stage")
    exposure = decision.get("training_exposure")
    terminal = decision.get("terminal_quality")
    trend = decision.get("milestone_trend")
    if (
        int(decision.get("schema_version", -1)) != FOLLOWUP_DECISION_SCHEMA_VERSION
        or decision.get("status") != "completed"
        or decision.get("role") != FOLLOWUP_DECISION_ROLE
        or decision.get("authorization_boundary") != FOLLOWUP_AUTHORIZATION_BOUNDARY
        or not isinstance(recommendation, Mapping)
        or not isinstance(exposure, Mapping)
        or not isinstance(terminal, Mapping)
        or not isinstance(trend, Mapping)
    ):
        raise ValueError("exposure qualification requires the canonical v2 decision")
    decision_git = _git(
        decision.get("decision_builder_git", {}),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        label="exposure-aware follow-up decision",
    )
    fallback = recommendation.get("fallback_if_capacity_not_supported")
    if (
        recommendation.get("id") != CAPACITY_RECOMMENDATION_ID
        or recommendation.get("category") != "capacity_qualification"
        or recommendation.get("execution_ready") is not False
        or recommendation.get("gpu_execution_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
        or not isinstance(fallback, Mapping)
        or fallback.get("id") != EXPOSURE_QUALIFICATION_RECOMMENDATION_ID
        or fallback.get("category") != "training_exposure_qualification"
        or fallback.get("execution_ready") is not False
        or fallback.get("gpu_execution_allowed") is not False
        or fallback.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("v2 decision does not select the exposure fallback")
    failed_checks = terminal.get("failed_checks")
    if (
        terminal.get("status") != "hold"
        or not isinstance(failed_checks, list)
        or not failed_checks
        or any(not isinstance(value, str) for value in failed_checks)
        or not set(failed_checks).issubset(ABSOLUTE_QUALITY_CHECKS)
        or recommendation.get("trigger", {}).get("failed_checks")
        != sorted(failed_checks)
        or trend.get("shared_quality_trend_strictly_improved") is not True
    ):
        raise ValueError("v2 decision does not isolate absolute quality and exposure")
    source_epochs = _finite(
        exposure.get("full_data_equivalent_epochs"),
        label="source full-data equivalent epochs",
    )
    historical_epochs = _finite(
        exposure.get("historical_10pct_reference_equivalent_epochs"),
        label="historical reference equivalent epochs",
    )
    expected_source_epochs = SOURCE_STEPS * EFFECTIVE_BATCH_SIZE / TRAIN_IMAGE_COUNT
    expected_historical_epochs = (
        HISTORICAL_REFERENCE_IMAGES_SEEN / HISTORICAL_REFERENCE_TRAIN_IMAGES
    )
    if (
        exposure.get("dataset") != DATASET
        or int(exposure.get("train_image_count", -1)) != TRAIN_IMAGE_COUNT
        or int(exposure.get("steps_per_method", -1)) != SOURCE_STEPS
        or int(exposure.get("effective_batch_size", -1)) != EFFECTIVE_BATCH_SIZE
        or int(exposure.get("images_seen_per_method", -1))
        != SOURCE_STEPS * EFFECTIVE_BATCH_SIZE
        or not math.isclose(source_epochs, expected_source_epochs, abs_tol=1e-12)
        or not math.isclose(
            historical_epochs,
            expected_historical_epochs,
            abs_tol=1e-12,
        )
        or exposure.get("below_historical_reference_exposure") is not True
        or exposure.get("insufficient_exposure_is_live_hypothesis") is not True
        or exposure.get("terminal_result_content_bound") is not True
        or exposure.get("terminal_checkpoint_binding_verified") is not True
    ):
        raise ValueError("v2 decision training exposure differs")
    return {
        "git": decision_git,
        "source_epochs": source_epochs,
        "historical_reference_epochs": historical_epochs,
        "failed_checks": sorted(failed_checks),
    }


def _validate_capacity_result(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    sources = report.get("source_reports")
    decision = report.get("decision")
    selection = report.get("selection")
    evaluation = report.get("evaluation")
    estimands = report.get("estimands")
    claim = report.get("claim_policy")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "completed"
        or report.get("role") != CAPACITY_RESULT_ROLE
        or report.get("authorization_boundary") != CAPACITY_RESULT_BOUNDARY
        or not isinstance(sources, Mapping)
        or set(sources) != CAPACITY_RESULT_SOURCE_NAMES
        or not isinstance(decision, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(evaluation, Mapping)
        or not isinstance(estimands, Mapping)
        or not isinstance(claim, Mapping)
    ):
        raise ValueError("capacity result contract differs")
    result_git = _git(
        report.get("git", {}),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        label="capacity result",
    )
    recommendation = decision.get("recommendation")
    if (
        decision.get("shared_strict_fid_improvement") is not False
        or decision.get("cofitok_mechanism_invariants_valid") is not True
        or decision.get("capacity_supported") is not False
        or not isinstance(recommendation, Mapping)
        or recommendation.get("id")
        != "hold_capacity_scaling_and_revisit_training_objective"
        or recommendation.get("category") != "capacity_not_supported"
        or recommendation.get("execution_ready") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("capacity result does not support exposure qualification")
    arms = evaluation.get("arms")
    expected_arms = {
        "base128_cofitok",
        "base128_dense_identity",
        "base256_cofitok",
        "base256_dense_identity",
    }
    if (
        selection.get("dataset") != DATASET
        or int(selection.get("configured_training_horizon", -1)) != SOURCE_STEPS
        or int(selection.get("stop_after_step", -1)) != 10_000
        or int(selection.get("effective_batch_size", -1)) != EFFECTIVE_BATCH_SIZE
        or selection.get("base_channels") != [128, 256]
        or not isinstance(arms, Mapping)
        or set(arms) != expected_arms
        or claim.get("role") != "non_claim_capacity_causal_diagnostic"
        or int(claim.get("sample_count_per_arm", -1)) != 2_048
        or claim.get("formal_generation_claim_allowed") is not False
        or claim.get("cross_stage_numeric_ranking_allowed") is not False
    ):
        raise ValueError("capacity diagnostic scope differs")
    normalized_sources = {
        name: _identity(value, label=f"capacity result source {name}")
        for name, value in sources.items()
    }
    fids = {
        arm: _finite(arms[arm].get("fid"), label=f"capacity result {arm} FID")
        for arm in expected_arms
    }
    if any(value < 0.0 for value in fids.values()):
        raise ValueError("capacity result FID domain differs")
    cofitok_delta = fids["base256_cofitok"] - fids["base128_cofitok"]
    dense_delta = (
        fids["base256_dense_identity"] - fids["base128_dense_identity"]
    )
    interaction = cofitok_delta - dense_delta
    shared_improvement = cofitok_delta < 0.0 and dense_delta < 0.0
    mechanism_valid = all(
        arms[arm].get("mechanism_invariants_valid") is True
        for arm in ("base128_cofitok", "base256_cofitok")
    )
    if (
        not math.isclose(
            _finite(
                estimands.get("cofitok_fid_delta_base256_minus_base128"),
                label="capacity CoFiTok FID delta",
            ),
            cofitok_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(
                estimands.get("dense_fid_delta_base256_minus_base128"),
                label="capacity dense FID delta",
            ),
            dense_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(
                estimands.get("capacity_by_factorization_fid_interaction"),
                label="capacity-by-factorization FID interaction",
            ),
            interaction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or decision.get("shared_strict_fid_improvement") is not shared_improvement
        or decision.get("cofitok_mechanism_invariants_valid") is not mechanism_valid
        or shared_improvement
        or not mechanism_valid
    ):
        raise ValueError("capacity result estimands or trigger are inconsistent")
    return {
        "git": result_git,
        "sources": normalized_sources,
        "estimands": copy.deepcopy(dict(estimands)),
        "recommendation": copy.deepcopy(dict(recommendation)),
    }


def _validate_configs(
    *,
    source_cofitok: Mapping[str, Any],
    source_dense: Mapping[str, Any],
    target_cofitok: Mapping[str, Any],
    target_dense: Mapping[str, Any],
) -> dict[str, Any]:
    source_inferred = infer_generation_training_stage(
        dict(source_cofitok),
        dict(source_dense),
    )
    target_inferred = infer_generation_training_stage(
        dict(target_cofitok),
        dict(target_dense),
    )
    if source_inferred != "stability_quality_bridge":
        raise ValueError("source quality-bridge stage differs")
    if target_inferred != EXPOSURE_QUALIFICATION_STAGE:
        raise ValueError("target exposure stage differs")
    source_recipe = generation_training_recipe_contract(
        dict(source_cofitok),
        dict(source_dense),
        stage="stability_quality_bridge",
    )
    target_recipe = generation_training_recipe_contract(
        dict(target_cofitok),
        dict(target_dense),
        stage=EXPOSURE_QUALIFICATION_STAGE,
    )
    if source_recipe["valid"] is not True:
        raise ValueError(
            "source quality-bridge recipe is invalid: "
            + "; ".join(source_recipe["issues"])
        )
    if target_recipe["valid"] is not True:
        raise ValueError(
            "target exposure recipe is invalid: "
            + "; ".join(target_recipe["issues"])
        )
    differences = {
        "cofitok": sorted(_differences(source_cofitok, target_cofitok)),
        "dense_identity": sorted(_differences(source_dense, target_dense)),
    }
    expected_differences = sorted(ALLOWED_SOURCE_TO_TARGET_CONFIG_DIFFERENCES)
    if any(value != expected_differences for value in differences.values()):
        raise ValueError("source-to-target exposure config differences are not exact")
    return {
        "source_stage": source_inferred,
        "source_valid": True,
        "target_stage": target_inferred,
        "target_valid": True,
        "source_to_target_differences": differences,
        "target_pair_contract_valid": target_recipe["pair_contract"]["valid"],
        "target_token_layout": copy.deepcopy(target_recipe["token_layout"]),
    }


def build_training_exposure_qualification_preparation(
    *,
    followup_decision: Mapping[str, Any],
    capacity_probe_result: Mapping[str, Any],
    source_cofitok_config: Mapping[str, Any],
    source_dense_identity_config: Mapping[str, Any],
    target_cofitok_config: Mapping[str, Any],
    target_dense_identity_config: Mapping[str, Any],
    source_identities: Mapping[str, Mapping[str, Any]],
    expected_followup_revision: str,
    expected_followup_branch: str,
    expected_capacity_revision: str,
    expected_capacity_branch: str,
    expected_preparation_revision: str,
    expected_preparation_branch: str,
    preparation_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    if set(source_identities) != SOURCE_NAMES:
        raise ValueError("exposure qualification source set differs")
    sources = {
        name: _identity(identity, label=f"exposure qualification source {name}")
        for name, identity in source_identities.items()
    }
    followup = _validate_followup_decision(
        followup_decision,
        expected_revision=expected_followup_revision,
        expected_branch=expected_followup_branch,
    )
    capacity = _validate_capacity_result(
        capacity_probe_result,
        expected_revision=expected_capacity_revision,
        expected_branch=expected_capacity_branch,
    )
    current_git = _git(
        preparation_git,
        expected_revision=expected_preparation_revision,
        expected_branch=expected_preparation_branch,
        label="exposure qualification preparation",
    )
    config_contract = _validate_configs(
        source_cofitok=source_cofitok_config,
        source_dense=source_dense_identity_config,
        target_cofitok=target_cofitok_config,
        target_dense=target_dense_identity_config,
    )
    output_path = PurePosixPath(output_root)
    if (
        output_root != EXPOSURE_QUALIFICATION_OUTPUT_ROOT
        or not output_path.is_absolute()
        or ".." in output_path.parts
    ):
        raise ValueError("exposure qualification output root is unsafe")

    target_epochs = TARGET_IMAGES_SEEN / TRAIN_IMAGE_COUNT
    historical_epochs = (
        HISTORICAL_REFERENCE_IMAGES_SEEN / HISTORICAL_REFERENCE_TRAIN_IMAGES
    )
    return {
        "schema_version": EXPOSURE_QUALIFICATION_SCHEMA_VERSION,
        "status": "prepared",
        "role": EXPOSURE_QUALIFICATION_ROLE,
        "git": current_git,
        "output_root": output_root,
        "source_reports": sources,
        "source_replay": {
            "v2_terminal_and_exposure_sources_reverified": True,
            "capacity_result_sources_reverified": True,
            "followup_decision_git": followup["git"],
            "capacity_result_git": capacity["git"],
        },
        "trigger": {
            "v2_recommendation": CAPACITY_RECOMMENDATION_ID,
            "capacity_supported": False,
            "capacity_recommendation": capacity["recommendation"],
            "cofitok_mechanism_invariants_valid": True,
            "source_terminal_failed_checks": followup["failed_checks"],
            "insufficient_exposure_is_live_hypothesis": True,
        },
        "selection": {
            "id": "fresh_matched_base128_full_data_300k_exposure_stage1",
            "dataset": DATASET,
            "train_image_count": TRAIN_IMAGE_COUNT,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "model_base_channels": 128,
            "parameter_counts": copy.deepcopy(PARAMETER_COUNTS),
            "source_steps": SOURCE_STEPS,
            "target_steps": TARGET_STEPS,
            "target_images_seen_per_method": TARGET_IMAGES_SEEN,
            "target_full_data_equivalent_epochs": target_epochs,
            "historical_10pct_reference_equivalent_epochs": historical_epochs,
            "target_to_historical_reference_epoch_ratio": (
                target_epochs / historical_epochs
            ),
            "historical_exposure_match_steps": HISTORICAL_EXPOSURE_MATCH_STEPS,
            "historical_exposure_fully_matched": False,
            "residual_underexposure_hypothesis_after_300k": True,
            "fresh_initialization_required": True,
            "resume_from_quality_bridge_forbidden": True,
        },
        "matched_training_contract": {
            "intervention": "training_horizon_only",
            "allowed_source_to_target_config_differences": sorted(
                ALLOWED_SOURCE_TO_TARGET_CONFIG_DIFFERENCES
            ),
            "config_validation": config_contract,
            "same_dataset": True,
            "same_model_capacity": True,
            "same_factorization_contract": True,
            "same_optimizer_hyperparameters": True,
            "same_absolute_stability_loss_schedule": True,
            "same_effective_batch_size": True,
            "same_seed": True,
            "cosine_horizon_changes_with_target_steps": True,
            "capacity_intervention_allowed": False,
            "recipe_or_objective_intervention_allowed": False,
        },
        "evaluation_contract": {
            "protected_checkpoint_steps": copy.deepcopy(TARGET_MILESTONES),
            "trend_evaluation_steps": copy.deepcopy(TREND_EVALUATION_MILESTONES),
            "trend_samples_per_method": 2_048,
            "trend_sampler": "ddim",
            "trend_sample_steps": 50,
            "terminal_samples_per_method": 10_000,
            "terminal_sampler": "ddim",
            "terminal_sample_steps": 100,
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "weights": "ema",
            "fixed_random_stream_across_methods": True,
            "metrics": ["fid", "inception_score", "precision", "recall"],
            "cofitok_mechanism_diagnostics_required": True,
            "class_fidelity_required": True,
        },
        "decision_contract": {
            "300k_can_pass_terminal_quality_gate": True,
            "300k_can_prove_historical_exposure_saturation": False,
            "if_quality_passes": "build_new_source_compatible_formal_gate",
            "if_quality_holds_and_shared_trend_improves": (
                "prepare_matched_base128_500k_exposure_completion"
            ),
            "if_shared_trend_does_not_improve": (
                "diagnose_distribution_support_then_recipe_intervention"
            ),
            "new_decision_required_before_execution": True,
        },
        "authorization_boundary": copy.deepcopy(EXPOSURE_QUALIFICATION_BOUNDARY),
    }


def validate_training_exposure_qualification_preparation(
    report: Mapping[str, Any],
    *,
    expected_output_root: str,
    expected_preparation_revision: str,
    expected_preparation_branch: str,
) -> dict[str, Any]:
    expected_git = {
        "revision": expected_preparation_revision,
        "branch": expected_preparation_branch,
        "tracked_dirty": False,
    }
    selection = report.get("selection")
    training = report.get("matched_training_contract")
    evaluation = report.get("evaluation_contract")
    decision = report.get("decision_contract")
    trigger = report.get("trigger")
    sources = report.get("source_reports")
    if (
        int(report.get("schema_version", -1))
        != EXPOSURE_QUALIFICATION_SCHEMA_VERSION
        or report.get("status") != "prepared"
        or report.get("role") != EXPOSURE_QUALIFICATION_ROLE
        or report.get("output_root") != expected_output_root
        or expected_output_root != EXPOSURE_QUALIFICATION_OUTPUT_ROOT
        or report.get("git") != expected_git
        or report.get("authorization_boundary") != EXPOSURE_QUALIFICATION_BOUNDARY
        or not all(
            isinstance(value, Mapping)
            for value in (selection, training, evaluation, decision, trigger, sources)
        )
        or set(sources) != SOURCE_NAMES
    ):
        raise ValueError("exposure qualification preparation contract differs")
    for name, value in sources.items():
        _identity(value, label=f"exposure qualification source {name}")
    target_epochs = TARGET_IMAGES_SEEN / TRAIN_IMAGE_COUNT
    historical_epochs = (
        HISTORICAL_REFERENCE_IMAGES_SEEN / HISTORICAL_REFERENCE_TRAIN_IMAGES
    )
    if (
        selection.get("id")
        != "fresh_matched_base128_full_data_300k_exposure_stage1"
        or int(selection.get("target_steps", -1)) != TARGET_STEPS
        or selection.get("historical_exposure_fully_matched") is not False
        or selection.get("residual_underexposure_hypothesis_after_300k") is not True
        or selection.get("fresh_initialization_required") is not True
        or selection.get("resume_from_quality_bridge_forbidden") is not True
        or not math.isclose(
            _finite(
                selection.get("target_full_data_equivalent_epochs"),
                label="target full-data equivalent epochs",
            ),
            target_epochs,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(selection.get("historical_exposure_match_steps", -1))
        != HISTORICAL_EXPOSURE_MATCH_STEPS
        or not math.isclose(
            _finite(
                selection.get("target_to_historical_reference_epoch_ratio"),
                label="target-to-historical exposure ratio",
            ),
            target_epochs / historical_epochs,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or training.get("intervention") != "training_horizon_only"
        or training.get("same_model_capacity") is not True
        or training.get("capacity_intervention_allowed") is not False
        or training.get("recipe_or_objective_intervention_allowed") is not False
        or trigger.get("capacity_supported") is not False
        or trigger.get("cofitok_mechanism_invariants_valid") is not True
        or evaluation.get("protected_checkpoint_steps") != TARGET_MILESTONES
        or evaluation.get("trend_evaluation_steps")
        != TREND_EVALUATION_MILESTONES
        or int(evaluation.get("terminal_samples_per_method", -1)) != 10_000
        or decision.get("300k_can_prove_historical_exposure_saturation") is not False
        or decision.get("new_decision_required_before_execution") is not True
    ):
        raise ValueError("exposure qualification preparation selection differs")
    return copy.deepcopy(dict(selection))
