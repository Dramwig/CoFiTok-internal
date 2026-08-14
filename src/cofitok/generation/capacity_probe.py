from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from cofitok.generation.quality_bridge_followup import (
    ABSOLUTE_QUALITY_CHECKS,
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_ROLE,
    QUALITY_BRIDGE_EXECUTION_BRANCH,
    QUALITY_BRIDGE_EXECUTION_REVISION,
)


CAPACITY_PROBE_PREPARATION_SCHEMA_VERSION = 1
CAPACITY_PROBE_PREPARATION_ROLE = (
    "stability_full_data_matched_capacity_probe_preparation"
)
CAPACITY_PROBE_RECOMMENDATION_ID = (
    "prepare_matched_250m_capacity_qualification_probe"
)
CAPACITY_PROBE_SOURCE_DECISION_REVISION = (
    "9b02fa83d20b1459a2706d6d82371caf5c023f54"
)
CAPACITY_PROBE_SOURCE_DECISION_BRANCH = (
    "scale/generation-quality-bridge-followup-decision-v1"
)
CAPACITY_PROBE_REFERENCE_REASON = (
    "matched base128 step-10K reference for a possible 250M capacity "
    "qualification probe"
)
CAPACITY_PROBE_DATASET = "imagenet_256"
CAPACITY_PROBE_CONFIGURED_STEPS = 100_000
CAPACITY_PROBE_STOP_STEP = 10_000
CAPACITY_PROBE_EFFECTIVE_BATCH = 64
CAPACITY_PROBE_SAMPLES_PER_ARM = 2_048
CAPACITY_PROBE_MECHANISM_IMAGES = 256
CAPACITY_PROBE_BASE_CHANNELS = 128
CAPACITY_PROBE_LARGE_CHANNELS = 256
CAPACITY_PROBE_PARAMETER_COUNTS = {
    "base128": {
        "cofitok": 62_834_083,
        "dense_identity": 62_824_707,
    },
    "base256": {
        "cofitok": 250_153_763,
        "dense_identity": 250_135_043,
    },
}
CAPACITY_PROBE_ALLOWED_CONFIG_DIFFERENCES = {
    "name",
    "model.base_channels",
    "runtime.protected_checkpoint_steps",
}
CAPACITY_PROBE_TRAINING_INTERVENTION = {"model.base_channels"}
CAPACITY_PROBE_LOGISTICAL_DIFFERENCES = {
    "name",
    "runtime.protected_checkpoint_steps",
}

CAPACITY_PROBE_PREPARATION_BOUNDARY = {
    "capacity_probe_prepared": True,
    "capacity_probe_execution_allowed": False,
    "gpu_execution_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
    "exact_execution_receipt_required": True,
}


def validate_capacity_probe_preparation_contract(
    report: Mapping[str, Any],
    *,
    expected_output_root: str,
) -> dict[str, Any]:
    selection = report.get("selection")
    matched = report.get("matched_training_contract")
    evaluation = report.get("evaluation_contract")
    decision = report.get("decision_contract")
    references = report.get("checkpoint_references")
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_PROBE_PREPARATION_SCHEMA_VERSION
        or report.get("status") != "prepared"
        or report.get("role") != CAPACITY_PROBE_PREPARATION_ROLE
        or report.get("authorization_boundary")
        != CAPACITY_PROBE_PREPARATION_BOUNDARY
        or not all(
            isinstance(value, Mapping)
            for value in (selection, matched, evaluation, decision, references)
        )
    ):
        raise ValueError("capacity probe preparation contract differs")
    if (
        selection.get("dataset") != CAPACITY_PROBE_DATASET
        or int(selection.get("configured_training_horizon", -1))
        != CAPACITY_PROBE_CONFIGURED_STEPS
        or int(selection.get("stop_after_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or int(selection.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or selection.get("base_channels")
        != [CAPACITY_PROBE_BASE_CHANNELS, CAPACITY_PROBE_LARGE_CHANNELS]
        or selection.get("output_root") != expected_output_root
        or selection.get("fresh_training_arms")
        != ["base256_cofitok", "base256_dense_identity"]
        or selection.get("preserved_reference_arms")
        != ["base128_cofitok", "base128_dense_identity"]
        or selection.get("training_intervention")
        != sorted(CAPACITY_PROBE_TRAINING_INTERVENTION)
    ):
        raise ValueError("capacity probe preparation selection differs")
    if (
        matched.get("configured_100k_schedule_preserved_at_step_10k") is not True
        or matched.get("fresh_initialization_required") is not True
        or matched.get("resume_from_base128_forbidden") is not True
        or set(matched.get("capacity_configs", {}))
        != {"cofitok", "dense_identity"}
        or set(references) != {"cofitok", "dense_identity"}
    ):
        raise ValueError("capacity probe preparation matched contract differs")
    if (
        evaluation.get("arms")
        != [
            "base128_cofitok",
            "base128_dense_identity",
            "base256_cofitok",
            "base256_dense_identity",
        ]
        or int(evaluation.get("checkpoint_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or evaluation.get("weights") != "ema"
        or evaluation.get("sampler") != "ddim"
        or int(evaluation.get("sample_steps", -1)) != 50
        or float(evaluation.get("guidance_scale", -1.0)) != 1.5
        or int(evaluation.get("samples_per_arm", -1))
        != CAPACITY_PROBE_SAMPLES_PER_ARM
        or evaluation.get("fixed_random_stream_across_arms") is not True
        or evaluation.get("role") != "non_claim_capacity_causal_diagnostic"
        or decision.get("result_can_authorize_full_300k") is not False
        or decision.get("new_source_compatible_decision_required") is not True
    ):
        raise ValueError("capacity probe preparation evaluation contract differs")
    return copy.deepcopy(dict(selection))


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


def _validate_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    recommendation = decision.get("recommended_next_stage")
    terminal = decision.get("terminal_quality")
    trend = decision.get("milestone_trend")
    if (
        int(decision.get("schema_version", -1)) != 1
        or decision.get("status") != "completed"
        or decision.get("role") != FOLLOWUP_DECISION_ROLE
        or decision.get("authorization_boundary") != FOLLOWUP_AUTHORIZATION_BOUNDARY
        or not isinstance(recommendation, Mapping)
        or not isinstance(terminal, Mapping)
        or not isinstance(trend, Mapping)
    ):
        raise ValueError("capacity probe requires the canonical follow-up decision")
    decision_git = _git(
        decision.get("decision_builder_git", {}),
        label="quality bridge follow-up decision builder",
    )
    if decision_git != {
        "revision": CAPACITY_PROBE_SOURCE_DECISION_REVISION,
        "branch": CAPACITY_PROBE_SOURCE_DECISION_BRANCH,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity probe requires the exact follow-up decision builder")
    if (
        recommendation.get("id") != CAPACITY_PROBE_RECOMMENDATION_ID
        or recommendation.get("category") != "capacity_qualification"
        or recommendation.get("execution_ready") is not False
        or recommendation.get("gpu_execution_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
        or recommendation.get("release_authorization_allowed") is not False
    ):
        raise ValueError("follow-up decision does not select the bounded capacity probe")
    failed = terminal.get("failed_checks")
    if (
        terminal.get("status") != "hold"
        or not isinstance(failed, list)
        or not failed
        or not set(failed).issubset(ABSOLUTE_QUALITY_CHECKS)
        or trend.get("shared_quality_trend_strictly_improved") is not True
    ):
        raise ValueError("capacity-probe scientific trigger differs")
    execution_git = _git(
        decision.get("quality_bridge_execution_git", {}),
        label="quality bridge execution",
    )
    if execution_git != {
        "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
        "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity probe decision binds another quality bridge")
    replay = decision.get("source_replay")
    if (
        not isinstance(replay, Mapping)
        or replay.get("quality_bridge_result_rebuilt_byte_equivalent") is not True
        or replay.get("physical_checkpoint_sample_and_real_set_reverified") is not True
        or replay.get("milestone_source_reports_reverified") is not True
    ):
        raise ValueError("capacity probe decision lacks complete source replay")
    return {
        "status": terminal["status"],
        "failed_checks": list(failed),
        "shared_50k_to_100k_fid_improvement": True,
        "recommendation_id": CAPACITY_PROBE_RECOMMENDATION_ID,
    }


def _validate_config_pair(
    report: Mapping[str, Any],
    *,
    stage: str,
    expected_parameters: Mapping[str, int],
) -> dict[str, Any]:
    recipe = report.get("training_recipe")
    cofitok = report.get("cofitok")
    dense = report.get("dense")
    if (
        report.get("status") != "pass"
        or report.get("mismatches") != []
        or not isinstance(recipe, Mapping)
        or recipe.get("valid") is not True
        or recipe.get("stage") != stage
        or not isinstance(cofitok, Mapping)
        or not isinstance(dense, Mapping)
    ):
        raise ValueError(f"{stage} matched config validation failed")
    observed = {
        "cofitok": int(cofitok.get("parameter_count", -1)),
        "dense_identity": int(dense.get("parameter_count", -1)),
    }
    if observed != dict(expected_parameters):
        raise ValueError(f"{stage} parameter counts differ")
    if abs(float(report.get("relative_parameter_gap", 1.0))) > 0.02:
        raise ValueError(f"{stage} parameter gap is not matched")
    return {
        "status": "pass",
        "recipe_stage": stage,
        "parameter_counts": observed,
        "relative_parameter_gap": float(report["relative_parameter_gap"]),
        "effective_batches": copy.deepcopy(recipe.get("effective_batches")),
        "recipe_schema": recipe.get("schema"),
    }


def _validate_reference(
    value: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    receipt = _identity(value.get("receipt", {}), label=f"{method} reference receipt")
    report = value.get("report")
    if not isinstance(report, Mapping):
        raise ValueError(f"{method} checkpoint reference report is missing")
    expected = report.get("expected")
    reference = report.get("reference")
    storage = report.get("storage")
    boundary = report.get("authorization_boundary")
    if (
        report.get("status") != "preserved"
        or report.get("role") != "generation_checkpoint_hardlink_reference"
        or report.get("reason") != CAPACITY_PROBE_REFERENCE_REASON
        or expected
        != {
            "step": CAPACITY_PROBE_STOP_STEP,
            "git": {
                "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
                "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
                "tracked_dirty": False,
            },
        }
        or not isinstance(reference, Mapping)
        or storage
        != {
            "same_filesystem": True,
            "checkpoint_same_inode": True,
            "integrity_same_inode": True,
            "additional_checkpoint_data_blocks_required": False,
        }
        or boundary
        != {
            "training_launch_allowed": False,
            "gpu_use_allowed": False,
            "checkpoint_mutation_allowed": False,
            "checkpoint_deletion_allowed": False,
            "full_300k_launch_allowed": False,
            "release_allowed": False,
        }
    ):
        raise ValueError(f"{method} checkpoint reference contract differs")
    checkpoint = _identity(
        reference.get("checkpoint", {}),
        label=f"{method} reference checkpoint",
    )
    integrity = _identity(
        reference.get("integrity", {}),
        label=f"{method} reference integrity",
    )
    expected_name = f"checkpoint_step_{CAPACITY_PROBE_STOP_STEP:08d}.pt"
    if Path(checkpoint["path"]).name != expected_name:
        raise ValueError(f"{method} checkpoint reference step filename differs")
    if Path(integrity["path"]).name != f"{expected_name}.integrity.json":
        raise ValueError(f"{method} checkpoint integrity filename differs")
    return {
        "receipt": receipt,
        "checkpoint": checkpoint,
        "integrity": integrity,
        "same_inode_preservation_verified": True,
    }


def build_capacity_probe_preparation(
    *,
    followup_decision: Mapping[str, Any],
    followup_decision_identity: Mapping[str, Any],
    quality_bridge_result_identity: Mapping[str, Any],
    quality_bridge_preparation_identity: Mapping[str, Any],
    base_configs: Mapping[str, Mapping[str, Any]],
    capacity_configs: Mapping[str, Mapping[str, Any]],
    base_config_identities: Mapping[str, Mapping[str, Any]],
    capacity_config_identities: Mapping[str, Mapping[str, Any]],
    base_config_validation: Mapping[str, Any],
    capacity_config_validation: Mapping[str, Any],
    checkpoint_references: Mapping[str, Mapping[str, Any]],
    preparation_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    trigger = _validate_decision(followup_decision)
    decision_identity = _identity(
        followup_decision_identity,
        label="quality bridge follow-up decision",
    )
    result_identity = _identity(
        quality_bridge_result_identity,
        label="quality bridge result",
    )
    bridge_preparation_identity = _identity(
        quality_bridge_preparation_identity,
        label="quality bridge preparation",
    )
    git = _git(preparation_git, label="capacity probe preparation")
    if not output_root or not (
        Path(output_root).is_absolute() or PurePosixPath(output_root).is_absolute()
    ):
        raise ValueError("capacity probe output root must be absolute")
    decision_source_result = followup_decision.get("source_reports", {}).get(
        "quality_bridge_result"
    )
    if decision_source_result != result_identity:
        raise ValueError("capacity probe decision binds another quality bridge result")

    methods = {"cofitok", "dense_identity"}
    inputs = (
        base_configs,
        capacity_configs,
        base_config_identities,
        capacity_config_identities,
        checkpoint_references,
    )
    if any(set(value) != methods for value in inputs):
        raise ValueError("capacity probe method set differs")

    base_validation = _validate_config_pair(
        base_config_validation,
        stage="stability_quality_bridge",
        expected_parameters=CAPACITY_PROBE_PARAMETER_COUNTS["base128"],
    )
    capacity_validation = _validate_config_pair(
        capacity_config_validation,
        stage="stability_capacity_probe",
        expected_parameters=CAPACITY_PROBE_PARAMETER_COUNTS["base256"],
    )

    differences: dict[str, list[str]] = {}
    normalized_base_identities: dict[str, Any] = {}
    normalized_capacity_identities: dict[str, Any] = {}
    references: dict[str, Any] = {}
    for method in sorted(methods):
        base = base_configs[method]
        capacity = capacity_configs[method]
        if not isinstance(base, Mapping) or not isinstance(capacity, Mapping):
            raise ValueError(f"{method} capacity-probe config is malformed")
        observed = _differences(base, capacity)
        if observed != CAPACITY_PROBE_ALLOWED_CONFIG_DIFFERENCES:
            raise ValueError(
                f"{method} capacity intervention differs: {sorted(observed)}"
            )
        if (
            base.get("data", {}).get("dataset") != CAPACITY_PROBE_DATASET
            or capacity.get("data", {}).get("dataset") != CAPACITY_PROBE_DATASET
            or int(base.get("runtime", {}).get("steps", -1))
            != CAPACITY_PROBE_CONFIGURED_STEPS
            or int(capacity.get("runtime", {}).get("steps", -1))
            != CAPACITY_PROBE_CONFIGURED_STEPS
            or int(base.get("model", {}).get("base_channels", -1))
            != CAPACITY_PROBE_BASE_CHANNELS
            or int(capacity.get("model", {}).get("base_channels", -1))
            != CAPACITY_PROBE_LARGE_CHANNELS
            or capacity.get("runtime", {}).get("protected_checkpoint_steps")
            != [CAPACITY_PROBE_STOP_STEP]
        ):
            raise ValueError(f"{method} capacity-probe selection differs")
        base_batch = int(base.get("data", {}).get("batch_size", 0)) * int(
            base.get("optimization", {}).get("gradient_accumulation_steps", 0)
        )
        capacity_batch = int(capacity.get("data", {}).get("batch_size", 0)) * int(
            capacity.get("optimization", {}).get(
                "gradient_accumulation_steps", 0
            )
        )
        if base_batch != capacity_batch or base_batch != CAPACITY_PROBE_EFFECTIVE_BATCH:
            raise ValueError(f"{method} capacity probe changes effective batch")
        differences[method] = sorted(observed)
        normalized_base_identities[method] = _identity(
            base_config_identities[method],
            label=f"{method} base config",
        )
        normalized_capacity_identities[method] = _identity(
            capacity_config_identities[method],
            label=f"{method} capacity config",
        )
        references[method] = _validate_reference(
            checkpoint_references[method],
            method=method,
        )

    return {
        "schema_version": CAPACITY_PROBE_PREPARATION_SCHEMA_VERSION,
        "status": "prepared",
        "role": CAPACITY_PROBE_PREPARATION_ROLE,
        "git": git,
        "source_evidence": {
            "followup_decision": decision_identity,
            "quality_bridge_result": result_identity,
            "quality_bridge_preparation": bridge_preparation_identity,
            "source_replayed": True,
            "scientific_trigger": trigger,
        },
        "selection": {
            "dataset": CAPACITY_PROBE_DATASET,
            "configured_training_horizon": CAPACITY_PROBE_CONFIGURED_STEPS,
            "stop_after_step": CAPACITY_PROBE_STOP_STEP,
            "effective_batch_size": CAPACITY_PROBE_EFFECTIVE_BATCH,
            "images_seen_per_training_arm": (
                CAPACITY_PROBE_STOP_STEP * CAPACITY_PROBE_EFFECTIVE_BATCH
            ),
            "base_channels": [
                CAPACITY_PROBE_BASE_CHANNELS,
                CAPACITY_PROBE_LARGE_CHANNELS,
            ],
            "output_root": output_root,
            "fresh_training_arms": ["base256_cofitok", "base256_dense_identity"],
            "preserved_reference_arms": [
                "base128_cofitok",
                "base128_dense_identity",
            ],
            "training_intervention": sorted(CAPACITY_PROBE_TRAINING_INTERVENTION),
            "logistical_differences": sorted(CAPACITY_PROBE_LOGISTICAL_DIFFERENCES),
            "rejected_alternatives": {
                "continue_base128_to_150k": "does_not_isolate_capacity",
                "launch_250m_300k": "not_authorized_by_a_bounded_capacity_probe",
            },
        },
        "matched_training_contract": {
            "base128": base_validation,
            "base256": capacity_validation,
            "base_configs": normalized_base_identities,
            "capacity_configs": normalized_capacity_identities,
            "recursive_config_differences": differences,
            "configured_100k_schedule_preserved_at_step_10k": True,
            "fresh_initialization_required": True,
            "resume_from_base128_forbidden": True,
        },
        "checkpoint_references": references,
        "evaluation_contract": {
            "arms": [
                "base128_cofitok",
                "base128_dense_identity",
                "base256_cofitok",
                "base256_dense_identity",
            ],
            "checkpoint_step": CAPACITY_PROBE_STOP_STEP,
            "weights": "ema",
            "sampler": "ddim",
            "sample_steps": 50,
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "samples_per_arm": CAPACITY_PROBE_SAMPLES_PER_ARM,
            "mechanism_images_per_cofitok_arm": CAPACITY_PROBE_MECHANISM_IMAGES,
            "balanced_modulo_class_schedule": True,
            "fixed_random_stream_across_arms": True,
            "primary_estimands": [
                "within_method_fid_delta_base256_minus_base128",
                "capacity_by_factorization_fid_interaction",
            ],
            "role": "non_claim_capacity_causal_diagnostic",
            "precision_recall_required": False,
        },
        "decision_contract": {
            "shared_strict_fid_improvement_is_capacity_support": True,
            "mechanism_invariants_must_remain_valid": True,
            "result_can_authorize_full_300k": False,
            "new_source_compatible_decision_required": True,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_PROBE_PREPARATION_BOUNDARY
        ),
    }
