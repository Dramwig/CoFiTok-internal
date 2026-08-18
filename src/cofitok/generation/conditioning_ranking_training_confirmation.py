from __future__ import annotations

import copy
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_probe import (
    CONTROL_RANKING_CONFIG,
    RANKING_FIELDS,
    validate_standing_experiment_authorization,
)
from cofitok.generation_pair import generation_pair_contract


SCHEMA_VERSION = 1
STAGE = "conditioning_ranking_four_arm_train5k_confirmation_v1"
SCOPE = "imagenet256_10pct_four_arm_class_ranking_train5k_confirmation_only"
PREPARATION_ROLE = (
    "generation_conditioning_ranking_four_arm_train5k_confirmation_preparation"
)
EXPECTED_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_train5k_confirmation_v1"
)
SOURCE_SAMPLING_ROLE = (
    "generation_conditioning_ranking_four_arm_sampling_validation"
)
SOURCE_SAMPLING_STAGE = "conditioning_ranking_four_arm_sampling5k_v1"
SOURCE_SAMPLING_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_sampling5k_v1"
)
SOURCE_SAMPLING_GIT = {
    "revision": "f77e311546beba697020debd35e26888096ca258",
    "branch": "scale/generation-label-ranking-standing-authorization-v1",
    "tracked_dirty": False,
}
SOURCE_DECISION = {
    "method_passes": {"cofitok": True, "dense_identity": True},
    "shared_generated_class_alignment_recovery_supported": True,
    "cofitok_specific_advantage_claim_allowed": False,
    "recommended_next_action": (
        "prepare_separately_bound_matched_5k_training_recipe_confirmation"
    ),
}
SOURCE_CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "formal_quality_gate": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_full_training": False,
    "authorizes_full_100k_or_300k": False,
    "authorizes_release": False,
    "replaces_active_quality_bridge": False,
    "broad_generation_superiority_claim_allowed": False,
    "cofitok_specific_advantage_claim_allowed": False,
}
SOURCE_METHOD_GATES = {
    "exact_provenance",
    "fid_within_tolerance",
    "mean_target_log_probability_delta",
    "paired_target_log_probability_significant",
    "predicted_class_fraction_within_tolerance",
    "target_probability_ratio",
    "top1_not_regressed",
    "top5_not_regressed",
}
RANKED_RANKING_CONFIG = {
    "class_conditioning_ranking_weight": 0.05,
    "class_conditioning_ranking_start_step": 500,
    "class_conditioning_ranking_warmup_steps": 1_000,
    "class_conditioning_ranking_batch_fraction": 0.0625,
    "class_conditioning_ranking_margin": 0.01,
    "class_conditioning_ranking_wrong_label_offset": 500,
    "class_conditioning_ranking_min_timestep": 500,
}
RUN_NAMES = (
    "control_cofitok",
    "ranked_cofitok",
    "control_dense_identity",
    "ranked_dense_identity",
)
EXECUTION_BOUNDARY = {
    "preparation_only": True,
    "standing_authorization_required": True,
    "exact_shared_sampling_validation_required": True,
    "exact_revision_stage_and_output_binding_required": True,
    "independent_clean_checkout_required": True,
    "fresh_four_arm_training_required": True,
    "resume_from_1k_checkpoint_allowed": False,
    "steps_per_run": 5_000,
    "training_runs": list(RUN_NAMES),
    "gpu_execution_authorized": False,
    "training_allowed": False,
    "sampling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_100k_or_300k_launch_allowed": False,
    "release_authorization_allowed": False,
    "unrelated_process_signaling_allowed": False,
    "required_next_evidence": "separately_source_bound_execution_receipt",
}
CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "formal_quality_gate": False,
    "training_quality_claim_allowed": False,
    "sample_quality_claim_allowed": False,
    "cofitok_specific_advantage_claim_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_full_training": False,
    "authorizes_full_100k_or_300k": False,
    "authorizes_release": False,
    "replaces_active_quality_bridge": False,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _is_sha256(digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _clean_git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or not all(character in "0123456789abcdef" for character in revision)
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


def _ranking_config(config: Mapping[str, Any]) -> dict[str, Any]:
    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        return {}
    return {field: loss.get(field) for field in RANKING_FIELDS}


def _without_ranking_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized.pop("name", None)
    loss = normalized.get("loss")
    if isinstance(loss, dict):
        for field in RANKING_FIELDS:
            loss.pop(field, None)
    return normalized


def validate_shared_sampling_recovery(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        report.get("schema_version") != 1
        or report.get("role") != SOURCE_SAMPLING_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != SOURCE_SAMPLING_STAGE
        or report.get("output_root") != SOURCE_SAMPLING_OUTPUT_ROOT
        or report.get("git") != SOURCE_SAMPLING_GIT
        or report.get("decision") != SOURCE_DECISION
        or report.get("claim_boundary") != SOURCE_CLAIM_BOUNDARY
    ):
        raise ValueError(
            "conditioning-ranking 5K sampling validation did not select "
            "matched training confirmation"
        )
    methods = report.get("methods")
    sampling_contract = report.get("sampling_contract")
    if (
        not isinstance(methods, Mapping)
        or set(methods) != {"cofitok", "dense_identity"}
        or any(
            not isinstance(methods[name], Mapping)
            or methods[name].get("pass") is not True
            or not isinstance(methods[name].get("gates"), Mapping)
            or set(methods[name]["gates"]) != SOURCE_METHOD_GATES
            or any(
                methods[name]["gates"][gate] is not True
                for gate in SOURCE_METHOD_GATES
            )
            for name in ("cofitok", "dense_identity")
        )
        or not isinstance(sampling_contract, Mapping)
        or sampling_contract.get("sample_count_per_arm") != 5_000
        or sampling_contract.get("sample_run_name")
        != "samples_5000_ddim50_cfg15"
        or not isinstance(sampling_contract.get("methods"), Mapping)
        or set(sampling_contract["methods"]) != {"cofitok", "dense_identity"}
    ):
        raise ValueError("conditioning-ranking shared sampling evidence is malformed")
    return copy.deepcopy(dict(report))


def training_confirmation_contract(
    *,
    control_cofitok: dict[str, Any],
    control_dense: dict[str, Any],
    ranked_cofitok: dict[str, Any],
    ranked_dense: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    pairs = {
        "control": generation_pair_contract(control_cofitok, control_dense),
        "ranked": generation_pair_contract(ranked_cofitok, ranked_dense),
    }
    for name, contract in pairs.items():
        issues.extend(f"{name}_pair: {issue}" for issue in contract["issues"])

    for label, config, expected_ranking in (
        ("control_cofitok", control_cofitok, CONTROL_RANKING_CONFIG),
        ("control_dense", control_dense, CONTROL_RANKING_CONFIG),
        ("ranked_cofitok", ranked_cofitok, RANKED_RANKING_CONFIG),
        ("ranked_dense", ranked_dense, RANKED_RANKING_CONFIG),
    ):
        if _ranking_config(config) != expected_ranking:
            issues.append(f"{label} ranking config differs")
        data = config.get("data", {})
        runtime = config.get("runtime", {})
        optimization = config.get("optimization", {})
        if runtime.get("steps") != 5_000:
            issues.append(f"{label} runtime.steps is not 5000")
        if runtime.get("checkpoint_interval") != 1_250:
            issues.append(f"{label} checkpoint interval is not 1250")
        if runtime.get("evaluation_interval") != 1_250:
            issues.append(f"{label} evaluation interval is not 1250")
        if runtime.get("protected_checkpoint_steps") != [1_250, 2_500, 5_000]:
            issues.append(f"{label} protected checkpoint steps differ")
        if data.get("dataset") != "imagenet_256_10pct":
            issues.append(f"{label} dataset is not imagenet_256_10pct")
        if int(data.get("batch_size", 0)) * int(
            optimization.get("gradient_accumulation_steps", 0)
        ) != 64:
            issues.append(f"{label} effective batch size is not 64")

    for method, control, ranked in (
        ("cofitok", control_cofitok, ranked_cofitok),
        ("dense_identity", control_dense, ranked_dense),
    ):
        if _without_ranking_fields(control) != _without_ranking_fields(ranked):
            issues.append(
                f"{method} control/ranked configs differ outside ranking fields"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "role": PREPARATION_ROLE,
        "status": "pass" if not issues else "fail",
        "valid": not issues,
        "stage": STAGE,
        "scope": SCOPE,
        "issues": issues,
        "ranking_fields": list(RANKING_FIELDS),
        "control_ranking_config": copy.deepcopy(CONTROL_RANKING_CONFIG),
        "ranked_ranking_config": copy.deepcopy(RANKED_RANKING_CONFIG),
        "pair_contracts": pairs,
        "execution_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
    }


def build_training_confirmation_preparation(
    *,
    sampling_validation: Mapping[str, Any],
    sampling_validation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    control_cofitok: dict[str, Any],
    control_dense: dict[str, Any],
    ranked_cofitok: dict[str, Any],
    ranked_dense: dict[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
    parameter_counts: Mapping[str, int],
    builder_git: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    git = _clean_git(builder_git, label="training confirmation builder")
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError("training confirmation builder Git differs")
    if expected_output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError("training confirmation output root differs")
    source = validate_shared_sampling_recovery(sampling_validation)
    standing = validate_standing_experiment_authorization(standing_authorization)
    contract = training_confirmation_contract(
        control_cofitok=control_cofitok,
        control_dense=control_dense,
        ranked_cofitok=ranked_cofitok,
        ranked_dense=ranked_dense,
    )
    if contract["valid"] is not True:
        raise ValueError(
            "training confirmation config contract failed: "
            + "; ".join(contract["issues"])
        )
    if set(config_identities) != set(RUN_NAMES) or set(parameter_counts) != set(
        RUN_NAMES
    ):
        raise ValueError("training confirmation config evidence set differs")
    configs = {
        name: _identity(config_identities[name], label=f"{name} config")
        for name in RUN_NAMES
    }
    if any(type(parameter_counts[name]) is not int for name in RUN_NAMES):
        raise ValueError("training confirmation parameter count must be an integer")
    counts = {name: parameter_counts[name] for name in RUN_NAMES}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("training confirmation parameter count is invalid")
    if counts["control_cofitok"] != counts["ranked_cofitok"]:
        raise ValueError("CoFiTok control/ranked parameter counts differ")
    if counts["control_dense_identity"] != counts["ranked_dense_identity"]:
        raise ValueError("dense control/ranked parameter counts differ")
    return {
        **contract,
        "output_root": expected_output_root,
        "git": git,
        "source_reports": {
            "sampling_validation": _identity(
                sampling_validation_identity,
                label="conditioning-ranking 5K sampling validation",
            ),
            "standing_authorization": _identity(
                standing_authorization_identity,
                label="standing experiment authorization",
            ),
        },
        "source_sampling_decision": copy.deepcopy(source["decision"]),
        "standing_authorization": standing,
        "configs": configs,
        "parameter_counts": counts,
        "gpu_execution_authorized": False,
        "authorization_required": True,
        "authorization_source": (
            "active_standing_experiment_authorization_plus_separately_"
            "source_bound_execution_receipt"
        ),
    }
