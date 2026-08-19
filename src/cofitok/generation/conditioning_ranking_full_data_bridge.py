from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    CLAIM_BOUNDARY as POSTTRAINING_CLAIM_BOUNDARY,
    EXPECTED_OUTPUT_ROOT as POSTTRAINING_OUTPUT_ROOT,
    STAGE as POSTTRAINING_STAGE,
)
from cofitok.generation.conditioning_ranking_probe import (
    RANKING_FIELDS,
    validate_class_conditioning_followup_decision,
    validate_standing_experiment_authorization,
)
from cofitok.generation_pair import generation_pair_contract


SCHEMA_VERSION = 1
STAGE = "conditioning_ranking_full_data_100k_bridge_v1"
SCOPE = "imagenet256_full_data_fresh_ranked_matched_100k_bridge_only"
PREPARATION_ROLE = "generation_conditioning_ranking_full_data_100k_preparation"
AUTHORIZATION_SOURCE = (
    "active_standing_experiment_authorization_plus_separately_"
    "source_bound_execution_receipt"
)
PARAMETER_RELATIVE_GAP_LIMIT = 0.02
EXPECTED_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_conditioning_ranked_bridge_v1"
)
POSTTRAINING_REPORT_ROLE = (
    "generation_conditioning_ranking_four_arm_posttraining_sampling_confirmation"
)
POSTTRAINING_GIT = {
    "revision": "5cc51b2bef473b49bd90e95399cf3086c7dbed54",
    "branch": "scale/generation-label-ranking-posttraining-5k-sampling-confirmation-v1",
    "tracked_dirty": False,
}
POSTTRAINING_DECISION = {
    "method_passes": {"cofitok": True, "dense_identity": True},
    "shared_posttraining_generated_class_alignment_recovery_confirmed": True,
    "cofitok_specific_advantage_claim_allowed": False,
    "recommended_next_action": (
        "retain_shared_conditioning_ranking_recipe_for_separately_"
        "authorized_future_scaling"
    ),
}
RANKED_RANKING_CONFIG = {
    "class_conditioning_ranking_weight": 0.05,
    "class_conditioning_ranking_start_step": 10_000,
    "class_conditioning_ranking_warmup_steps": 20_000,
    "class_conditioning_ranking_batch_fraction": 0.0625,
    "class_conditioning_ranking_margin": 0.01,
    "class_conditioning_ranking_wrong_label_offset": 500,
    "class_conditioning_ranking_min_timestep": 500,
}
BASE_CONFIG_RELATIVE_PATHS = {
    "cofitok": (
        "configs/generation/imagenet256_stability_quality_bridge_rgbtail3_"
        "rollout_x0_u2_ema_teacher_k8_100k.json"
    ),
    "dense_identity": (
        "configs/generation/imagenet256_stability_quality_bridge_"
        "rollout_x0_u2_ema_teacher_dense_100k.json"
    ),
}
RANKED_CONFIG_RELATIVE_PATHS = {
    "cofitok": (
        "configs/generation/imagenet256_stability_conditioning_ranked_bridge_"
        "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
    ),
    "dense_identity": (
        "configs/generation/imagenet256_stability_conditioning_ranked_bridge_"
        "rollout_x0_u2_ema_teacher_dense_100k.json"
    ),
}
PLANNED_EXECUTION = {
    "dataset": "imagenet_256",
    "training_runs": ["cofitok", "dense_identity"],
    "steps_per_run": 100_000,
    "effective_batch_size": 64,
    "fresh_start_required": True,
    "resume_from_existing_quality_bridge_checkpoint_allowed": False,
    "source_quality_bridge_checkpoints_are_training_inputs": False,
    "serial_single_gpu_execution_required": True,
    "checkpoint_steps": [50_000, 100_000],
    "terminal_matched_sampling_required": True,
}
PREPARATION_BOUNDARY = {
    "standing_authorization_required": True,
    "exact_quality_bridge_conditioning_route_required": True,
    "exact_shared_posttraining_5k_pass_required": True,
    "independent_clean_checkout_required": True,
    "exact_revision_stage_config_and_output_binding_required": True,
    "separate_execution_receipt_required": True,
    "five_consecutive_idle_gpu_polls_required": True,
    "gpu_execution_authorized": False,
    "training_allowed": False,
    "sampling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
    "unrelated_process_signaling_allowed": False,
}
CLAIM_BOUNDARY = {
    "diagnostic_bridge_only": True,
    "replaces_active_quality_bridge": False,
    "existing_100k_checkpoint_repair_claim_allowed": False,
    "five_k_result_is_full_data_evidence": False,
    "broad_generation_superiority_claim_allowed": False,
    "cofitok_specific_advantage_claim_allowed": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_full_300k": False,
    "authorizes_release": False,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
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


def clean_git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
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
    return {"revision": revision, "branch": branch, "tracked_dirty": False}


def validate_posttraining_sampling_confirmation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    methods = report.get("methods")
    sources = report.get("sources")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != POSTTRAINING_REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != POSTTRAINING_STAGE
        or report.get("output_root") != POSTTRAINING_OUTPUT_ROOT
        or report.get("git") != POSTTRAINING_GIT
        or report.get("decision") != POSTTRAINING_DECISION
        or report.get("claim_boundary") != POSTTRAINING_CLAIM_BOUNDARY
        or not isinstance(methods, Mapping)
        or set(methods) != {"cofitok", "dense_identity"}
        or not isinstance(sources, Mapping)
    ):
        raise ValueError("posttraining ranking confirmation did not select scaling")
    for method in ("cofitok", "dense_identity"):
        method_report = methods.get(method)
        if (
            not isinstance(method_report, Mapping)
            or method_report.get("pass") is not True
        ):
            raise ValueError(f"posttraining ranking method did not pass: {method}")
    return copy.deepcopy(dict(report))


def _ranking_config(config: Mapping[str, Any]) -> dict[str, Any]:
    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        return {}
    return {field: loss.get(field) for field in RANKING_FIELDS}


def _without_name_and_ranking(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized.pop("name", None)
    loss = normalized.get("loss")
    if isinstance(loss, dict):
        for field in RANKING_FIELDS:
            loss.pop(field, None)
    return normalized


def _effective_batch(config: Mapping[str, Any]) -> int:
    data = config.get("data")
    optimization = config.get("optimization")
    if not isinstance(data, Mapping) or not isinstance(optimization, Mapping):
        return -1
    return int(data.get("batch_size", 0)) * int(
        optimization.get("gradient_accumulation_steps", 0)
    )


def full_data_ranked_bridge_contract(
    *,
    base_cofitok: dict[str, Any],
    base_dense_identity: dict[str, Any],
    ranked_cofitok: dict[str, Any],
    ranked_dense_identity: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    pair_contracts = {
        "base": generation_pair_contract(base_cofitok, base_dense_identity),
        "ranked": generation_pair_contract(
            ranked_cofitok, ranked_dense_identity
        ),
    }
    for name, contract in pair_contracts.items():
        issues.extend(f"{name}_pair: {issue}" for issue in contract["issues"])

    for method, base, ranked in (
        ("cofitok", base_cofitok, ranked_cofitok),
        ("dense_identity", base_dense_identity, ranked_dense_identity),
    ):
        if _ranking_config(ranked) != RANKED_RANKING_CONFIG:
            issues.append(f"{method} ranked loss schedule differs")
        if _without_name_and_ranking(base) != _without_name_and_ranking(ranked):
            issues.append(
                f"{method} differs from its base recipe outside ranking fields"
            )
        for label, config in (("base", base), ("ranked", ranked)):
            data = config.get("data", {})
            runtime = config.get("runtime", {})
            if data.get("dataset") != "imagenet_256":
                issues.append(f"{method} {label} dataset is not imagenet_256")
            if runtime.get("steps") != 100_000:
                issues.append(f"{method} {label} runtime.steps is not 100000")
            if _effective_batch(config) != 64:
                issues.append(f"{method} {label} effective batch is not 64")

    return {
        "schema_version": SCHEMA_VERSION,
        "role": PREPARATION_ROLE,
        "stage": STAGE,
        "scope": SCOPE,
        "status": "pass" if not issues else "fail",
        "valid": not issues,
        "issues": issues,
        "ranking_fields": list(RANKING_FIELDS),
        "ranked_ranking_config": copy.deepcopy(RANKED_RANKING_CONFIG),
        "pair_contracts": pair_contracts,
        "planned_execution": copy.deepcopy(PLANNED_EXECUTION),
        "preparation_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
    }


def build_full_data_ranked_bridge_preparation(
    *,
    posttraining_confirmation: Mapping[str, Any],
    posttraining_confirmation_identity: Mapping[str, Any],
    quality_bridge_followup: Mapping[str, Any],
    quality_bridge_followup_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    base_cofitok: dict[str, Any],
    base_dense_identity: dict[str, Any],
    ranked_cofitok: dict[str, Any],
    ranked_dense_identity: dict[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
    parameter_counts: Mapping[str, int],
    builder_git: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    git = clean_git(builder_git, label="ranked full-data bridge builder")
    if git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("ranked full-data bridge builder Git differs")
    if expected_output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError("ranked full-data bridge output root differs")

    posttraining = validate_posttraining_sampling_confirmation(
        posttraining_confirmation
    )
    followup = validate_class_conditioning_followup_decision(
        quality_bridge_followup
    )
    standing = validate_standing_experiment_authorization(standing_authorization)
    contract = full_data_ranked_bridge_contract(
        base_cofitok=base_cofitok,
        base_dense_identity=base_dense_identity,
        ranked_cofitok=ranked_cofitok,
        ranked_dense_identity=ranked_dense_identity,
    )
    if not contract["valid"]:
        raise ValueError(
            "ranked full-data bridge config contract failed: "
            + "; ".join(contract["issues"])
        )

    expected_config_names = {
        "base_cofitok",
        "base_dense_identity",
        "ranked_cofitok",
        "ranked_dense_identity",
    }
    if set(config_identities) != expected_config_names:
        raise ValueError("ranked full-data bridge config identity set differs")
    configs = {
        name: identity(value, label=f"{name} config")
        for name, value in config_identities.items()
    }
    if set(parameter_counts) != expected_config_names or any(
        type(value) is not int or value <= 0 for value in parameter_counts.values()
    ):
        raise ValueError("ranked full-data bridge parameter counts are invalid")
    counts = dict(parameter_counts)
    if counts["base_cofitok"] != counts["ranked_cofitok"]:
        raise ValueError("ranked CoFiTok parameter count differs from base")
    if counts["base_dense_identity"] != counts["ranked_dense_identity"]:
        raise ValueError("ranked dense parameter count differs from base")
    parameter_relative_gap = abs(
        counts["base_cofitok"] - counts["base_dense_identity"]
    ) / counts["base_dense_identity"]
    if parameter_relative_gap > PARAMETER_RELATIVE_GAP_LIMIT:
        raise ValueError("ranked full-data bridge parameter gap exceeds 2%")

    return {
        **contract,
        "output_root": expected_output_root,
        "git": git,
        "source_reports": {
            "posttraining_sampling_confirmation": identity(
                posttraining_confirmation_identity,
                label="posttraining sampling confirmation",
            ),
            "quality_bridge_followup_decision": identity(
                quality_bridge_followup_identity,
                label="quality-bridge follow-up decision",
            ),
            "quality_bridge_result": followup["quality_bridge_result"],
            "standing_authorization": identity(
                standing_authorization_identity,
                label="standing experiment authorization",
            ),
            "configs": configs,
        },
        "source_decisions": {
            "quality_bridge_failed_checks": followup["failed_checks"],
            "posttraining_sampling": copy.deepcopy(posttraining["decision"]),
        },
        "standing_authorization": standing,
        "parameter_counts": counts,
        "parameter_relative_gap": parameter_relative_gap,
        "gpu_execution_authorized": False,
        "authorization_required": True,
        "authorization_source": AUTHORIZATION_SOURCE,
    }


def validate_full_data_ranked_bridge_preparation(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str = EXPECTED_OUTPUT_ROOT,
) -> dict[str, Any]:
    pair_contracts = report.get("pair_contracts")
    source_reports = report.get("source_reports")
    source_decisions = report.get("source_decisions")
    parameter_counts = report.get("parameter_counts")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != PREPARATION_ROLE
        or report.get("stage") != STAGE
        or report.get("scope") != SCOPE
        or report.get("status") != "pass"
        or report.get("valid") is not True
        or report.get("issues") != []
        or report.get("output_root") != expected_output_root
        or report.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or report.get("planned_execution") != PLANNED_EXECUTION
        or report.get("preparation_boundary") != PREPARATION_BOUNDARY
        or report.get("claim_boundary") != CLAIM_BOUNDARY
        or report.get("ranking_fields") != list(RANKING_FIELDS)
        or report.get("ranked_ranking_config") != RANKED_RANKING_CONFIG
        or not isinstance(pair_contracts, Mapping)
        or set(pair_contracts) != {"base", "ranked"}
        or any(
            not isinstance(contract, Mapping)
            or contract.get("valid") is not True
            or contract.get("issues") != []
            for contract in pair_contracts.values()
        )
        or not isinstance(source_reports, Mapping)
        or set(source_reports)
        != {
            "posttraining_sampling_confirmation",
            "quality_bridge_followup_decision",
            "quality_bridge_result",
            "standing_authorization",
            "configs",
        }
        or not isinstance(source_decisions, Mapping)
        or source_decisions.get("posttraining_sampling") != POSTTRAINING_DECISION
        or not isinstance(
            source_decisions.get("quality_bridge_failed_checks"), list
        )
        or "class_fidelity"
        not in source_decisions.get("quality_bridge_failed_checks", [])
        or not isinstance(parameter_counts, Mapping)
        or set(parameter_counts)
        != {
            "base_cofitok",
            "base_dense_identity",
            "ranked_cofitok",
            "ranked_dense_identity",
        }
        or report.get("gpu_execution_authorized") is not False
        or report.get("authorization_required") is not True
        or report.get("authorization_source") != AUTHORIZATION_SOURCE
    ):
        raise ValueError("ranked full-data bridge preparation differs")

    for name in (
        "posttraining_sampling_confirmation",
        "quality_bridge_followup_decision",
        "quality_bridge_result",
        "standing_authorization",
    ):
        identity(source_reports[name], label=f"{name} source")
    config_sources = source_reports.get("configs")
    if not isinstance(config_sources, Mapping) or set(config_sources) != set(
        parameter_counts
    ):
        raise ValueError("ranked full-data bridge config sources differ")
    for name, value in config_sources.items():
        identity(value, label=f"{name} config")

    validate_standing_experiment_authorization(
        report.get("standing_authorization", {})
    )
    if any(
        type(value) is not int or value <= 0
        for value in parameter_counts.values()
    ):
        raise ValueError("ranked full-data bridge parameter counts differ")
    if (
        parameter_counts["base_cofitok"] != parameter_counts["ranked_cofitok"]
        or parameter_counts["base_dense_identity"]
        != parameter_counts["ranked_dense_identity"]
    ):
        raise ValueError("ranked full-data bridge parameter counts differ")
    parameter_relative_gap = abs(
        parameter_counts["base_cofitok"]
        - parameter_counts["base_dense_identity"]
    ) / parameter_counts["base_dense_identity"]
    if (
        parameter_relative_gap > PARAMETER_RELATIVE_GAP_LIMIT
        or not math.isclose(
            report.get("parameter_relative_gap", math.nan),
            parameter_relative_gap,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("ranked full-data bridge parameter gap differs")
    return copy.deepcopy(dict(report))
