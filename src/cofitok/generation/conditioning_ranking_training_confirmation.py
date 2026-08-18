from __future__ import annotations

import copy
import math
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
EXECUTION_RECEIPT_ROLE = (
    "generation_conditioning_ranking_four_arm_train5k_confirmation_execution_receipt"
)
IDLE_GPU_EVIDENCE_ROLE = (
    "generation_conditioning_ranking_four_arm_train5k_confirmation_idle_gpu_evidence"
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
CONFIG_RELATIVE_PATHS = {
    "control_cofitok": (
        "configs/generation/imagenet256_10pct_stability_rgbtail3_"
        "rollout_x0_u2_ema_teacher_k8_probe5k.json"
    ),
    "control_dense_identity": (
        "configs/generation/imagenet256_10pct_stability_"
        "rollout_x0_u2_ema_teacher_dense_probe5k.json"
    ),
    "ranked_cofitok": (
        "configs/generation/imagenet256_10pct_stability_rgbtail3_"
        "rollout_x0_u2_ema_teacher_classrank_k8_confirm5k.json"
    ),
    "ranked_dense_identity": (
        "configs/generation/imagenet256_10pct_stability_"
        "rollout_x0_u2_ema_teacher_classrank_dense_confirm5k.json"
    ),
}
RUNBOOK_RELATIVE_PATH = (
    "artifacts/runbooks/"
    "generation_conditioning_ranking_four_arm_train5k_confirmation_v1.sh"
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
EXECUTION_AUTHORIZATION_BOUNDARY = {
    "standing_authorization_required": True,
    "exact_shared_sampling_validation_required": True,
    "exact_preparation_required": True,
    "exact_revision_stage_output_config_and_runbook_binding_required": True,
    "independent_clean_checkout_required": True,
    "five_consecutive_idle_gpu_polls_required": True,
    "fresh_four_arm_training_required": True,
    "resume_from_1k_checkpoint_allowed": False,
    "steps_per_run": 5_000,
    "training_runs": list(RUN_NAMES),
    "gpu_execution_authorized": True,
    "training_allowed": True,
    "runbook_launch_count_maximum": 1,
    "automatic_runbook_relaunch_allowed": False,
    "sampling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_100k_or_300k_launch_allowed": False,
    "release_authorization_allowed": False,
    "unrelated_process_signaling_allowed": False,
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


def sampling_validation_route(report: Mapping[str, Any]) -> str:
    methods = report.get("methods")
    sampling_contract = report.get("sampling_contract")
    decision = report.get("decision")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != SOURCE_SAMPLING_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != SOURCE_SAMPLING_STAGE
        or report.get("output_root") != SOURCE_SAMPLING_OUTPUT_ROOT
        or report.get("git") != SOURCE_SAMPLING_GIT
        or report.get("claim_boundary") != SOURCE_CLAIM_BOUNDARY
        or not isinstance(methods, Mapping)
        or set(methods) != {"cofitok", "dense_identity"}
        or not isinstance(sampling_contract, Mapping)
        or sampling_contract.get("sample_count_per_arm") != 5_000
        or sampling_contract.get("sample_run_name")
        != "samples_5000_ddim50_cfg15"
        or not isinstance(sampling_contract.get("methods"), Mapping)
        or set(sampling_contract["methods"]) != {"cofitok", "dense_identity"}
        or not isinstance(decision, Mapping)
    ):
        return "invalid"
    method_passes: dict[str, bool] = {}
    for name in ("cofitok", "dense_identity"):
        row = methods.get(name)
        gates = row.get("gates") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or type(row.get("pass")) is not bool
            or not isinstance(gates, Mapping)
            or set(gates) != SOURCE_METHOD_GATES
            or any(type(gates[gate]) is not bool for gate in SOURCE_METHOD_GATES)
            or row["pass"] is not all(gates.values())
        ):
            return "invalid"
        method_passes[name] = row["pass"]
    shared = all(method_passes.values())
    if shared:
        action = "prepare_separately_bound_matched_5k_training_recipe_confirmation"
    elif any(method_passes.values()):
        action = "reject_shared_repair_due_method_asymmetry"
    else:
        action = "revise_training_time_semantic_alignment_objective"
    expected_decision = {
        "method_passes": method_passes,
        "shared_generated_class_alignment_recovery_supported": shared,
        "cofitok_specific_advantage_claim_allowed": False,
        "recommended_next_action": action,
    }
    if dict(decision) != expected_decision:
        return "invalid"
    if shared:
        validate_shared_sampling_recovery(report)
        return "selected"
    return "not_selected"


def validate_idle_gpu_evidence(
    report: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    expected_output_root: str,
    required_polls: int = 5,
) -> dict[str, Any]:
    observations = report.get("observations")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != IDLE_GPU_EVIDENCE_ROLE
        or report.get("status") != "pass"
        or report.get("stage") != STAGE
        or report.get("output_root") != expected_output_root
        or report.get("required_consecutive_idle_polls") != required_polls
        or _clean_git(report.get("git", {}), label="idle GPU evidence")
        != _clean_git(expected_git, label="expected training confirmation Git")
        or not isinstance(observations, list)
        or len(observations) != required_polls
    ):
        raise ValueError("training confirmation idle GPU evidence differs")
    previous = -math.inf
    for index, observation in enumerate(observations, start=1):
        if not isinstance(observation, Mapping):
            raise ValueError("training confirmation idle GPU observation is malformed")
        timestamp = observation.get("observed_at_unix")
        if (
            observation.get("poll_index") != index
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or float(timestamp) <= previous
            or observation.get("gpu_compute_pids") != []
        ):
            raise ValueError("training confirmation idle GPU observation differs")
        previous = float(timestamp)
    return copy.deepcopy(dict(report))


def _validate_preparation_for_execution(
    *,
    preparation: Mapping[str, Any],
    sampling_validation: Mapping[str, Any],
    sampling_validation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
    expected_git: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    expected_git_clean = _clean_git(
        expected_git,
        label="expected training confirmation Git",
    )
    source = validate_shared_sampling_recovery(sampling_validation)
    standing = validate_standing_experiment_authorization(standing_authorization)
    sampling_identity = _identity(
        sampling_validation_identity,
        label="conditioning-ranking 5K sampling validation",
    )
    standing_identity = _identity(
        standing_authorization_identity,
        label="standing experiment authorization",
    )
    if set(config_identities) != set(RUN_NAMES):
        raise ValueError("training confirmation execution config set differs")
    configs = {
        name: _identity(config_identities.get(name, {}), label=f"{name} config")
        for name in RUN_NAMES
    }
    counts = preparation.get("parameter_counts")
    if (
        preparation.get("schema_version") != SCHEMA_VERSION
        or preparation.get("role") != PREPARATION_ROLE
        or preparation.get("status") != "pass"
        or preparation.get("valid") is not True
        or preparation.get("stage") != STAGE
        or preparation.get("scope") != SCOPE
        or preparation.get("output_root") != expected_output_root
        or preparation.get("git") != expected_git_clean
        or preparation.get("source_reports")
        != {
            "sampling_validation": sampling_identity,
            "standing_authorization": standing_identity,
        }
        or preparation.get("source_sampling_decision") != source["decision"]
        or preparation.get("standing_authorization") != standing
        or preparation.get("configs") != configs
        or preparation.get("execution_boundary") != EXECUTION_BOUNDARY
        or preparation.get("claim_boundary") != CLAIM_BOUNDARY
        or preparation.get("gpu_execution_authorized") is not False
        or preparation.get("authorization_required") is not True
        or not isinstance(counts, Mapping)
        or set(counts) != set(RUN_NAMES)
        or any(type(counts[name]) is not int or counts[name] <= 0 for name in RUN_NAMES)
        or counts["control_cofitok"] != counts["ranked_cofitok"]
        or counts["control_dense_identity"] != counts["ranked_dense_identity"]
    ):
        raise ValueError("training confirmation preparation differs")
    return copy.deepcopy(dict(preparation))


def build_training_confirmation_execution_receipt(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    sampling_validation: Mapping[str, Any],
    sampling_validation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    idle_gpu_evidence: Mapping[str, Any],
    idle_gpu_evidence_identity: Mapping[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
    runbook_identity: Mapping[str, Any],
    receipt_git: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    git = _clean_git(receipt_git, label="training confirmation receipt builder")
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError("training confirmation receipt builder Git differs")
    if expected_output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError("training confirmation receipt output root differs")
    validated_preparation = _validate_preparation_for_execution(
        preparation=preparation,
        sampling_validation=sampling_validation,
        sampling_validation_identity=sampling_validation_identity,
        standing_authorization=standing_authorization,
        standing_authorization_identity=standing_authorization_identity,
        config_identities=config_identities,
        expected_git=expected_git,
        expected_output_root=expected_output_root,
    )
    idle = validate_idle_gpu_evidence(
        idle_gpu_evidence,
        expected_git=expected_git,
        expected_output_root=expected_output_root,
    )
    standing = validate_standing_experiment_authorization(standing_authorization)
    return {
        "schema_version": SCHEMA_VERSION,
        "role": EXECUTION_RECEIPT_ROLE,
        "status": "authorized",
        "authorization_mode": "active_standing_experiment_authorization",
        "stage": STAGE,
        "scope": SCOPE,
        "authorized_revision": expected_revision,
        "authorized_branch": expected_branch,
        "output_root": expected_output_root,
        "git": git,
        "source_reports": {
            "preparation": _identity(
                preparation_identity,
                label="training confirmation preparation",
            ),
            "sampling_validation": _identity(
                sampling_validation_identity,
                label="conditioning-ranking 5K sampling validation",
            ),
            "standing_authorization": _identity(
                standing_authorization_identity,
                label="standing experiment authorization",
            ),
            "idle_gpu_evidence": _identity(
                idle_gpu_evidence_identity,
                label="idle GPU evidence",
            ),
            "runbook": _identity(
                runbook_identity,
                label="training confirmation runbook",
            ),
            "configs": {
                name: _identity(
                    config_identities[name],
                    label=f"{name} config",
                )
                for name in RUN_NAMES
            },
        },
        "standing_authorization": standing,
        "source_sampling_decision": copy.deepcopy(
            validated_preparation["source_sampling_decision"]
        ),
        "parameter_counts": copy.deepcopy(
            validated_preparation["parameter_counts"]
        ),
        "idle_gpu_evidence": idle,
        "authorization_boundary": copy.deepcopy(
            EXECUTION_AUTHORIZATION_BOUNDARY
        ),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
    }


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
