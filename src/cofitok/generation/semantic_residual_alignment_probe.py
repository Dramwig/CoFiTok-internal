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
PREPARATION_ROLE = "generation_semantic_residual_alignment_probe_preparation"
AUTHORIZATION_ROLE = "generation_semantic_residual_alignment_execution_authorization"
TRAINING_STATUS_ROLE = "generation_semantic_residual_alignment_four_arm_training_status"
POSTEVALUATION_ROLE = "generation_semantic_residual_alignment_four_arm_postevaluation"
SUPERVISOR_ROLE = "generation_semantic_residual_alignment_probe_supervisor"
SCOPE = "imagenet256_10pct_four_arm_semantic_residual_alignment_probe1k_only"
STAGE = "semantic_residual_alignment_four_arm_probe1k_v1"
OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "semantic_residual_alignment_four_arm_probe1k_v1"
)
SOURCE_POSTEVALUATION_PATH = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_probe1k_terminal_rebind_v2/reports/"
    "conditioning_ranking_posteval_v1/postevaluation.json"
)
QUALITY_BRIDGE_RESULT_PATH = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_base128_quality_bridge_v1/reports/"
    "quality_bridge_result.json"
)
EXPECTED_SOURCE_SHA256 = {
    "source_postevaluation": (
        "97f33bf774ad4537a674a2d37af3e50f7f03b6e1283f7644f6c629be430a08bb"
    ),
    "quality_bridge_result": (
        "15752e05611fa888e15352934c1627ccb95418df0339f9ad120e195bc088d165"
    ),
}
SOURCE_POSTEVALUATION_GIT = {
    "revision": "d019a17b742c5da867e71fa72b2d67f75cb0bc7c",
    "branch": "analysis/generation-conditioning-ranking-terminal-rebind-v2-20260824",
    "tracked_dirty": False,
}
QUALITY_BRIDGE_GIT = {
    "revision": "cf0e5faa94bf4ab38d947b921935b3b765b5537a",
    "branch": "scale/generation-stability-quality-bridge-100k",
    "tracked_dirty": False,
}
EXPECTED_FAILED_CHECKS = [
    "cofitok_absolute_fid",
    "cofitok_recall_floor",
    "class_fidelity",
]
RUN_NAMES = (
    "control_cofitok",
    "residual_cofitok",
    "control_dense_identity",
    "residual_dense_identity",
)
METHOD_ARMS = {
    "cofitok": ("control_cofitok", "residual_cofitok"),
    "dense_identity": ("control_dense_identity", "residual_dense_identity"),
}
PREPARATION_CONFIG_KEYS = {
    "control_cofitok": "control_cofitok",
    "residual_cofitok": "residual_cofitok",
    "control_dense_identity": "control_dense",
    "residual_dense_identity": "residual_dense",
}
CONFIG_PATHS = {
    "control_cofitok": (
        "configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_"
        "u2_ema_teacher_k8_probe1k.json"
    ),
    "residual_cofitok": (
        "configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_"
        "u2_ema_teacher_classresidualalign_k8_probe1k.json"
    ),
    "control_dense_identity": (
        "configs/generation/imagenet256_10pct_stability_rollout_x0_u2_"
        "ema_teacher_dense_probe1k.json"
    ),
    "residual_dense_identity": (
        "configs/generation/imagenet256_10pct_stability_rollout_x0_u2_"
        "ema_teacher_classresidualalign_dense_probe1k.json"
    ),
}
RESIDUAL_ALIGNMENT_FIELDS = (
    "class_conditioning_residual_alignment_weight",
    "class_conditioning_residual_alignment_start_step",
    "class_conditioning_residual_alignment_warmup_steps",
    "class_conditioning_residual_alignment_batch_fraction",
    "class_conditioning_residual_alignment_margin",
    "class_conditioning_residual_alignment_temperature",
    "class_conditioning_residual_alignment_wrong_label_offsets",
    "class_conditioning_residual_alignment_min_timestep",
    "class_conditioning_residual_alignment_pooling_factors",
    "class_conditioning_residual_alignment_reconstruction_weight",
)
CONTROL_RESIDUAL_ALIGNMENT_CONFIG = {
    "class_conditioning_residual_alignment_weight": 0.0,
    "class_conditioning_residual_alignment_start_step": 0,
    "class_conditioning_residual_alignment_warmup_steps": 0,
    "class_conditioning_residual_alignment_batch_fraction": 0.0625,
    "class_conditioning_residual_alignment_margin": 0.0,
    "class_conditioning_residual_alignment_temperature": 0.1,
    "class_conditioning_residual_alignment_wrong_label_offsets": [1],
    "class_conditioning_residual_alignment_min_timestep": 0,
    "class_conditioning_residual_alignment_pooling_factors": [8, 16],
    "class_conditioning_residual_alignment_reconstruction_weight": 0.0,
}
CANDIDATE_RESIDUAL_ALIGNMENT_CONFIG = {
    "class_conditioning_residual_alignment_weight": 0.05,
    "class_conditioning_residual_alignment_start_step": 100,
    "class_conditioning_residual_alignment_warmup_steps": 200,
    "class_conditioning_residual_alignment_batch_fraction": 0.0625,
    "class_conditioning_residual_alignment_margin": 0.1,
    "class_conditioning_residual_alignment_temperature": 0.1,
    "class_conditioning_residual_alignment_wrong_label_offsets": [1, 500],
    "class_conditioning_residual_alignment_min_timestep": 500,
    "class_conditioning_residual_alignment_pooling_factors": [8, 16, 32],
    "class_conditioning_residual_alignment_reconstruction_weight": 0.25,
}
EXECUTION_BOUNDARY = {
    "training_allowed": True,
    "training_runs": list(RUN_NAMES),
    "steps_per_run": 1_000,
    "dataset": "imagenet_256_10pct",
    "required_checkpoint_steps": [500, 750, 1_000],
    "held_out_cpu_evaluation_allowed": True,
    "generation_sampling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "unrelated_process_signaling_allowed": False,
}
CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_quality_claim_allowed": False,
    "sample_quality_claim_allowed": False,
    "cofitok_specific_advantage_claim_allowed": False,
    "generation_advantage_proven": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "process_signal_allowed": False,
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


def _clean_git(
    value: Mapping[str, Any],
    *,
    label: str,
    include_tree: bool,
) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    tree = value.get("tree")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or not all(character in "0123456789abcdef" for character in revision)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
        or (
            include_tree
            and (
                not isinstance(tree, str)
                or len(tree) != 40
                or not all(character in "0123456789abcdef" for character in tree)
            )
        )
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    result = {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }
    if include_tree:
        result["tree"] = tree
    return result


def _residual_alignment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        return {}
    return {field: copy.deepcopy(loss.get(field)) for field in RESIDUAL_ALIGNMENT_FIELDS}


def _without_residual_alignment_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized.pop("name", None)
    loss = normalized.get("loss")
    if isinstance(loss, dict):
        for field in RESIDUAL_ALIGNMENT_FIELDS:
            loss.pop(field, None)
    return normalized


def semantic_residual_alignment_probe_contract(
    *,
    control_cofitok: dict[str, Any],
    control_dense: dict[str, Any],
    residual_cofitok: dict[str, Any],
    residual_dense: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    pair_contracts = {
        "control": generation_pair_contract(control_cofitok, control_dense),
        "residual_alignment": generation_pair_contract(
            residual_cofitok,
            residual_dense,
        ),
    }
    for name, contract in pair_contracts.items():
        issues.extend(f"{name}_pair: {issue}" for issue in contract["issues"])

    rows = (
        ("control_cofitok", control_cofitok, CONTROL_RESIDUAL_ALIGNMENT_CONFIG),
        ("control_dense", control_dense, CONTROL_RESIDUAL_ALIGNMENT_CONFIG),
        (
            "residual_cofitok",
            residual_cofitok,
            CANDIDATE_RESIDUAL_ALIGNMENT_CONFIG,
        ),
        (
            "residual_dense",
            residual_dense,
            CANDIDATE_RESIDUAL_ALIGNMENT_CONFIG,
        ),
    )
    for label, config, expected in rows:
        if _residual_alignment_config(config) != expected:
            issues.append(
                f"{label} residual-alignment config differs from the probe contract"
            )
        loss = config.get("loss", {})
        if not isinstance(loss, Mapping) or {
            field: copy.deepcopy(loss.get(field)) for field in RANKING_FIELDS
        } != CONTROL_RANKING_CONFIG:
            issues.append(f"{label} enables or changes legacy class-ranking fields")
        runtime = config.get("runtime", {})
        data = config.get("data", {})
        optimization = config.get("optimization", {})
        if runtime.get("steps") != 1_000:
            issues.append(f"{label} runtime.steps is not 1000")
        if data.get("dataset") != "imagenet_256_10pct":
            issues.append(f"{label} dataset is not imagenet_256_10pct")
        if int(data.get("batch_size", 0)) * int(
            optimization.get("gradient_accumulation_steps", 0)
        ) != 64:
            issues.append(f"{label} effective batch size is not 64")

    for method, control, residual in (
        ("cofitok", control_cofitok, residual_cofitok),
        ("dense_identity", control_dense, residual_dense),
    ):
        if _without_residual_alignment_fields(control) != (
            _without_residual_alignment_fields(residual)
        ):
            issues.append(
                f"{method} control/residual configs differ outside objective fields"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "role": PREPARATION_ROLE,
        "status": "pass" if not issues else "fail",
        "scope": SCOPE,
        "valid": not issues,
        "issues": issues,
        "objective": "shared_low_frequency_x0_residual_alignment",
        "residual_alignment_fields": list(RESIDUAL_ALIGNMENT_FIELDS),
        "control_residual_alignment_config": copy.deepcopy(
            CONTROL_RESIDUAL_ALIGNMENT_CONFIG
        ),
        "candidate_residual_alignment_config": copy.deepcopy(
            CANDIDATE_RESIDUAL_ALIGNMENT_CONFIG
        ),
        "pair_contracts": pair_contracts,
        "execution_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "authorization_required": True,
        "gpu_execution_authorized": False,
    }


def validate_preparation(
    preparation: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    git = _clean_git(
        preparation.get("git", {}),
        label="residual-alignment preparation",
        include_tree=True,
    )
    if (
        preparation.get("schema_version") != SCHEMA_VERSION
        or preparation.get("role") != PREPARATION_ROLE
        or preparation.get("status") != "pass"
        or preparation.get("scope") != SCOPE
        or preparation.get("valid") is not True
        or preparation.get("issues") != []
        or preparation.get("objective")
        != "shared_low_frequency_x0_residual_alignment"
        or preparation.get("execution_boundary") != EXECUTION_BOUNDARY
        or preparation.get("claim_boundary") != CLAIM_BOUNDARY
        or preparation.get("authorization_required") is not True
        or preparation.get("gpu_execution_authorized") is not False
        or preparation.get("output_root") != expected_output_root
        or git
        != {
            "revision": expected_revision,
            "tree": expected_tree,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
    ):
        raise ValueError("semantic residual-alignment preparation contract differs")
    return copy.deepcopy(dict(preparation))


def validate_quality_bridge_hold(result: Mapping[str, Any]) -> dict[str, Any]:
    screen = result.get("quality_screen")
    boundary = result.get("authorization_boundary")
    terminal = result.get("terminal")
    class_fidelity = terminal.get("class_fidelity") if isinstance(terminal, Mapping) else None
    if (
        result.get("schema_version") != 1
        or result.get("role") != "stability_full_data_quality_bridge_result"
        or result.get("status") != "completed"
        or result.get("git") != QUALITY_BRIDGE_GIT
        or not isinstance(screen, Mapping)
        or screen.get("status") != "hold"
        or screen.get("failed_checks") != EXPECTED_FAILED_CHECKS
        or screen.get("non_authorizing") is not True
        or not isinstance(class_fidelity, Mapping)
        or class_fidelity.get("status") != "hold"
        or not isinstance(boundary, Mapping)
        or boundary.get("full_training_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("release_authorization_allowed") is not False
    ):
        raise ValueError("quality-bridge hold source differs")
    return copy.deepcopy(dict(result))


def validate_source_postevaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision")
    claim_boundary = report.get("claim_boundary")
    if (
        report.get("schema_version") != 1
        or report.get("role")
        != "generation_conditioning_ranking_four_arm_postevaluation"
        or report.get("status") != "completed"
        or report.get("git") != SOURCE_POSTEVALUATION_GIT
        or not isinstance(decision, Mapping)
        or decision.get("method_passes")
        != {"cofitok": False, "dense_identity": False}
        or decision.get("shared_semantic_alignment_recovery_supported") is not False
        or decision.get("cofitok_specific_advantage_claim_allowed") is not False
        or decision.get("recommended_next_action")
        != "revise_training_time_semantic_alignment_objective"
        or not isinstance(claim_boundary, Mapping)
        or claim_boundary.get("diagnostic_only") is not True
        or any(
            claim_boundary.get(field) is not False
            for field in (
                "authorizes_training",
                "authorizes_sampling",
                "authorizes_checkpoint_promotion",
                "authorizes_full_training",
                "authorizes_release",
            )
        )
    ):
        raise ValueError("source four-arm postevaluation decision differs")
    return copy.deepcopy(dict(report))


def build_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    quality_bridge_result: Mapping[str, Any],
    quality_bridge_result_identity: Mapping[str, Any],
    source_postevaluation: Mapping[str, Any],
    source_postevaluation_identity: Mapping[str, Any],
    runbook_identity: Mapping[str, Any],
    authorization_git: Mapping[str, Any],
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    preparation_id = _identity(
        preparation_identity,
        label="residual-alignment preparation",
    )
    standing_id = _identity(
        standing_authorization_identity,
        label="standing authorization",
    )
    quality_id = _identity(
        quality_bridge_result_identity,
        label="quality-bridge result",
    )
    source_id = _identity(
        source_postevaluation_identity,
        label="source postevaluation",
    )
    runbook_id = _identity(runbook_identity, label="residual-alignment runbook")
    if (
        quality_id["path"] != QUALITY_BRIDGE_RESULT_PATH
        or quality_id["sha256"]
        != EXPECTED_SOURCE_SHA256["quality_bridge_result"]
    ):
        raise ValueError("quality-bridge result identity differs")
    if (
        source_id["path"] != SOURCE_POSTEVALUATION_PATH
        or source_id["sha256"]
        != EXPECTED_SOURCE_SHA256["source_postevaluation"]
    ):
        raise ValueError("source postevaluation identity differs")
    git = _clean_git(
        authorization_git,
        label="residual-alignment authorization builder",
        include_tree=True,
    )
    if git != {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("residual-alignment authorization Git differs")
    validate_preparation(
        preparation,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )
    validate_standing_experiment_authorization(standing_authorization)
    validate_quality_bridge_hold(quality_bridge_result)
    validate_source_postevaluation(source_postevaluation)
    if expected_output_root != OUTPUT_ROOT:
        raise ValueError("residual-alignment output root differs")

    return {
        "schema_version": SCHEMA_VERSION,
        "role": AUTHORIZATION_ROLE,
        "status": "authorized",
        "authorization_mode": "active_standing_experiment_authorization",
        "scope": SCOPE,
        "stage": STAGE,
        "authorized_git": git,
        "output_root": expected_output_root,
        "source_reports": {
            "preparation": preparation_id,
            "standing_authorization": standing_id,
            "quality_bridge_result": quality_id,
            "source_postevaluation": source_id,
            "runbook": runbook_id,
        },
        "scientific_route": {
            "terminal_status": "hold",
            "failed_checks": copy.deepcopy(EXPECTED_FAILED_CHECKS),
            "source_method_passes": {
                "cofitok": False,
                "dense_identity": False,
            },
            "source_recommended_next_action": (
                "revise_training_time_semantic_alignment_objective"
            ),
            "candidate_objective": "shared_low_frequency_x0_residual_alignment",
            "generation_advantage_proven": False,
        },
        "execution_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "generation_advantage_proven": False,
    }


def validate_execution_authorization(
    authorization: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    expected = build_execution_authorization(**kwargs)
    if dict(authorization) != expected:
        raise ValueError("semantic residual-alignment execution authorization differs")
    return expected
