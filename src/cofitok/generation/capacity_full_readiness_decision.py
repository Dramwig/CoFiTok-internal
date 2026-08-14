from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_completion_result import (
    ABSOLUTE_QUALITY_CHECKS,
    validate_capacity_completion_100k_result,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)


CAPACITY_FULL_READINESS_DECISION_SCHEMA_VERSION = 1
CAPACITY_FULL_READINESS_DECISION_ROLE = (
    "stability_capacity_full_300k_readiness_execution_decision"
)
CAPACITY_FULL_ROOT_ID = "stability_capacity_full_300k_v1"
SELECTED_RECOMMENDATION_ID = (
    "build_source_compatible_250m_full_300k_readiness_decision"
)
EXPECTED_PARAMETER_COUNTS = {
    "cofitok": 250_153_763,
    "dense_identity": 250_135_043,
}
EXPECTED_SOURCE_CONFIG_NAMES = {
    "cofitok": (
        "imagenet256_stability_capacity_probe_rgbtail3_rollout_x0_u2_"
        "ema_teacher_k8_100k"
    ),
    "dense_identity": (
        "imagenet256_stability_capacity_probe_rollout_x0_u2_"
        "ema_teacher_dense_100k"
    ),
}
EXPECTED_TARGET_CONFIG_NAMES = {
    "cofitok": (
        "imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k"
    ),
    "dense_identity": (
        "imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k"
    ),
}
EXPECTED_BRIDGE_CHANGES = {
    "loss.ema_teacher_consistency_start_step": (30_000, 180_000),
    "loss.ema_teacher_consistency_warmup_steps": (10_000, 60_000),
    "loss.rollout_consistency_warmup_steps": (10_000, 60_000),
    "optimization.log_interval": (50, 100),
    "optimization.min_learning_rate": (1e-5, 5e-6),
    "optimization.warmup_steps": (1_000, 5_000),
    "runtime.evaluation_interval": (1_000, 2_000),
    "runtime.protected_checkpoint_steps": (
        [10_000],
        [50_000, 100_000, 200_000, 300_000],
    ),
    "runtime.steps": (100_000, 300_000),
}
CAPACITY_FULL_READINESS_SOURCE_NAMES = {
    "capacity_completion_result",
    "capacity_completion_result_waiter_status",
    "capacity_completion_result_waiter_deployment_receipt",
    "capacity_completion_launch_receipt",
    "standing_authorization",
    "source_cofitok_config",
    "source_dense_identity_config",
    "target_cofitok_config",
    "target_dense_identity_config",
}
RESULT_WAITER_BOUNDARY = {
    "cpu_only_waiter": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "additional_training_allowed": False,
    "process_signaling_allowed": False,
    "source_reports_must_be_physically_replayed": True,
    "existing_result_must_be_byte_equivalent": True,
    "result_is_promotion_gate": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "new_source_compatible_gate_or_decision_required": True,
}
CAPACITY_FULL_READINESS_DECISION_BOUNDARY = {
    "decision_evidence_complete": True,
    "readiness_gpu_runtime_benchmark_allowed": True,
    "readiness_artifact_build_allowed": True,
    "capacity_100k_checkpoint_resume_allowed": False,
    "fresh_full_training_required": True,
    "training_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "new_readiness_artifact_required_before_training": True,
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
    revision: str,
    tree: str,
    branch: str,
    label: str,
) -> dict[str, Any]:
    expected = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    if not _hex(revision, length=40) or not _hex(tree, length=40):
        raise ValueError(f"{label} expected Git identity is malformed")
    if dict(value) != expected:
        raise ValueError(f"{label} Git identity differs")
    return expected


def _path_value(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"config field is missing: {dotted}")
        current = current[part]
    return copy.deepcopy(current)


def _config_differences(
    source: Any,
    target: Any,
    *,
    prefix: str = "",
) -> dict[str, tuple[Any, Any]]:
    if isinstance(source, Mapping) and isinstance(target, Mapping):
        differences: dict[str, tuple[Any, Any]] = {}
        for key in sorted(set(source) | set(target)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in source:
                differences[path] = ("<missing>", copy.deepcopy(target[key]))
            elif key not in target:
                differences[path] = (copy.deepcopy(source[key]), "<missing>")
            else:
                differences.update(
                    _config_differences(source[key], target[key], prefix=path)
                )
        return differences
    if source != target:
        return {prefix: (copy.deepcopy(source), copy.deepcopy(target))}
    return {}


def _pair_validation(
    report: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    cofitok = report.get("cofitok")
    dense = report.get("dense")
    recipe = report.get("training_recipe")
    if (
        report.get("status") != "pass"
        or report.get("mismatches") != []
        or not isinstance(cofitok, Mapping)
        or not isinstance(dense, Mapping)
        or not isinstance(recipe, Mapping)
        or cofitok.get("parameter_count") != EXPECTED_PARAMETER_COUNTS["cofitok"]
        or dense.get("parameter_count")
        != EXPECTED_PARAMETER_COUNTS["dense_identity"]
        or report.get("matched_backbone", {}).get("base_channels") != 256
        or recipe.get("stage") != stage
        or recipe.get("valid") is not True
    ):
        raise ValueError(f"capacity-full {stage} pair validation differs")
    return copy.deepcopy(dict(report))


def validate_capacity_to_full_config_bridge(
    *,
    source_configs: Mapping[str, Mapping[str, Any]],
    target_configs: Mapping[str, Mapping[str, Any]],
    source_pair_validation: Mapping[str, Any],
    target_pair_validation: Mapping[str, Any],
) -> dict[str, Any]:
    methods = {"cofitok", "dense_identity"}
    if set(source_configs) != methods or set(target_configs) != methods:
        raise ValueError("capacity-full config method set differs")
    source_validation = _pair_validation(
        source_pair_validation,
        stage="stability_capacity_probe",
    )
    target_validation = _pair_validation(
        target_pair_validation,
        stage="stability_full",
    )
    normalized: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        source = source_configs[method]
        target = target_configs[method]
        if source.get("name") != EXPECTED_SOURCE_CONFIG_NAMES[method]:
            raise ValueError(f"capacity-full {method} source config differs")
        if target.get("name") != EXPECTED_TARGET_CONFIG_NAMES[method]:
            raise ValueError(f"capacity-full {method} target config differs")
        differences = _config_differences(source, target)
        expected = {
            "name": (
                EXPECTED_SOURCE_CONFIG_NAMES[method],
                EXPECTED_TARGET_CONFIG_NAMES[method],
            ),
            **EXPECTED_BRIDGE_CHANGES,
        }
        if differences != expected:
            raise ValueError(
                f"capacity-full {method} source-to-target config bridge differs"
            )
        normalized[method] = {
            "source_name": source["name"],
            "target_name": target["name"],
            "changes": {
                path: {
                    "source": copy.deepcopy(values[0]),
                    "target": copy.deepcopy(values[1]),
                }
                for path, values in differences.items()
            },
            "unchanged_method_and_backbone_contract": True,
            "fresh_start_required": True,
        }
    return {
        "status": "pass",
        "source_stage": "stability_capacity_probe",
        "target_stage": "stability_full",
        "source_pair_validation": source_validation,
        "target_pair_validation": target_validation,
        "methods": normalized,
        "interpretation": (
            "The target is the predeclared fresh 300K stability-full recipe. "
            "The 100K checkpoints are qualification evidence only: the changed "
            "horizon, scheduler, evaluation, retention, and scaled consistency "
            "windows make checkpoint resume invalid under exact-resume rules."
        ),
    }


def readiness_branch_selected(report: Mapping[str, Any]) -> bool:
    recommendation = report.get("decision_support", {}).get(
        "recommended_next_stage", {}
    )
    return isinstance(recommendation, Mapping) and recommendation.get(
        "id"
    ) == SELECTED_RECOMMENDATION_ID


def build_capacity_full_readiness_decision(
    *,
    capacity_completion_result: Mapping[str, Any],
    capacity_completion_result_identity: Mapping[str, Any],
    result_waiter_status: Mapping[str, Any],
    result_waiter_status_identity: Mapping[str, Any],
    result_waiter_deployment_receipt: Mapping[str, Any],
    result_waiter_deployment_receipt_identity: Mapping[str, Any],
    capacity_completion_launch_receipt: Mapping[str, Any],
    capacity_completion_launch_receipt_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    source_configs: Mapping[str, Mapping[str, Any]],
    source_config_identities: Mapping[str, Mapping[str, Any]],
    target_configs: Mapping[str, Mapping[str, Any]],
    target_config_identities: Mapping[str, Mapping[str, Any]],
    source_pair_validation: Mapping[str, Any],
    target_pair_validation: Mapping[str, Any],
    training_run_dirs: Mapping[str, str],
    training_state_absent: bool,
    full_output_root: str,
    benchmark_root: str,
    storage_path: str,
    decision_builder_git: Mapping[str, Any],
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    expected_result_execution_revision: str,
    expected_result_execution_tree: str,
    expected_result_execution_branch: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    expected_result_revision: str,
    expected_result_tree: str,
    expected_result_branch: str,
) -> dict[str, Any]:
    result_identity = _identity(
        capacity_completion_result_identity,
        label="capacity completion result",
    )
    waiter_identity = _identity(
        result_waiter_status_identity,
        label="capacity completion result waiter status",
    )
    deployment_identity = _identity(
        result_waiter_deployment_receipt_identity,
        label="capacity completion result waiter deployment",
    )
    launch_identity = _identity(
        capacity_completion_launch_receipt_identity,
        label="capacity completion launch receipt",
    )
    standing_identity = _identity(
        standing_authorization_identity,
        label="standing authorization",
    )
    builder_git = _git(
        decision_builder_git,
        revision=expected_decision_revision,
        tree=expected_decision_tree,
        branch=expected_decision_branch,
        label="capacity-full readiness decision builder",
    )
    result_evidence = validate_capacity_completion_100k_result(
        capacity_completion_result,
        expected_execution_revision=expected_result_execution_revision,
        expected_execution_tree=expected_result_execution_tree,
        expected_execution_branch=expected_result_execution_branch,
        expected_training_revision=expected_training_revision,
        expected_training_tree=expected_training_tree,
        expected_training_branch=expected_training_branch,
        expected_result_revision=expected_result_revision,
        expected_result_tree=expected_result_tree,
        expected_result_branch=expected_result_branch,
    )
    recommendation = result_evidence["recommended_next_stage"]
    failed_checks = capacity_completion_result["quality_screen"]["failed_checks"]
    if (
        recommendation.get("id") != SELECTED_RECOMMENDATION_ID
        or recommendation.get("category")
        != "scale_responsive_absolute_quality_hold"
        or capacity_completion_result["quality_screen"].get("status") != "hold"
        or not failed_checks
        or not set(failed_checks) <= ABSOLUTE_QUALITY_CHECKS
        or capacity_completion_result["decision_support"].get(
            "step_50000_to_100000_shared_strict_fid_improvement"
        )
        is not True
    ):
        raise ValueError(
            "capacity completion result does not select full-readiness decision"
        )
    result_git = {
        "revision": expected_result_revision,
        "tree": expected_result_tree,
        "branch": expected_result_branch,
        "tracked_dirty": False,
    }
    if (
        result_waiter_status.get("schema_version") != 1
        or result_waiter_status.get("role")
        != "capacity_completion_100k_result_source_replay_waiter"
        or result_waiter_status.get("status") != "completed"
        or result_waiter_status.get("detail")
        != "source_replayed_capacity_completion_100k_result_emitted"
        or result_waiter_status.get("git") != result_git
        or result_waiter_status.get("capacity_completion_100k_result")
        != result_identity
        or result_waiter_status.get("recommended_next_stage") != recommendation
        or result_waiter_status.get("authorization_boundary")
        != RESULT_WAITER_BOUNDARY
    ):
        raise ValueError("capacity completion result waiter status differs")
    deployment_waiter = result_waiter_deployment_receipt.get("waiter", {})
    if (
        result_waiter_deployment_receipt.get("schema_version") != 1
        or result_waiter_deployment_receipt.get("role")
        != "capacity_completion_100k_result_waiter_deployment"
        or result_waiter_deployment_receipt.get("status") != "active"
        or result_waiter_deployment_receipt.get("git") != result_git
        or result_waiter_deployment_receipt.get("authorization_boundary")
        != RESULT_WAITER_BOUNDARY
        or result_waiter_deployment_receipt.get("incremental_bundle", {}).get(
            "advertised_revision"
        )
        != expected_result_revision
        or deployment_waiter.get("status_path") != waiter_identity["path"]
        or deployment_waiter.get("result_path") != result_identity["path"]
    ):
        raise ValueError("capacity completion result waiter deployment differs")
    if (
        capacity_completion_result.get("source_reports", {}).get(
            "capacity_completion_launch_receipt"
        )
        != launch_identity
    ):
        raise ValueError("capacity completion result launch source differs")
    methods = {"cofitok", "dense_identity"}
    if (
        set(source_config_identities) != methods
        or set(target_config_identities) != methods
    ):
        raise ValueError("capacity-full config identity set differs")
    normalized_source_identities = {
        method: _identity(
            source_config_identities[method],
            label=f"capacity-full source {method} config",
        )
        for method in methods
    }
    normalized_target_identities = {
        method: _identity(
            target_config_identities[method],
            label=f"capacity-full target {method} config",
        )
        for method in methods
    }
    launch_sources = capacity_completion_launch_receipt.get("source_reports", {})
    if (
        launch_sources.get("cofitok_config")
        != normalized_source_identities["cofitok"]
        or launch_sources.get("dense_config")
        != normalized_source_identities["dense_identity"]
    ):
        raise ValueError("capacity-full source configs differ from completion launch")
    standing = validate_standing_experiment_authorization(standing_authorization)
    if standing["preserved_safety_boundaries"] != (
        STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("capacity-full standing authorization boundary differs")
    bridge = validate_capacity_to_full_config_bridge(
        source_configs=source_configs,
        target_configs=target_configs,
        source_pair_validation=source_pair_validation,
        target_pair_validation=target_pair_validation,
    )
    if set(training_run_dirs) != methods or training_state_absent is not True:
        raise ValueError("capacity-full training state is not absent")
    normalized_runs = {}
    root_path = PurePosixPath(full_output_root)
    if not root_path.is_absolute() or root_path.name != CAPACITY_FULL_ROOT_ID:
        raise ValueError("capacity-full output root differs")
    for method in methods:
        run = PurePosixPath(training_run_dirs[method])
        if not run.is_absolute() or run.parent != root_path:
            raise ValueError(f"capacity-full {method} run path differs")
        normalized_runs[method] = str(run)
    if set(normalized_runs.values()) != {
        str(root_path / "cofitok"),
        str(root_path / "dense_identity"),
    }:
        raise ValueError("capacity-full matched run names differ")
    benchmark = PurePosixPath(benchmark_root)
    storage = PurePosixPath(storage_path)
    if (
        not benchmark.is_absolute()
        or benchmark != root_path / "reports/runtime_benchmark"
        or not storage.is_absolute()
    ):
        raise ValueError("capacity-full readiness paths differ")
    sources = {
        "capacity_completion_result": result_identity,
        "capacity_completion_result_waiter_status": waiter_identity,
        "capacity_completion_result_waiter_deployment_receipt": (
            deployment_identity
        ),
        "capacity_completion_launch_receipt": launch_identity,
        "standing_authorization": standing_identity,
        "source_cofitok_config": normalized_source_identities["cofitok"],
        "source_dense_identity_config": normalized_source_identities[
            "dense_identity"
        ],
        "target_cofitok_config": normalized_target_identities["cofitok"],
        "target_dense_identity_config": normalized_target_identities[
            "dense_identity"
        ],
    }
    return {
        "schema_version": CAPACITY_FULL_READINESS_DECISION_SCHEMA_VERSION,
        "status": "authorized",
        "role": CAPACITY_FULL_READINESS_DECISION_ROLE,
        "decision_builder_git": builder_git,
        "result_builder_git": result_git,
        "source_reports": sources,
        "result_trigger": {
            "quality_status": "hold",
            "failed_checks": copy.deepcopy(failed_checks),
            "shared_50000_to_100000_fid_improvement": True,
            "recommended_next_stage": copy.deepcopy(recommendation),
        },
        "config_bridge": bridge,
        "fresh_training_contract": {
            "source_capacity_checkpoints_are_evidence_only": True,
            "capacity_100k_checkpoint_resume_allowed": False,
            "target_start_step": 0,
            "target_steps": 300_000,
            "exact_resume_allowed_only_from_target_config_checkpoints": True,
            "training_state_absent_at_decision": True,
            "training_run_dirs": normalized_runs,
        },
        "readiness_plan": {
            "stage": "stability_full",
            "full_output_root": str(root_path),
            "benchmark_root": str(benchmark),
            "storage_path": str(storage),
            "runtime_candidates": [[1, 64], [2, 32], [4, 16], [8, 8], [16, 4]],
            "runtime_baseline": [1, 64],
            "effective_batch_size": 64,
            "target_steps": 300_000,
            "protected_checkpoint_steps": [50_000, 100_000, 200_000, 300_000],
            "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
            "readiness_artifact_required_before_training": True,
        },
        "standing_authorization": standing,
        "execution_authorization": {
            "readiness_gpu_runtime_benchmark_allowed": True,
            "readiness_artifact_build_allowed": True,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "claim_policy": {
            "result_is_promotion_gate": False,
            "readiness_decision_is_training_gate": False,
            "cross_protocol_numeric_ranking_allowed": False,
            "formal_generation_claim_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_FULL_READINESS_DECISION_BOUNDARY
        ),
    }


def validate_capacity_full_readiness_decision(
    report: Mapping[str, Any],
    *,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    expected_result_revision: str,
    expected_result_tree: str,
    expected_result_branch: str,
) -> dict[str, Any]:
    sources = report.get("source_reports")
    trigger = report.get("result_trigger")
    fresh = report.get("fresh_training_contract")
    plan = report.get("readiness_plan")
    authorization = report.get("execution_authorization")
    if (
        report.get("schema_version")
        != CAPACITY_FULL_READINESS_DECISION_SCHEMA_VERSION
        or report.get("status") != "authorized"
        or report.get("role") != CAPACITY_FULL_READINESS_DECISION_ROLE
        or report.get("authorization_boundary")
        != CAPACITY_FULL_READINESS_DECISION_BOUNDARY
        or not isinstance(sources, Mapping)
        or set(sources) != CAPACITY_FULL_READINESS_SOURCE_NAMES
        or not isinstance(trigger, Mapping)
        or not isinstance(fresh, Mapping)
        or not isinstance(plan, Mapping)
        or not isinstance(authorization, Mapping)
    ):
        raise ValueError("capacity-full readiness decision contract differs")
    decision_git = _git(
        report.get("decision_builder_git", {}),
        revision=expected_decision_revision,
        tree=expected_decision_tree,
        branch=expected_decision_branch,
        label="capacity-full readiness decision",
    )
    result_git = _git(
        report.get("result_builder_git", {}),
        revision=expected_result_revision,
        tree=expected_result_tree,
        branch=expected_result_branch,
        label="capacity-full result source",
    )
    for name, identity in sources.items():
        _identity(identity, label=f"capacity-full decision source {name}")
    failed = trigger.get("failed_checks")
    recommendation = trigger.get("recommended_next_stage")
    if (
        trigger.get("quality_status") != "hold"
        or not isinstance(failed, list)
        or not failed
        or not set(failed) <= ABSOLUTE_QUALITY_CHECKS
        or trigger.get("shared_50000_to_100000_fid_improvement") is not True
        or not isinstance(recommendation, Mapping)
        or recommendation.get("id") != SELECTED_RECOMMENDATION_ID
        or recommendation.get("execution_ready") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("capacity-full readiness result trigger differs")
    if (
        fresh.get("source_capacity_checkpoints_are_evidence_only") is not True
        or fresh.get("capacity_100k_checkpoint_resume_allowed") is not False
        or fresh.get("target_start_step") != 0
        or fresh.get("target_steps") != 300_000
        or fresh.get("exact_resume_allowed_only_from_target_config_checkpoints")
        is not True
        or fresh.get("training_state_absent_at_decision") is not True
        or set(fresh.get("training_run_dirs", {}))
        != {"cofitok", "dense_identity"}
    ):
        raise ValueError("capacity-full readiness fresh-training contract differs")
    root = str(plan.get("full_output_root", ""))
    if (
        PurePosixPath(root).name != CAPACITY_FULL_ROOT_ID
        or plan.get("stage") != "stability_full"
        or plan.get("runtime_candidates")
        != [[1, 64], [2, 32], [4, 16], [8, 8], [16, 4]]
        or plan.get("runtime_baseline") != [1, 64]
        or plan.get("effective_batch_size") != 64
        or plan.get("target_steps") != 300_000
        or plan.get("protected_checkpoint_steps")
        != [50_000, 100_000, 200_000, 300_000]
        or plan.get("parameter_counts") != EXPECTED_PARAMETER_COUNTS
        or plan.get("readiness_artifact_required_before_training") is not True
    ):
        raise ValueError("capacity-full readiness plan differs")
    expected_authorization = {
        "readiness_gpu_runtime_benchmark_allowed": True,
        "readiness_artifact_build_allowed": True,
        "training_launch_allowed": False,
        "full_300k_launch_allowed": False,
    }
    if authorization != expected_authorization:
        raise ValueError("capacity-full readiness execution authorization differs")
    return {
        "status": "authorized",
        "decision_builder_git": decision_git,
        "result_builder_git": result_git,
        "full_output_root": root,
        "training_run_dirs": copy.deepcopy(fresh["training_run_dirs"]),
        "readiness_plan": copy.deepcopy(dict(plan)),
        "execution_authorization": copy.deepcopy(expected_authorization),
        "authorization_boundary": copy.deepcopy(
            CAPACITY_FULL_READINESS_DECISION_BOUNDARY
        ),
    }
