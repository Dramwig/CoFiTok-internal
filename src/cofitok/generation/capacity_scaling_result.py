from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_CONFIGURED_STEPS,
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_MIN_COARSE_TOKEN_ENERGY_RATIO,
    CAPACITY_PROBE_PARAMETER_COUNTS,
    CAPACITY_PROBE_SAMPLES_PER_ARM,
    CAPACITY_PROBE_STOP_STEP,
)
from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_RESULT_BOUNDARY,
    CAPACITY_PROBE_RESULT_ROLE,
    _coarse_token_utilization,
)
from cofitok.generation.capacity_scaling_decision import (
    CAPACITY_SCALING_TARGET_STEP,
    validate_capacity_scaling_decision,
)
from cofitok.generation.capacity_scaling_execution import (
    CAPACITY_SCALING_LAUNCH_BOUNDARY,
    validate_capacity_scaling_launch_receipt,
)
from cofitok.generation.capacity_scaling_training import (
    CAPACITY_SCALING_TRAINING_BOUNDARY,
    CAPACITY_SCALING_TRAINING_ROLE,
)


CAPACITY_SCALING_RESULT_SCHEMA_VERSION = 1
CAPACITY_SCALING_RESULT_ROLE = "stability_full_data_capacity_scaling_50k_result"
CAPACITY_SCALING_RESULT_SOURCE_NAMES = {
    "capacity_probe_result",
    "capacity_scaling_decision",
    "capacity_scaling_launch_receipt",
    "capacity_scaling_execution_status",
    "cofitok_training_validation",
    "dense_identity_training_validation",
    "milestone_50000",
    "cofitok_checkpoint_eval",
}
CAPACITY_SCALING_RESULT_BOUNDARY = {
    "capacity_scaling_50k_evidence_complete": True,
    "capacity_scaling_execution_allowed": False,
    "additional_training_allowed": False,
    "configured_100k_completion_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "new_source_compatible_decision_required": True,
}
CAPACITY_COMPLETION_PREPARATION_ID = (
    "prepare_source_compatible_capacity_completion_decision"
)
CAPACITY_COMPLETION_QUALITY_HOLD_ID = (
    "hold_capacity_completion_and_revisit_training_quality"
)
CAPACITY_COMPLETION_MECHANISM_HOLD_ID = (
    "hold_capacity_completion_and_recover_factorization_mechanism"
)
CAPACITY_SCALING_MAX_COFITOK_FID_RELATIVE_GAP = 0.25


def _hex(value: Any, *, length: int = 64) -> bool:
    if not isinstance(value, str) or len(value) != length or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _hex(digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(
    value: Mapping[str, Any],
    *,
    revision: str,
    branch: str,
    with_tree: bool,
    label: str,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }
    if with_tree:
        tree = value.get("tree")
        if not _hex(tree, length=40):
            raise ValueError(f"{label} Git tree is malformed")
        expected["tree"] = tree
    if dict(value) != expected:
        raise ValueError(f"{label} Git identity differs")
    return expected


def _training(
    value: Mapping[str, Any],
    *,
    method: str,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    expected_parameters = CAPACITY_PROBE_PARAMETER_COUNTS["base256"][method]
    source = value.get("source_checkpoint")
    checkpoint = value.get("checkpoint")
    runtime_sha = value.get("runtime_environment_sha256")
    dataset_sha = value.get("dataset_identity_sha256")
    last_resume_step = int(value.get("last_resume_step", -1))
    recovery_resume_used = value.get("recovery_resume_used")
    if (
        int(value.get("schema_version", -1)) != 1
        or value.get("status") != "pass"
        or value.get("role") != CAPACITY_SCALING_TRAINING_ROLE
        or value.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or int(value.get("configured_steps", -1))
        != CAPACITY_PROBE_CONFIGURED_STEPS
        or int(value.get("source_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or int(value.get("completed_steps", -1)) != CAPACITY_SCALING_TARGET_STEP
        or value.get("training_complete") is not False
        or value.get("exact_resume") is not True
        or int(value.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or int(value.get("images_seen", -1))
        != CAPACITY_SCALING_TARGET_STEP * CAPACITY_PROBE_EFFECTIVE_BATCH
        or int(value.get("parameter_count", -1)) != expected_parameters
        or value.get("authorization_boundary") != CAPACITY_SCALING_TRAINING_BOUNDARY
        or not _hex(runtime_sha)
        or not _hex(dataset_sha)
        or not CAPACITY_PROBE_STOP_STEP <= last_resume_step < CAPACITY_SCALING_TARGET_STEP
        or type(recovery_resume_used) is not bool
        or recovery_resume_used is not (last_resume_step > CAPACITY_PROBE_STOP_STEP)
        or not isinstance(source, Mapping)
        or not isinstance(checkpoint, Mapping)
    ):
        raise ValueError(f"capacity scaling {method} training validation differs")
    source_identity = _identity(source, label=f"{method} source checkpoint")
    source_integrity = _identity(
        source.get("integrity_manifest", {}),
        label=f"{method} source integrity",
    )
    checkpoint_identity = _identity(checkpoint, label=f"{method} checkpoint")
    checkpoint_integrity = _identity(
        checkpoint.get("integrity_manifest", {}),
        label=f"{method} checkpoint integrity",
    )
    if (
        int(source.get("step", -1)) != CAPACITY_PROBE_STOP_STEP
        or not source_identity["path"].endswith("checkpoint_step_00010000.pt")
        or int(checkpoint.get("step", -1)) != CAPACITY_SCALING_TARGET_STEP
        or not checkpoint_identity["path"].endswith("checkpoint_step_00050000.pt")
        or source_integrity["path"] != f"{source_identity['path']}.integrity.json"
        or checkpoint_integrity["path"]
        != f"{checkpoint_identity['path']}.integrity.json"
    ):
        raise ValueError(f"capacity scaling {method} checkpoint identity differs")
    return {
        "source_checkpoint": source_identity,
        "source_checkpoint_integrity_manifest": source_integrity,
        "checkpoint": checkpoint_identity,
        "checkpoint_integrity_manifest": checkpoint_integrity,
        "images_seen": int(value["images_seen"]),
        "parameter_count": expected_parameters,
        "last_resume_step": last_resume_step,
        "recovery_resume_used": recovery_resume_used,
        "runtime_environment_sha256": runtime_sha,
        "dataset_identity_sha256": dataset_sha,
    }


def _milestone(
    report: Mapping[str, Any],
    *,
    source_verification: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    methods = report.get("methods")
    sources = report.get("source_reports")
    alerts = report.get("quality_alerts")
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("status") != "completed"
        or report.get("role") != "training_quality_trend_only"
        or report.get("source_profile") != "capacity_scaling"
        or int(report.get("milestone_step", -1)) != CAPACITY_SCALING_TARGET_STEP
        or int(report.get("expected_samples", -1)) != CAPACITY_PROBE_SAMPLES_PER_ARM
        or report.get("claim_policy", {}).get("formal_generation_claim_allowed")
        is not False
        or not isinstance(methods, Mapping)
        or set(methods) != {"cofitok", "dense_identity"}
        or not isinstance(sources, Mapping)
        or not isinstance(alerts, list)
        or source_verification
        != {
            "status": "verified",
            "source_profile": "capacity_scaling",
            "source_reports": sources,
        }
        or evidence.get("status") != "verified"
        or evidence.get("source_profile") != "capacity_scaling"
        or evidence.get("quality_alerts") != alerts
    ):
        raise ValueError("capacity scaling step-50K milestone differs")
    normalized: dict[str, Any] = {}
    for method, budget in (("cofitok", 8), ("dense_identity", 1)):
        row = methods[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"capacity scaling milestone {method} row is malformed")
        fid = _finite(row.get("fid"), label=f"{method} step-50K FID")
        inception = _finite(
            row.get("inception_score"),
            label=f"{method} step-50K inception score",
        )
        if (
            int(row.get("checkpoint_step", -1)) != CAPACITY_SCALING_TARGET_STEP
            or int(row.get("sample_count", -1)) != CAPACITY_PROBE_SAMPLES_PER_ARM
            or int(row.get("selected_prefix_budget", -1)) != budget
            or row.get("weights") != "ema"
            or not _hex(row.get("checkpoint_sha256"))
            or not _hex(row.get("sample_set_sha256"))
            or fid < 0.0
            or inception <= 0.0
        ):
            raise ValueError(f"capacity scaling milestone {method} identity differs")
        normalized[method] = {
            "checkpoint": row.get("checkpoint"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "checkpoint_integrity_manifest": row.get(
                "checkpoint_integrity_manifest"
            ),
            "sample_count": int(row["sample_count"]),
            "sample_set_sha256": row.get("sample_set_sha256"),
            "fid": fid,
            "inception_score": inception,
            "endpoint_clean_mse": _finite(
                row.get("endpoint_clean_mse"),
                label=f"{method} endpoint MSE",
            ),
            "prefix_path_mse_auc": _finite(
                row.get("prefix_path_mse_auc"),
                label=f"{method} path AUC",
            ),
            "ordered_rank_by_path_auc": int(
                row.get("ordered_rank_by_path_auc", -1)
            ),
            "order_count": int(row.get("order_count", -1)),
            "zero_token_max_abs": _finite(
                row.get("zero_token_max_abs"),
                label=f"{method} zero token",
            ),
            "shuffled_to_ordered_endpoint_ratio": _finite(
                row.get("shuffled_to_ordered_endpoint_ratio"),
                label=f"{method} shuffle ratio",
            ),
        }
    cofitok = normalized["cofitok"]
    dense = normalized["dense_identity"]
    expected_alerts = []
    fid_relative_gap = (cofitok["fid"] - dense["fid"]) / max(
        dense["fid"], 1e-12
    )
    if fid_relative_gap > CAPACITY_SCALING_MAX_COFITOK_FID_RELATIVE_GAP:
        expected_alerts.append("cofitok_fid_more_than_25pct_above_dense")
    if cofitok["ordered_rank_by_path_auc"] != 1:
        expected_alerts.append("cofitok_ordered_prefix_not_rank1")
    if cofitok["zero_token_max_abs"] != 0.0:
        expected_alerts.append("cofitok_zero_token_contract_failed")
    if cofitok["shuffled_to_ordered_endpoint_ratio"] <= 1.0:
        expected_alerts.append("cofitok_shuffle_mismatch_not_detected")
    if alerts != expected_alerts:
        raise ValueError("capacity scaling milestone quality alerts differ")
    if (
        cofitok["order_count"] < 1
        or not 1 <= cofitok["ordered_rank_by_path_auc"] <= cofitok["order_count"]
        or not math.isclose(
            float(evidence.get("cofitok_fid", math.nan)),
            normalized["cofitok"]["fid"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(evidence.get("dense_fid", math.nan)),
            normalized["dense_identity"]["fid"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("capacity scaling milestone evidence FID differs")
    return {
        "methods": normalized,
        "quality_alerts": copy.deepcopy(alerts),
        "source_reports": copy.deepcopy(dict(sources)),
    }


def _utilization_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("capacity scaling token-utilization summary is missing")
    expected_keys = {
        "source_metric",
        "partition_schema",
        "token_count",
        "coarse_token_count",
        "full_resolution_tail_token_count",
        "token_spatial_strides",
        "component_energy_ratios",
        "coarse_token_energy_ratio",
        "minimum_coarse_token_energy_ratio",
        "passed",
    }
    try:
        token_count = int(value.get("token_count", -1))
        coarse_count = int(value.get("coarse_token_count", -1))
        tail_count = int(value.get("full_resolution_tail_token_count", -1))
        strides = [int(item) for item in value.get("token_spatial_strides", [])]
        ratios = [float(item) for item in value.get("component_energy_ratios", [])]
    except (TypeError, ValueError):
        raise ValueError("capacity scaling token-utilization summary is malformed")
    coarse_ratio = sum(ratios[:coarse_count]) if coarse_count >= 0 else math.nan
    passed = coarse_ratio >= CAPACITY_PROBE_MIN_COARSE_TOKEN_ENERGY_RATIO
    if (
        set(value) != expected_keys
        or value.get("source_metric")
        != "component_energy_ratio_per_sample_mean"
        or value.get("partition_schema") != "token_spatial_stride_suffix_v1"
        or token_count < 3
        or len(strides) != token_count
        or len(ratios) != token_count
        or not 0 < coarse_count < token_count
        or tail_count != token_count - coarse_count
        or any(item <= 1 for item in strides[:coarse_count])
        or any(item != 1 for item in strides[coarse_count:])
        or any(not math.isfinite(item) or item < 0.0 for item in ratios)
        or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(
            _finite(
                value.get("coarse_token_energy_ratio"),
                label="capacity scaling coarse-token energy ratio",
            ),
            coarse_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or _finite(
            value.get("minimum_coarse_token_energy_ratio"),
            label="capacity scaling minimum coarse-token energy ratio",
        )
        != CAPACITY_PROBE_MIN_COARSE_TOKEN_ENERGY_RATIO
        or value.get("passed") is not passed
    ):
        raise ValueError("capacity scaling token-utilization summary differs")
    return copy.deepcopy(dict(value))


def build_capacity_scaling_50k_result(
    *,
    capacity_probe_result: Mapping[str, Any],
    capacity_scaling_decision: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    execution_status: Mapping[str, Any],
    training_validations: Mapping[str, Mapping[str, Any]],
    milestone_report: Mapping[str, Any],
    milestone_source_verification: Mapping[str, Any],
    milestone_evidence: Mapping[str, Any],
    cofitok_checkpoint_evaluation: Mapping[str, Any],
    source_identities: Mapping[str, Mapping[str, Any]],
    result_builder_git: Mapping[str, Any],
    expected_capacity_revision: str,
    expected_capacity_branch: str,
    expected_decision_revision: str,
    expected_decision_branch: str,
    expected_execution_revision: str,
    expected_execution_tree: str,
    expected_execution_branch: str,
    expected_training_tree: str,
) -> dict[str, Any]:
    if set(source_identities) != CAPACITY_SCALING_RESULT_SOURCE_NAMES:
        raise ValueError("capacity scaling result source set differs")
    if set(training_validations) != {"cofitok", "dense_identity"}:
        raise ValueError("capacity scaling result training set differs")
    sources = {
        name: _identity(identity, label=f"capacity scaling result source {name}")
        for name, identity in source_identities.items()
    }
    decision_evidence = validate_capacity_scaling_decision(
        capacity_scaling_decision,
        expected_decision_revision=expected_decision_revision,
        expected_decision_branch=expected_decision_branch,
    )
    if decision_evidence["execution_authorized"] is not True:
        raise ValueError("capacity scaling result requires the selected 50K decision")
    if (
        int(capacity_probe_result.get("schema_version", -1)) != 1
        or capacity_probe_result.get("status") != "completed"
        or capacity_probe_result.get("role") != CAPACITY_PROBE_RESULT_ROLE
        or capacity_probe_result.get("authorization_boundary")
        != CAPACITY_PROBE_RESULT_BOUNDARY
        or capacity_probe_result.get("decision", {}).get("capacity_supported")
        is not True
        or capacity_scaling_decision.get("source_evidence", {}).get(
            "capacity_probe_result"
        )
        != sources["capacity_probe_result"]
    ):
        raise ValueError("capacity scaling result requires the supported 10K result")
    output_root = str(capacity_probe_result.get("output_root", ""))
    launch = validate_capacity_scaling_launch_receipt(
        launch_receipt,
        expected_execution_revision=expected_execution_revision,
        expected_execution_tree=expected_execution_tree,
        expected_execution_branch=expected_execution_branch,
        expected_training_revision=expected_capacity_revision,
        expected_training_tree=expected_training_tree,
        expected_training_branch=expected_capacity_branch,
        expected_output_root=output_root,
    )
    if (
        launch_receipt.get("authorization_boundary")
        != CAPACITY_SCALING_LAUNCH_BOUNDARY
        or launch_receipt.get("source_reports", {}).get(
            "capacity_scaling_decision"
        )
        != sources["capacity_scaling_decision"]
        or launch_receipt.get("source_reports", {}).get("capacity_probe_result")
        != sources["capacity_probe_result"]
    ):
        raise ValueError("capacity scaling result launch sources differ")

    training = {
        method: _training(
            training_validations[method],
            method=method,
            expected_revision=expected_capacity_revision,
            expected_branch=expected_capacity_branch,
        )
        for method in ("cofitok", "dense_identity")
    }
    launch_sources = launch_receipt["selection"]["resume_sources"]
    for method in ("cofitok", "dense_identity"):
        if (
            launch_sources[method]["checkpoint"]
            != training[method]["source_checkpoint"]
            or launch_sources[method]["checkpoint_integrity_manifest"]
            != training[method]["source_checkpoint_integrity_manifest"]
        ):
            raise ValueError(f"capacity scaling {method} resumed another source")

    milestone = _milestone(
        milestone_report,
        source_verification=milestone_source_verification,
        evidence=milestone_evidence,
    )
    if (
        sources["cofitok_checkpoint_eval"]
        != milestone["source_reports"]["cofitok_checkpoint_eval"]
    ):
        raise ValueError("capacity scaling result binds another mechanism report")
    for method in ("cofitok", "dense_identity"):
        row = milestone["methods"][method]
        if (
            row["checkpoint"] != training[method]["checkpoint"]["path"]
            or row["checkpoint_sha256"] != training[method]["checkpoint"]["sha256"]
            or row["checkpoint_integrity_manifest"]
            != training[method]["checkpoint_integrity_manifest"]["path"]
        ):
            raise ValueError(f"capacity scaling {method} milestone checkpoint differs")

    status_boundary = execution_status.get("authorization_boundary")
    if (
        int(execution_status.get("schema_version", -1)) != 1
        or execution_status.get("status") != "completed"
        or execution_status.get("role")
        != "stability_full_data_capacity_scaling_50k_execution"
        or execution_status.get("stage") != "complete"
        or execution_status.get("git")
        != {
            "revision": expected_execution_revision,
            "tree": expected_execution_tree,
            "branch": expected_execution_branch,
            "tracked_dirty": False,
        }
        or not isinstance(status_boundary, Mapping)
        or status_boundary.get("configured_100k_completion_allowed") is not False
        or status_boundary.get("full_300k_launch_allowed") is not False
        or status_boundary.get("promotion_or_release_allowed") is not False
        or execution_status.get("launch_receipt")
        != sources["capacity_scaling_launch_receipt"]
        or execution_status.get("milestone") != sources["milestone_50000"]
        or execution_status.get("training_validations")
        != {
            "cofitok": sources["cofitok_training_validation"],
            "dense_identity": sources["dense_identity_training_validation"],
        }
    ):
        raise ValueError("capacity scaling completed execution status differs")

    mechanism = cofitok_checkpoint_evaluation.get("metrics")
    request = cofitok_checkpoint_evaluation.get("request")
    cofitok_row = milestone["methods"]["cofitok"]
    if (
        int(cofitok_checkpoint_evaluation.get("schema_version", -1)) != 2
        or cofitok_checkpoint_evaluation.get("status") != "completed"
        or cofitok_checkpoint_evaluation.get("role")
        != "generation_checkpoint_evaluation_report"
        or cofitok_checkpoint_evaluation.get("git")
        != {
            "revision": expected_capacity_revision,
            "branch": expected_capacity_branch,
            "tracked_dirty": False,
        }
        or int(cofitok_checkpoint_evaluation.get("checkpoint_step", -1))
        != CAPACITY_SCALING_TARGET_STEP
        or cofitok_checkpoint_evaluation.get("checkpoint")
        != cofitok_row["checkpoint"]
        or cofitok_checkpoint_evaluation.get("checkpoint_sha256")
        != cofitok_row["checkpoint_sha256"]
        or cofitok_checkpoint_evaluation.get("checkpoint_integrity_manifest")
        != cofitok_row["checkpoint_integrity_manifest"]
        or not isinstance(mechanism, Mapping)
        or not isinstance(request, Mapping)
        or int(request.get("num_images", -1)) != 256
        or int(request.get("timestep", -1)) != 500
        or int(request.get("random_orders", -1)) != 4
        or request.get("weights") != "ema"
        or request.get("precision") != "bf16"
        or cofitok_checkpoint_evaluation.get("weights") != "ema"
        or int(mechanism.get("evaluated_images", -1)) != 256
    ):
        raise ValueError("capacity scaling CoFiTok mechanism source differs")
    utilization = _coarse_token_utilization(
        cofitok_checkpoint_evaluation,
        label="base256_cofitok_step50000",
    )
    mechanism_valid = bool(
        cofitok_row["zero_token_max_abs"] == 0.0
        and cofitok_row["ordered_rank_by_path_auc"] == 1
        and cofitok_row["order_count"] >= 6
        and cofitok_row["shuffled_to_ordered_endpoint_ratio"] > 1.0
        and utilization["passed"] is True
    )

    ten_k_arms = capacity_probe_result.get("evaluation", {}).get("arms", {})
    if not isinstance(ten_k_arms, Mapping):
        raise ValueError("capacity scaling source 10K arms are missing")
    ten_k = {}
    for arm, method in (
        ("base256_cofitok", "cofitok"),
        ("base256_dense_identity", "dense_identity"),
    ):
        row = ten_k_arms.get(arm)
        if not isinstance(row, Mapping):
            raise ValueError(f"capacity scaling source arm is missing: {arm}")
        ten_k[method] = _finite(row.get("fid"), label=f"{arm} step-10K FID")
    cofitok_delta = milestone["methods"]["cofitok"]["fid"] - ten_k["cofitok"]
    dense_delta = (
        milestone["methods"]["dense_identity"]["fid"]
        - ten_k["dense_identity"]
    )
    shared_improvement = cofitok_delta < 0.0 and dense_delta < 0.0
    quality_alerts = milestone["quality_alerts"]
    quality_valid = not quality_alerts
    completion_supported = shared_improvement and mechanism_valid and quality_valid
    if completion_supported:
        recommendation = {
            "id": CAPACITY_COMPLETION_PREPARATION_ID,
            "category": "capacity_completion_supported",
            "execution_ready": False,
            "configured_100k_completion_allowed": False,
            "full_300k_launch_allowed": False,
            "reason": (
                "Both matched base256 methods strictly improved from 10K to 50K, "
                "the matched quality screen passed, and CoFiTok mechanism invariants "
                "remained valid."
            ),
        }
    elif not mechanism_valid:
        recommendation = {
            "id": CAPACITY_COMPLETION_MECHANISM_HOLD_ID,
            "category": "factorization_mechanism_recovery",
            "execution_ready": False,
            "configured_100k_completion_allowed": False,
            "full_300k_launch_allowed": False,
            "reason": "Step-50K CoFiTok factorization mechanism invariants failed.",
        }
    else:
        recommendation = {
            "id": CAPACITY_COMPLETION_QUALITY_HOLD_ID,
            "category": "capacity_quality_not_supported",
            "execution_ready": False,
            "configured_100k_completion_allowed": False,
            "full_300k_launch_allowed": False,
            "reason": (
                "The matched 10K-to-50K quality trend or the step-50K matched "
                "quality screen did not support further optimization."
            ),
        }
    execution_git = _git(
        launch_receipt.get("git", {}),
        revision=expected_execution_revision,
        branch=expected_execution_branch,
        with_tree=True,
        label="capacity scaling result execution",
    )
    builder_git = _git(
        result_builder_git,
        revision=str(result_builder_git.get("revision", "")),
        branch=str(result_builder_git.get("branch", "")),
        with_tree=True,
        label="capacity scaling result builder",
    )
    return {
        "schema_version": CAPACITY_SCALING_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "role": CAPACITY_SCALING_RESULT_ROLE,
        "git": execution_git,
        "result_builder_git": builder_git,
        "training_git": launch_receipt.get("training_git"),
        "output_root": output_root,
        "source_reports": sources,
        "launch_selection": copy.deepcopy(launch_receipt["selection"]),
        "partial_training": training,
        "evaluation": {
            "source_step": CAPACITY_PROBE_STOP_STEP,
            "target_step": CAPACITY_SCALING_TARGET_STEP,
            "sample_count_per_method": CAPACITY_PROBE_SAMPLES_PER_ARM,
            "step_10000_fid": ten_k,
            "step_50000": milestone,
            "cofitok_coarse_token_utilization": utilization,
        },
        "estimands": {
            "cofitok_fid_delta_50000_minus_10000": cofitok_delta,
            "dense_fid_delta_50000_minus_10000": dense_delta,
            "optimization_by_factorization_fid_interaction": (
                cofitok_delta - dense_delta
            ),
            "cofitok_fid_relative_change": cofitok_delta
            / max(ten_k["cofitok"], 1e-12),
            "dense_fid_relative_change": dense_delta
            / max(ten_k["dense_identity"], 1e-12),
        },
        "decision": {
            "shared_strict_fid_improvement": shared_improvement,
            "step_50000_quality_alerts": copy.deepcopy(quality_alerts),
            "step_50000_quality_valid": quality_valid,
            "cofitok_mechanism_invariants_valid": mechanism_valid,
            "capacity_completion_supported": completion_supported,
            "recommendation": recommendation,
        },
        "claim_policy": {
            "role": "non_claim_capacity_scaling_diagnostic",
            "sample_count_per_method": CAPACITY_PROBE_SAMPLES_PER_ARM,
            "formal_generation_claim_allowed": False,
            "cross_stage_numeric_ranking_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_RESULT_BOUNDARY
        ),
    }


def validate_capacity_scaling_50k_result(
    report: Mapping[str, Any],
    *,
    expected_execution_revision: str,
    expected_execution_tree: str,
    expected_execution_branch: str,
    expected_result_revision: str,
    expected_result_tree: str,
    expected_result_branch: str,
) -> dict[str, Any]:
    decision = report.get("decision")
    evaluation = report.get("evaluation")
    training = report.get("partial_training")
    sources = report.get("source_reports")
    claim_policy = report.get("claim_policy")
    output_root = report.get("output_root")
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_SCALING_RESULT_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("role") != CAPACITY_SCALING_RESULT_ROLE
        or report.get("authorization_boundary")
        != CAPACITY_SCALING_RESULT_BOUNDARY
        or claim_policy
        != {
            "role": "non_claim_capacity_scaling_diagnostic",
            "sample_count_per_method": CAPACITY_PROBE_SAMPLES_PER_ARM,
            "formal_generation_claim_allowed": False,
            "cross_stage_numeric_ranking_allowed": False,
        }
        or not isinstance(output_root, str)
        or not PurePosixPath(output_root).is_absolute()
        or not isinstance(decision, Mapping)
        or not isinstance(evaluation, Mapping)
        or not isinstance(training, Mapping)
        or set(training) != {"cofitok", "dense_identity"}
        or not isinstance(sources, Mapping)
        or set(sources) != CAPACITY_SCALING_RESULT_SOURCE_NAMES
    ):
        raise ValueError("capacity scaling 50K result contract differs")
    git = _git(
        report.get("git", {}),
        revision=expected_execution_revision,
        branch=expected_execution_branch,
        with_tree=True,
        label="capacity scaling 50K result",
    )
    if git["tree"] != expected_execution_tree:
        raise ValueError("capacity scaling 50K result Git tree differs")
    builder_git = _git(
        report.get("result_builder_git", {}),
        revision=expected_result_revision,
        branch=expected_result_branch,
        with_tree=True,
        label="capacity scaling 50K result builder",
    )
    if builder_git["tree"] != expected_result_tree:
        raise ValueError("capacity scaling 50K result builder tree differs")
    for name, identity in sources.items():
        _identity(identity, label=f"capacity scaling result source {name}")
    ten_k = evaluation.get("step_10000_fid")
    step_50000 = evaluation.get("step_50000")
    fifty = step_50000.get("methods") if isinstance(step_50000, Mapping) else None
    if (
        int(evaluation.get("source_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or int(evaluation.get("target_step", -1)) != CAPACITY_SCALING_TARGET_STEP
        or int(evaluation.get("sample_count_per_method", -1))
        != CAPACITY_PROBE_SAMPLES_PER_ARM
        or not isinstance(ten_k, Mapping)
        or set(ten_k) != {"cofitok", "dense_identity"}
        or not isinstance(step_50000, Mapping)
        or not isinstance(fifty, Mapping)
        or set(fifty) != {"cofitok", "dense_identity"}
        or step_50000.get("source_reports", {}).get("cofitok_checkpoint_eval")
        != sources["cofitok_checkpoint_eval"]
    ):
        raise ValueError("capacity scaling 50K result evaluation differs")
    cofitok_10k = _finite(ten_k["cofitok"], label="result CoFiTok 10K FID")
    dense_10k = _finite(ten_k["dense_identity"], label="result dense 10K FID")
    cofitok_50k = _finite(
        fifty["cofitok"].get("fid"), label="result CoFiTok 50K FID"
    )
    dense_50k = _finite(
        fifty["dense_identity"].get("fid"), label="result dense 50K FID"
    )
    if min(cofitok_10k, dense_10k, cofitok_50k, dense_50k) < 0.0:
        raise ValueError("capacity scaling 50K result FID domain differs")
    cofitok_delta = cofitok_50k - cofitok_10k
    dense_delta = dense_50k - dense_10k
    estimands = report.get("estimands")
    if (
        not isinstance(estimands, Mapping)
        or set(estimands)
        != {
            "cofitok_fid_delta_50000_minus_10000",
            "dense_fid_delta_50000_minus_10000",
            "optimization_by_factorization_fid_interaction",
            "cofitok_fid_relative_change",
            "dense_fid_relative_change",
        }
        or not (
        math.isclose(
            float(estimands.get("cofitok_fid_delta_50000_minus_10000", math.nan)),
            cofitok_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(estimands.get("dense_fid_delta_50000_minus_10000", math.nan)),
            dense_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(
                estimands.get(
                    "optimization_by_factorization_fid_interaction", math.nan
                )
            ),
            cofitok_delta - dense_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(estimands.get("cofitok_fid_relative_change", math.nan)),
            cofitok_delta / max(cofitok_10k, 1e-12),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(estimands.get("dense_fid_relative_change", math.nan)),
            dense_delta / max(dense_10k, 1e-12),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        )
    ):
        raise ValueError("capacity scaling 50K result estimands differ")
    shared = cofitok_delta < 0.0 and dense_delta < 0.0
    alerts = decision.get("step_50000_quality_alerts")
    mechanism = decision.get("cofitok_mechanism_invariants_valid")
    quality = decision.get("step_50000_quality_valid")
    supported = decision.get("capacity_completion_supported")
    recommendation = decision.get("recommendation")
    utilization = _utilization_summary(
        evaluation.get("cofitok_coarse_token_utilization")
    )
    cofitok_row = fifty["cofitok"]
    recomputed_mechanism = bool(
        _finite(
            cofitok_row.get("zero_token_max_abs"),
            label="result CoFiTok zero token",
        )
        == 0.0
        and int(cofitok_row.get("ordered_rank_by_path_auc", -1)) == 1
        and int(cofitok_row.get("order_count", -1)) >= 6
        and _finite(
            cofitok_row.get("shuffled_to_ordered_endpoint_ratio"),
            label="result CoFiTok shuffle ratio",
        )
        > 1.0
        and utilization["passed"] is True
    )
    expected_alerts = []
    if (cofitok_50k - dense_50k) / max(dense_50k, 1e-12) > (
        CAPACITY_SCALING_MAX_COFITOK_FID_RELATIVE_GAP
    ):
        expected_alerts.append("cofitok_fid_more_than_25pct_above_dense")
    if int(cofitok_row.get("ordered_rank_by_path_auc", -1)) != 1:
        expected_alerts.append("cofitok_ordered_prefix_not_rank1")
    if float(cofitok_row.get("zero_token_max_abs", math.nan)) != 0.0:
        expected_alerts.append("cofitok_zero_token_contract_failed")
    if float(
        cofitok_row.get("shuffled_to_ordered_endpoint_ratio", math.nan)
    ) <= 1.0:
        expected_alerts.append("cofitok_shuffle_mismatch_not_detected")
    if (
        not isinstance(alerts, list)
        or type(mechanism) is not bool
        or type(quality) is not bool
        or type(supported) is not bool
        or alerts != expected_alerts
        or alerts != step_50000.get("quality_alerts")
        or mechanism is not recomputed_mechanism
        or decision.get("shared_strict_fid_improvement") is not shared
        or quality is not (not alerts)
        or supported is not (shared and quality and mechanism)
        or not isinstance(recommendation, Mapping)
        or recommendation.get("execution_ready") is not False
        or recommendation.get("configured_100k_completion_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("capacity scaling 50K decision differs")
    expected_id = (
        CAPACITY_COMPLETION_PREPARATION_ID
        if supported
        else (
            CAPACITY_COMPLETION_MECHANISM_HOLD_ID
            if not mechanism
            else CAPACITY_COMPLETION_QUALITY_HOLD_ID
        )
    )
    if recommendation.get("id") != expected_id:
        raise ValueError("capacity scaling 50K recommendation differs")
    expected_category = {
        CAPACITY_COMPLETION_PREPARATION_ID: "capacity_completion_supported",
        CAPACITY_COMPLETION_MECHANISM_HOLD_ID: "factorization_mechanism_recovery",
        CAPACITY_COMPLETION_QUALITY_HOLD_ID: "capacity_quality_not_supported",
    }[expected_id]
    if recommendation.get("category") != expected_category:
        raise ValueError("capacity scaling 50K recommendation category differs")
    launch_selection = report.get("launch_selection")
    if not isinstance(launch_selection, Mapping):
        raise ValueError("capacity scaling 50K launch selection is missing")
    launch_sources = launch_selection.get("resume_sources")
    if not isinstance(launch_sources, Mapping):
        raise ValueError("capacity scaling 50K launch resume sources are missing")
    expected_training_keys = {
        "source_checkpoint",
        "source_checkpoint_integrity_manifest",
        "checkpoint",
        "checkpoint_integrity_manifest",
        "images_seen",
        "parameter_count",
        "last_resume_step",
        "recovery_resume_used",
        "runtime_environment_sha256",
        "dataset_identity_sha256",
    }
    for method in ("cofitok", "dense_identity"):
        row = training[method]
        if not isinstance(row, Mapping) or set(row) != expected_training_keys:
            raise ValueError(f"capacity scaling 50K {method} training differs")
        source_checkpoint = _identity(
            row.get("source_checkpoint", {}),
            label=f"capacity scaling result {method} source checkpoint",
        )
        source_integrity = _identity(
            row.get("source_checkpoint_integrity_manifest", {}),
            label=f"capacity scaling result {method} source integrity",
        )
        checkpoint = _identity(
            row.get("checkpoint", {}),
            label=f"capacity scaling result {method} checkpoint",
        )
        integrity = _identity(
            row.get("checkpoint_integrity_manifest", {}),
            label=f"capacity scaling result {method} integrity",
        )
        milestone_row = fifty[method]
        if (
            launch_sources.get(method, {}).get("checkpoint") != source_checkpoint
            or launch_sources.get(method, {}).get("checkpoint_integrity_manifest")
            != source_integrity
            or checkpoint.get("path") != milestone_row.get("checkpoint")
            or checkpoint.get("sha256") != milestone_row.get("checkpoint_sha256")
            or integrity.get("path")
            != milestone_row.get("checkpoint_integrity_manifest")
            or int(row.get("images_seen", -1))
            != CAPACITY_SCALING_TARGET_STEP * CAPACITY_PROBE_EFFECTIVE_BATCH
            or int(row.get("parameter_count", -1))
            != CAPACITY_PROBE_PARAMETER_COUNTS["base256"][method]
            or not _hex(row.get("runtime_environment_sha256"))
            or not _hex(row.get("dataset_identity_sha256"))
        ):
            raise ValueError(f"capacity scaling 50K {method} evidence differs")
    resume_sources = {
        method: {
            "checkpoint": copy.deepcopy(training[method]["checkpoint"]),
            "checkpoint_integrity_manifest": copy.deepcopy(
                training[method]["checkpoint_integrity_manifest"]
            ),
        }
        for method in ("cofitok", "dense_identity")
    }
    return {
        "capacity_completion_supported": supported,
        "shared_strict_fid_improvement": shared,
        "cofitok_mechanism_invariants_valid": mechanism,
        "quality_alerts": copy.deepcopy(alerts),
        "resume_sources": resume_sources,
        "output_root": output_root,
        "git": git,
        "result_builder_git": builder_git,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_RESULT_BOUNDARY
        ),
    }
