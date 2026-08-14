from __future__ import annotations

import copy
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_completion_decision import (
    validate_capacity_completion_decision,
)
from cofitok.generation.capacity_completion_execution import (
    validate_capacity_completion_launch_receipt,
)
from cofitok.generation.capacity_completion_training import (
    CAPACITY_COMPLETION_TRAINING_BOUNDARY,
    CAPACITY_COMPLETION_TRAINING_ROLE,
)
from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_PARAMETER_COUNTS,
    validate_capacity_probe_preparation_contract,
)
from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_RESULT_BOUNDARY,
    CAPACITY_PROBE_RESULT_ROLE,
)
from cofitok.generation.capacity_scaling_result import (
    validate_capacity_scaling_50k_result,
)
from cofitok.generation.quality_bridge import (
    _matched_sampling_protocol,
    _quality_screen,
    _terminal_method_row,
    _validate_physical_evidence,
    _validate_terminal_preflight,
    validate_quality_bridge_preparation,
)
from cofitok.generation_class_fidelity import (
    validate_class_fidelity_qualification,
)


CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION = 1
CAPACITY_COMPLETION_RESULT_ROLE = (
    "stability_full_data_capacity_completion_100k_result"
)
CAPACITY_COMPLETION_RESULT_SOURCE_NAMES = {
    "quality_bridge_preparation",
    "capacity_probe_preparation",
    "capacity_probe_result",
    "capacity_scaling_50k_result",
    "capacity_completion_decision",
    "capacity_completion_launch_receipt",
    "capacity_completion_execution_status",
    "source_checkpoint_archive",
    "cofitok_training_validation",
    "dense_identity_training_validation",
    "milestone_50000",
    "milestone_100000",
    "cofitok_sampling_preflight",
    "dense_sampling_preflight",
    "cofitok_generation",
    "dense_generation",
    "cofitok_checkpoint_eval",
    "dense_checkpoint_eval",
    "class_fidelity_qualification",
    "cofitok_class_fidelity",
    "dense_identity_class_fidelity",
}
CAPACITY_COMPLETION_RESULT_BOUNDARY = {
    "capacity_completion_100k_evidence_complete": True,
    "additional_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "gpu_execution_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "new_source_compatible_gate_or_decision_required": True,
}
CAPACITY_COMPLETION_EXECUTION_BOUNDARY = {
    "matched_250m_resume_to_100000_allowed": True,
    "configured_100k_training_complete": True,
    "terminal_evidence_complete": True,
    "additional_training_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "new_source_compatible_result_required": True,
}
ABSOLUTE_QUALITY_CHECKS = {
    "cofitok_absolute_fid",
    "cofitok_precision_floor",
    "cofitok_recall_floor",
}
MATCHED_QUALITY_CHECKS = {
    "matched_fid_tolerance",
    "matched_precision_tolerance",
    "matched_recall_tolerance",
    "matched_endpoint_tolerance",
}
MECHANISM_CHECKS = {
    "ordered_prefix_rank",
    "coarse_token_utilization",
    "restricted_synthesis_zero_token",
    "shuffle_mismatch",
}
CLASS_FIDELITY_CHECKS = {"class_fidelity"}


def _hex(value: Any, *, length: int = 64) -> bool:
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
        or not _hex(digest)
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


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _training(
    report: Mapping[str, Any],
    *,
    method: str,
    expected_revision: str,
    expected_branch: str,
    source_archive: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    source = report.get("source_checkpoint")
    expected_parameters = CAPACITY_PROBE_PARAMETER_COUNTS["base256"][method]
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "pass"
        or report.get("role") != CAPACITY_COMPLETION_TRAINING_ROLE
        or report.get("method") != method
        or report.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or int(report.get("source_step", -1)) != 50_000
        or int(report.get("completed_steps", -1)) != 100_000
        or report.get("training_complete") is not True
        or report.get("exact_resume") is not True
        or int(report.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or int(report.get("images_seen", -1))
        != 100_000 * CAPACITY_PROBE_EFFECTIVE_BATCH
        or int(report.get("validation_event_count", -1)) != 100
        or int(report.get("parameter_count", -1)) != expected_parameters
        or report.get("source_checkpoint_archive") != dict(source_archive)
        or report.get("authorization_boundary")
        != CAPACITY_COMPLETION_TRAINING_BOUNDARY
        or not isinstance(checkpoint, Mapping)
        or not isinstance(source, Mapping)
    ):
        raise ValueError(f"capacity completion {method} training validation differs")
    checkpoint_identity = _identity(
        checkpoint,
        label=f"capacity completion {method} checkpoint",
    )
    checkpoint_integrity = _identity(
        checkpoint.get("integrity_manifest", {}),
        label=f"capacity completion {method} checkpoint integrity",
    )
    source_identity = _identity(
        source,
        label=f"capacity completion {method} source checkpoint",
    )
    source_integrity = _identity(
        source.get("integrity_manifest", {}),
        label=f"capacity completion {method} source integrity",
    )
    if (
        int(checkpoint.get("step", -1)) != 100_000
        or not checkpoint_identity["path"].endswith(
            "checkpoint_step_00100000.pt"
        )
        or checkpoint_integrity["path"]
        != f"{checkpoint_identity['path']}.integrity.json"
        or int(source.get("step", -1)) != 50_000
        or not source_identity["path"].endswith("checkpoint_step_00050000.pt")
        or source_integrity["path"]
        != f"{source_identity['path']}.integrity.json"
    ):
        raise ValueError(f"capacity completion {method} checkpoint identity differs")
    return {
        "source_checkpoint": source_identity,
        "source_checkpoint_integrity_manifest": source_integrity,
        "checkpoint": checkpoint_identity,
        "checkpoint_integrity_manifest": checkpoint_integrity,
        "images_seen": int(report["images_seen"]),
        "validation_event_count": int(report["validation_event_count"]),
        "parameter_count": expected_parameters,
        "last_resume_step": int(report.get("last_resume_step", -1)),
        "recovery_resume_used": report.get("recovery_resume_used"),
        "runtime_environment_sha256": report.get(
            "runtime_environment_sha256"
        ),
        "dataset_identity_sha256": report.get("dataset_identity_sha256"),
    }


def _milestone(
    report: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    step: int,
) -> dict[str, Any]:
    methods = report.get("methods")
    sources = report.get("source_reports")
    evidence = verification.get("evidence")
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("status") != "completed"
        or report.get("role") != "training_quality_trend_only"
        or report.get("source_profile") != "capacity_scaling"
        or int(report.get("milestone_step", -1)) != step
        or int(report.get("expected_samples", -1)) != 2_048
        or report.get("claim_policy", {}).get(
            "formal_generation_claim_allowed"
        )
        is not False
        or not isinstance(methods, Mapping)
        or set(methods) != {"cofitok", "dense_identity"}
        or not isinstance(sources, Mapping)
        or verification.get("status") != "verified"
        or not isinstance(evidence, Mapping)
        or evidence.get("status") != "verified"
        or evidence.get("source_profile") != "capacity_scaling"
        or evidence.get("quality_alerts") != report.get("quality_alerts")
    ):
        raise ValueError(f"capacity completion milestone {step} differs")
    normalized = {}
    for method, budget in (("cofitok", 8), ("dense_identity", 1)):
        row = methods[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"capacity completion milestone {method} is malformed")
        fid = _finite(row.get("fid"), label=f"{method} milestone FID")
        inception = _finite(
            row.get("inception_score"),
            label=f"{method} milestone Inception Score",
        )
        if (
            int(row.get("checkpoint_step", -1)) != step
            or int(row.get("sample_count", -1)) != 2_048
            or int(row.get("selected_prefix_budget", -1)) != budget
            or row.get("weights") != "ema"
            or not _hex(row.get("checkpoint_sha256"))
            or not _hex(row.get("sample_set_sha256"))
            or fid < 0.0
            or inception <= 0.0
        ):
            raise ValueError(
                f"capacity completion milestone {method} identity differs"
            )
        normalized[method] = {
            "checkpoint": row.get("checkpoint"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "checkpoint_integrity_manifest": row.get(
                "checkpoint_integrity_manifest"
            ),
            "sample_count": int(row["sample_count"]),
            "sample_set_sha256": row.get("sample_set_sha256"),
            "selected_prefix_budget": budget,
            "weights": "ema",
            "sampling": copy.deepcopy(row.get("sampling")),
            "fid": fid,
            "inception_score": inception,
            "endpoint_clean_mse": _finite(
                row.get("endpoint_clean_mse"),
                label=f"{method} milestone endpoint MSE",
            ),
            "prefix_path_mse_auc": _finite(
                row.get("prefix_path_mse_auc"),
                label=f"{method} milestone path AUC",
            ),
            "ordered_rank_by_path_auc": int(
                row.get("ordered_rank_by_path_auc", -1)
            ),
            "order_count": int(row.get("order_count", -1)),
            "zero_token_max_abs": _finite(
                row.get("zero_token_max_abs"),
                label=f"{method} milestone zero token",
            ),
            "shuffled_to_ordered_endpoint_ratio": _finite(
                row.get("shuffled_to_ordered_endpoint_ratio"),
                label=f"{method} milestone shuffle ratio",
            ),
        }
    return {
        "step": step,
        "methods": normalized,
        "quality_alerts": copy.deepcopy(report.get("quality_alerts", [])),
        "source_reports": copy.deepcopy(dict(sources)),
        "verification": copy.deepcopy(dict(verification)),
    }


def _policy(
    *,
    quality_status: str,
    failed_checks: list[str],
    milestone_alerts: list[str],
    shared_strict_fid_improvement: bool,
) -> dict[str, Any]:
    failed = set(failed_checks)
    if quality_status == "pass" and not milestone_alerts:
        recommendation_id = "build_source_compatible_formal_quality_gate"
        category = "terminal_quality_supported"
        reason = (
            "The terminal quality, matched distribution, mechanism, and class-"
            "fidelity screen passed without a milestone contradiction."
        )
    elif milestone_alerts:
        recommendation_id = "reconcile_50k_100k_milestone_terminal_evidence"
        category = "milestone_terminal_contradiction"
        reason = (
            "The step-100K trend milestone contains an alert and must be "
            "reconciled with the terminal DDIM-100 evidence before more training."
        )
    elif failed & MECHANISM_CHECKS:
        recommendation_id = "run_factorization_mechanism_recovery_diagnostic"
        category = "factorization_mechanism_failure"
        reason = "At least one ordered restricted-factorization invariant failed."
    elif failed & MATCHED_QUALITY_CHECKS:
        recommendation_id = "run_matched_factorization_quality_regression_probe"
        category = "matched_quality_regression"
        reason = "The terminal CoFiTok result exceeded a matched dense tolerance."
    elif failed & CLASS_FIDELITY_CHECKS:
        recommendation_id = "run_class_conditioning_fidelity_diagnostic"
        category = "class_fidelity_failure"
        reason = "The matched terminal class-fidelity qualification did not pass."
    elif failed and failed <= ABSOLUTE_QUALITY_CHECKS and shared_strict_fid_improvement:
        recommendation_id = (
            "build_source_compatible_250m_full_300k_readiness_decision"
        )
        category = "scale_responsive_absolute_quality_hold"
        reason = (
            "Only absolute quality floors remain unmet while both matched methods "
            "strictly improve from step 50K to 100K under the same milestone protocol."
        )
    elif failed and failed <= ABSOLUTE_QUALITY_CHECKS:
        recommendation_id = "revisit_matched_training_objective_before_more_scale"
        category = "absolute_quality_not_scale_responsive"
        reason = (
            "Absolute quality remains below threshold without shared strict "
            "50K-to-100K FID improvement."
        )
    else:
        recommendation_id = "extend_capacity_completion_policy_before_execution"
        category = "unclassified_terminal_failure"
        reason = "The terminal failure combination is not execution-classified."
    return {
        "id": recommendation_id,
        "category": category,
        "reason": reason,
        "execution_ready": False,
        "gpu_execution_allowed": False,
        "training_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_or_release_allowed": False,
        "new_source_compatible_gate_or_decision_required": True,
    }


def build_capacity_completion_100k_result(
    *,
    quality_bridge_preparation: Mapping[str, Any],
    capacity_probe_preparation: Mapping[str, Any],
    capacity_probe_result: Mapping[str, Any],
    capacity_scaling_result: Mapping[str, Any],
    capacity_completion_decision: Mapping[str, Any],
    capacity_completion_launch_receipt: Mapping[str, Any],
    capacity_completion_execution_status: Mapping[str, Any],
    source_checkpoint_archive: Mapping[str, Any],
    training_validations: Mapping[str, Mapping[str, Any]],
    milestone_reports: Mapping[int, Mapping[str, Any]],
    milestone_verifications: Mapping[int, Mapping[str, Any]],
    sampling_preflights: Mapping[str, Mapping[str, Any]],
    generation_reports: Mapping[str, Mapping[str, Any]],
    checkpoint_evaluations: Mapping[str, Mapping[str, Any]],
    class_fidelity_qualification: Mapping[str, Any],
    physical_evidence: Mapping[str, Mapping[str, Any]],
    source_identities: Mapping[str, Mapping[str, Any]],
    completion_execution_git: Mapping[str, Any],
    training_git: Mapping[str, Any],
    result_builder_git: Mapping[str, Any],
    expected_completion_decision_revision: str,
    expected_completion_decision_tree: str,
    expected_completion_decision_branch: str,
    expected_scaling_execution_revision: str,
    expected_scaling_execution_tree: str,
    expected_scaling_execution_branch: str,
    expected_scaling_result_revision: str,
    expected_scaling_result_tree: str,
    expected_scaling_result_branch: str,
) -> dict[str, Any]:
    if set(source_identities) != CAPACITY_COMPLETION_RESULT_SOURCE_NAMES:
        raise ValueError("capacity completion result source set differs")
    sources = {
        name: _identity(identity, label=f"capacity completion result {name}")
        for name, identity in source_identities.items()
    }
    execution_git = _git(
        completion_execution_git,
        revision=str(completion_execution_git.get("revision", "")),
        tree=str(completion_execution_git.get("tree", "")),
        branch=str(completion_execution_git.get("branch", "")),
        label="capacity completion execution",
    )
    normalized_training_git = _git(
        training_git,
        revision=str(training_git.get("revision", "")),
        tree=str(training_git.get("tree", "")),
        branch=str(training_git.get("branch", "")),
        label="capacity completion training",
    )
    builder_git = _git(
        result_builder_git,
        revision=str(result_builder_git.get("revision", "")),
        tree=str(result_builder_git.get("tree", "")),
        branch=str(result_builder_git.get("branch", "")),
        label="capacity completion result builder",
    )
    scaling = validate_capacity_scaling_50k_result(
        capacity_scaling_result,
        expected_execution_revision=expected_scaling_execution_revision,
        expected_execution_tree=expected_scaling_execution_tree,
        expected_execution_branch=expected_scaling_execution_branch,
        expected_result_revision=expected_scaling_result_revision,
        expected_result_tree=expected_scaling_result_tree,
        expected_result_branch=expected_scaling_result_branch,
    )
    decision = validate_capacity_completion_decision(
        capacity_completion_decision,
        expected_decision_revision=expected_completion_decision_revision,
        expected_decision_tree=expected_completion_decision_tree,
        expected_decision_branch=expected_completion_decision_branch,
    )
    output_root = decision["output_root"]
    launch = validate_capacity_completion_launch_receipt(
        capacity_completion_launch_receipt,
        expected_execution_revision=execution_git["revision"],
        expected_execution_tree=execution_git["tree"],
        expected_execution_branch=execution_git["branch"],
        expected_training_revision=normalized_training_git["revision"],
        expected_training_tree=normalized_training_git["tree"],
        expected_training_branch=normalized_training_git["branch"],
        expected_output_root=output_root,
    )
    if (
        decision["execution_authorized"] is not True
        or scaling["capacity_completion_supported"] is not True
        or capacity_completion_launch_receipt.get("source_reports", {}).get(
            "capacity_completion_decision"
        )
        != sources["capacity_completion_decision"]
        or capacity_completion_launch_receipt.get("source_reports", {}).get(
            "capacity_scaling_50k_result"
        )
        != sources["capacity_scaling_50k_result"]
        or capacity_completion_launch_receipt.get("source_reports", {}).get(
            "source_checkpoint_archive"
        )
        != sources["source_checkpoint_archive"]
    ):
        raise ValueError("capacity completion result launch chain differs")
    if (
        capacity_scaling_result.get("source_reports", {}).get(
            "capacity_probe_result"
        )
        != sources["capacity_probe_result"]
        or capacity_probe_result.get("status") != "completed"
        or capacity_probe_result.get("role") != CAPACITY_PROBE_RESULT_ROLE
        or capacity_probe_result.get("authorization_boundary")
        != CAPACITY_PROBE_RESULT_BOUNDARY
        or capacity_probe_result.get("source_reports", {}).get("preparation")
        != sources["capacity_probe_preparation"]
    ):
        raise ValueError("capacity completion capacity-probe source chain differs")
    validate_capacity_probe_preparation_contract(
        capacity_probe_preparation,
        expected_output_root=output_root,
    )
    if capacity_probe_preparation.get("source_evidence", {}).get(
        "quality_bridge_preparation"
    ) != sources["quality_bridge_preparation"]:
        raise ValueError("capacity completion quality-bridge source differs")
    bridge = validate_quality_bridge_preparation(quality_bridge_preparation)
    if bridge["steps"] != 100_000 or bridge["terminal_samples"] != 10_000:
        raise ValueError("capacity completion frozen quality policy differs")

    archive_identity = sources["source_checkpoint_archive"]
    if (
        source_checkpoint_archive.get("status") != "pass"
        or source_checkpoint_archive.get("role")
        != "capacity_completion_50k_source_checkpoint_archive"
        or source_checkpoint_archive.get("capacity_completion_decision")
        != sources["capacity_completion_decision"]
        or source_checkpoint_archive.get("output_root") != output_root
    ):
        raise ValueError("capacity completion source archive differs")
    if set(training_validations) != {"cofitok", "dense_identity"}:
        raise ValueError("capacity completion training validation set differs")
    training = {
        method: _training(
            training_validations[method],
            method=method,
            expected_revision=normalized_training_git["revision"],
            expected_branch=normalized_training_git["branch"],
            source_archive=archive_identity,
        )
        for method in ("cofitok", "dense_identity")
    }

    if set(milestone_reports) != {50_000, 100_000} or set(
        milestone_verifications
    ) != {50_000, 100_000}:
        raise ValueError("capacity completion milestone set differs")
    milestones = {
        step: _milestone(
            milestone_reports[step],
            milestone_verifications[step],
            step=step,
        )
        for step in (50_000, 100_000)
    }
    if (
        sources["milestone_50000"]
        != capacity_scaling_result.get("source_reports", {}).get(
            "milestone_50000"
        )
        or milestones[50_000]["methods"]["cofitok"]["checkpoint_sha256"]
        != scaling["resume_sources"]["cofitok"]["checkpoint"]["sha256"]
        or milestones[50_000]["methods"]["dense_identity"][
            "checkpoint_sha256"
        ]
        != scaling["resume_sources"]["dense_identity"]["checkpoint"]["sha256"]
    ):
        raise ValueError("capacity completion step-50K milestone binding differs")
    if _matched_sampling_protocol(
        milestones[50_000]["methods"]["cofitok"]["sampling"]
    ) != _matched_sampling_protocol(
        milestones[100_000]["methods"]["cofitok"]["sampling"]
    ):
        raise ValueError("capacity completion 50K/100K milestone protocols differ")

    expected_git = {
        "revision": normalized_training_git["revision"],
        "branch": normalized_training_git["branch"],
        "tracked_dirty": False,
    }
    if set(generation_reports) != {"cofitok", "dense_identity"} or set(
        checkpoint_evaluations
    ) != {"cofitok", "dense_identity"}:
        raise ValueError("capacity completion terminal method set differs")
    methods = {
        "cofitok": _terminal_method_row(
            generation_reports["cofitok"],
            checkpoint_evaluations["cofitok"],
            label="CoFiTok",
            expected_prefix_budget=8,
            expected_random_orders=4,
            expected_git=expected_git,
        ),
        "dense_identity": _terminal_method_row(
            generation_reports["dense_identity"],
            checkpoint_evaluations["dense_identity"],
            label="dense",
            expected_prefix_budget=1,
            expected_random_orders=0,
            expected_git=expected_git,
        ),
    }
    preflights = {
        "cofitok": _validate_terminal_preflight(
            sampling_preflights["cofitok"],
            label="CoFiTok",
            expected_git=expected_git,
            expected_prefix_budget=8,
        ),
        "dense_identity": _validate_terminal_preflight(
            sampling_preflights["dense_identity"],
            label="dense",
            expected_git=expected_git,
            expected_prefix_budget=1,
        ),
    }
    physical = {
        method: _validate_physical_evidence(
            physical_evidence[method],
            method=methods[method],
            label="CoFiTok" if method == "cofitok" else "dense",
        )
        for method in ("cofitok", "dense_identity")
    }
    if (
        _matched_sampling_protocol(methods["cofitok"]["sampling"])
        != _matched_sampling_protocol(methods["dense_identity"]["sampling"])
        or methods["cofitok"]["real_set"]
        != methods["dense_identity"]["real_set"]
        or methods["cofitok"]["metrics_evaluator_git"]
        != methods["dense_identity"]["metrics_evaluator_git"]
        or methods["cofitok"]["metrics_runtime_environment_sha256"]
        != methods["dense_identity"]["metrics_runtime_environment_sha256"]
        or methods["cofitok"]["sampling_runtime_environment_sha256"]
        != methods["dense_identity"]["sampling_runtime_environment_sha256"]
    ):
        raise ValueError("capacity completion terminal pair is not matched")
    for method in ("cofitok", "dense_identity"):
        if (
            preflights[method]["checkpoint_sha256"]
            != methods[method]["checkpoint_sha256"]
            or preflights[method]["checkpoint_integrity_manifest"]
            != methods[method]["checkpoint_integrity_manifest"]
            or preflights[method]["runtime_environment_sha256"]
            != methods[method]["sampling_runtime_environment_sha256"]
            or training[method]["checkpoint"]["sha256"]
            != methods[method]["checkpoint_sha256"]
            or milestones[100_000]["methods"][method]["checkpoint_sha256"]
            != methods[method]["checkpoint_sha256"]
        ):
            raise ValueError(
                f"capacity completion {method} terminal checkpoint binding differs"
            )

    class_fidelity = validate_class_fidelity_qualification(
        dict(class_fidelity_qualification),
        expected_stage="scaling",
        expected_revision=normalized_training_git["revision"],
        expected_branch=normalized_training_git["branch"],
        require_pass=False,
    )
    expected_class_sources = {
        "cofitok": sources["cofitok_class_fidelity"],
        "dense_identity": sources["dense_identity_class_fidelity"],
    }
    class_sampling = class_fidelity.get("sampling_contract", {})
    if (
        class_fidelity.get("sources") != expected_class_sources
        or class_sampling.get("cofitok_checkpoint_sha256")
        != methods["cofitok"]["checkpoint_sha256"]
        or class_sampling.get("dense_checkpoint_sha256")
        != methods["dense_identity"]["checkpoint_sha256"]
        or class_sampling.get("cofitok_sample_set_sha256")
        != methods["cofitok"]["sample_set_sha256"]
        or class_sampling.get("dense_sample_set_sha256")
        != methods["dense_identity"]["sample_set_sha256"]
    ):
        raise ValueError("capacity completion class-fidelity binding differs")

    status_methods = capacity_completion_execution_status.get("methods")
    if (
        int(capacity_completion_execution_status.get("schema_version", -1)) != 1
        or capacity_completion_execution_status.get("status") != "completed"
        or capacity_completion_execution_status.get("role")
        != "stability_full_data_capacity_completion_100k_execution"
        or capacity_completion_execution_status.get("stage") != "complete"
        or capacity_completion_execution_status.get("git") != execution_git
        or capacity_completion_execution_status.get("output_root") != output_root
        or capacity_completion_execution_status.get("authorization_boundary")
        != CAPACITY_COMPLETION_EXECUTION_BOUNDARY
        or capacity_completion_execution_status.get("launch_receipt")
        != sources["capacity_completion_launch_receipt"]
        or capacity_completion_execution_status.get("source_checkpoint_archive")
        != sources["source_checkpoint_archive"]
        or capacity_completion_execution_status.get("milestone_100000")
        != sources["milestone_100000"]
        or capacity_completion_execution_status.get(
            "class_fidelity_qualification"
        )
        != sources["class_fidelity_qualification"]
        or not isinstance(status_methods, Mapping)
        or set(status_methods) != {"cofitok", "dense_identity"}
    ):
        raise ValueError("capacity completion completed execution status differs")
    for method in ("cofitok", "dense_identity"):
        expected_status_sources = {
            "training_validation": sources[
                f"{method}_training_validation"
            ],
            "sampling_preflight": sources[f"{method}_sampling_preflight"],
            "generation_metrics": sources[f"{method}_generation"],
            "checkpoint_evaluation": sources[f"{method}_checkpoint_eval"],
            "class_fidelity": sources[f"{method}_class_fidelity"],
        }
        if status_methods[method] != expected_status_sources:
            raise ValueError(
                f"capacity completion {method} execution evidence differs"
            )

    quality = _quality_screen(
        cofitok=methods["cofitok"],
        dense=methods["dense_identity"],
        class_fidelity=class_fidelity,
        source_hold=bridge["source_hold"],
    )
    quality["interpretation"] = (
        "This source-bound terminal screen is non-authorizing. Passing it can "
        "only support construction of a new formal gate; a scale-responsive "
        "absolute hold can only support construction of a separate readiness "
        "decision. Neither branch authorizes GPU work by itself."
    )
    fifty = milestones[50_000]["methods"]
    hundred = milestones[100_000]["methods"]
    cofitok_delta = hundred["cofitok"]["fid"] - fifty["cofitok"]["fid"]
    dense_delta = (
        hundred["dense_identity"]["fid"]
        - fifty["dense_identity"]["fid"]
    )
    shared_improvement = cofitok_delta < 0.0 and dense_delta < 0.0
    milestone_alerts = copy.deepcopy(milestones[100_000]["quality_alerts"])
    recommendation = _policy(
        quality_status=quality["status"],
        failed_checks=list(quality["failed_checks"]),
        milestone_alerts=milestone_alerts,
        shared_strict_fid_improvement=shared_improvement,
    )
    return {
        "schema_version": CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "role": CAPACITY_COMPLETION_RESULT_ROLE,
        "output_root": output_root,
        "execution_git": execution_git,
        "training_git": normalized_training_git,
        "result_builder_git": builder_git,
        "source_reports": sources,
        "training": training,
        "milestones": {
            "50000": milestones[50_000],
            "100000": milestones[100_000],
            "trend_50000_to_100000": {
                "protocol_matched": True,
                "cofitok_fid_delta": cofitok_delta,
                "dense_fid_delta": dense_delta,
                "shared_strict_fid_improvement": shared_improvement,
                "step_100000_quality_alerts": milestone_alerts,
            },
        },
        "terminal": {
            "sampling_preflights": preflights,
            "physical_evidence": physical,
            "methods": methods,
            "matched_comparison": {
                "fid_relative_change": (
                    methods["cofitok"]["fid"]
                    - methods["dense_identity"]["fid"]
                )
                / max(methods["dense_identity"]["fid"], 1e-12),
                "precision_delta": methods["cofitok"]["precision"]
                - methods["dense_identity"]["precision"],
                "recall_delta": methods["cofitok"]["recall"]
                - methods["dense_identity"]["recall"],
                "endpoint_clean_mse_relative_change": (
                    methods["cofitok"]["endpoint_clean_mse"]
                    - methods["dense_identity"]["endpoint_clean_mse"]
                )
                / max(methods["dense_identity"]["endpoint_clean_mse"], 1e-12),
            },
            "class_fidelity": class_fidelity,
        },
        "quality_screen": quality,
        "decision_support": {
            "terminal_quality_pass": quality["status"] == "pass",
            "failed_checks": copy.deepcopy(quality["failed_checks"]),
            "step_50000_to_100000_shared_strict_fid_improvement": (
                shared_improvement
            ),
            "step_100000_quality_alerts": milestone_alerts,
            "recommended_next_stage": recommendation,
        },
        "claim_policy": {
            "role": "non_claim_capacity_completion_terminal_diagnostic",
            "terminal_sample_count_per_method": 10_000,
            "milestone_sample_count_per_method": 2_048,
            "cross_protocol_numeric_ranking_allowed": False,
            "formal_generation_claim_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_COMPLETION_RESULT_BOUNDARY
        ),
    }


def validate_capacity_completion_100k_result(
    report: Mapping[str, Any],
    *,
    expected_execution_revision: str,
    expected_execution_tree: str,
    expected_execution_branch: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    expected_result_revision: str,
    expected_result_tree: str,
    expected_result_branch: str,
) -> dict[str, Any]:
    sources = report.get("source_reports")
    decision = report.get("decision_support")
    quality = report.get("quality_screen")
    milestones = report.get("milestones")
    terminal = report.get("terminal")
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("role") != CAPACITY_COMPLETION_RESULT_ROLE
        or report.get("authorization_boundary")
        != CAPACITY_COMPLETION_RESULT_BOUNDARY
        or report.get("claim_policy")
        != {
            "role": "non_claim_capacity_completion_terminal_diagnostic",
            "terminal_sample_count_per_method": 10_000,
            "milestone_sample_count_per_method": 2_048,
            "cross_protocol_numeric_ranking_allowed": False,
            "formal_generation_claim_allowed": False,
        }
        or not isinstance(sources, Mapping)
        or set(sources) != CAPACITY_COMPLETION_RESULT_SOURCE_NAMES
        or not isinstance(decision, Mapping)
        or not isinstance(quality, Mapping)
        or not isinstance(milestones, Mapping)
        or not isinstance(terminal, Mapping)
    ):
        raise ValueError("capacity completion 100K result contract differs")
    execution_git = _git(
        report.get("execution_git", {}),
        revision=expected_execution_revision,
        tree=expected_execution_tree,
        branch=expected_execution_branch,
        label="capacity completion result execution",
    )
    training_git = _git(
        report.get("training_git", {}),
        revision=expected_training_revision,
        tree=expected_training_tree,
        branch=expected_training_branch,
        label="capacity completion result training",
    )
    result_git = _git(
        report.get("result_builder_git", {}),
        revision=expected_result_revision,
        tree=expected_result_tree,
        branch=expected_result_branch,
        label="capacity completion result builder",
    )
    for name, identity in sources.items():
        _identity(identity, label=f"capacity completion result source {name}")
    trend = milestones.get("trend_50000_to_100000")
    failed = quality.get("failed_checks")
    checks = quality.get("checks")
    if (
        quality.get("status") not in {"pass", "hold"}
        or not isinstance(failed, list)
        or not isinstance(checks, list)
        or quality.get("status") != ("pass" if not failed else "hold")
        or not isinstance(trend, Mapping)
        or trend.get("protocol_matched") is not True
        or type(trend.get("shared_strict_fid_improvement")) is not bool
        or not isinstance(trend.get("step_100000_quality_alerts"), list)
    ):
        raise ValueError("capacity completion result quality evidence differs")
    expected_recommendation = _policy(
        quality_status=str(quality["status"]),
        failed_checks=list(failed),
        milestone_alerts=list(trend["step_100000_quality_alerts"]),
        shared_strict_fid_improvement=bool(
            trend["shared_strict_fid_improvement"]
        ),
    )
    if (
        decision.get("terminal_quality_pass") is not (quality["status"] == "pass")
        or decision.get("failed_checks") != failed
        or decision.get(
            "step_50000_to_100000_shared_strict_fid_improvement"
        )
        is not trend["shared_strict_fid_improvement"]
        or decision.get("step_100000_quality_alerts")
        != trend["step_100000_quality_alerts"]
        or decision.get("recommended_next_stage") != expected_recommendation
    ):
        raise ValueError("capacity completion result next-stage policy differs")
    methods = terminal.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {
        "cofitok",
        "dense_identity",
    }:
        raise ValueError("capacity completion terminal method set differs")
    for method in methods.values():
        if (
            not isinstance(method, Mapping)
            or int(method.get("checkpoint_step", -1)) != 100_000
            or int(method.get("sample_count", -1)) != 10_000
            or method.get("weights") != "ema"
            or not _hex(method.get("checkpoint_sha256"))
            or not _hex(method.get("sample_set_sha256"))
        ):
            raise ValueError("capacity completion terminal method identity differs")
    output_root = report.get("output_root")
    if (
        not isinstance(output_root, str)
        or not (
            PurePosixPath(output_root).is_absolute()
            or Path(output_root).is_absolute()
        )
    ):
        raise ValueError("capacity completion result output root differs")
    return {
        "status": "completed",
        "output_root": output_root,
        "terminal_quality_pass": decision["terminal_quality_pass"],
        "recommended_next_stage": copy.deepcopy(expected_recommendation),
        "execution_git": execution_git,
        "training_git": training_git,
        "result_builder_git": result_git,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_COMPLETION_RESULT_BOUNDARY
        ),
    }
