from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)
from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    EXPECTED_CHECKS,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
    FOLLOWUP_DECISION_ROLE,
    MATCHED_QUALITY_CHECKS,
    QUALITY_BRIDGE_DATASET,
    QUALITY_BRIDGE_EFFECTIVE_BATCH,
    QUALITY_BRIDGE_EXECUTION_BRANCH,
    QUALITY_BRIDGE_EXECUTION_REVISION,
    QUALITY_BRIDGE_IMAGES_SEEN,
    QUALITY_BRIDGE_TARGET_STEPS,
)
from cofitok.generation.stability_qualification import (
    DEFAULT_HIGH_FREQUENCY_TIMESTEPS,
    build_stability_qualification,
)
SCHEMA_VERSION = 1
CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION = 2
PREPARATION_ROLE = "generation_factorization_quality_regression_preparation"
SOURCE_BINDING_ROLE = "generation_factorization_quality_regression_source_binding"
EXECUTION_AUTHORIZATION_ROLE = (
    "generation_factorization_quality_regression_execution_authorization"
)
DIAGNOSTIC_REPORT_ROLE = "generation_factorization_quality_regression_diagnostic"
SCOPE = "imagenet256_full_100k_matched_factorization_rollout_diagnostic_v1"
FOLLOWUP_DECISION_ID = "run_matched_factorization_quality_regression_probe"
FOLLOWUP_DECISION_CATEGORY = "matched_quality_regression"
FOLLOWUP_DECISION_BUILDER_GIT = {
    "revision": "cd78a348769f0efad0d42de063e5b0943444a29b",
    "branch": "analysis/generation-quality-bridge-class-only-route-v2-20260822",
    "tracked_dirty": False,
}
QUALITY_BRIDGE_GIT = {
    "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
    "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
    "tracked_dirty": False,
}
TERMINAL_SYSTEM_GUARD_ROLE = "generation_terminal_system_claim_guard"
QUALITY_BRIDGE_RESULT_ROLE = "stability_full_data_quality_bridge_result"
QUALITY_BRIDGE_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_base128_quality_bridge_v1"
)
FOLLOWUP_DECISION_PATH = (
    f"{QUALITY_BRIDGE_ROOT}/reports/"
    "followup_experiment_decision_exposure_aware_v2.json"
)
OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_factorization_quality_regression_v1"
)
RUN_DIRS = {
    "cofitok": f"{QUALITY_BRIDGE_ROOT}/cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
    "dense_identity": f"{QUALITY_BRIDGE_ROOT}/dense_rollout_x0_u2_ema_teacher",
}
CHECKPOINT_STEP = 100_000
SEEDS = (2029, 2039)
PROTOCOL = {
    "dataset": "imagenet_256",
    "checkpoint_step": CHECKPOINT_STEP,
    "weights": "ema",
    "num_images_per_seed": 64,
    "batch_size": 8,
    "sample_steps": 100,
    "teacher_timesteps": [999, 900, 750, 500, 250, 100, 10],
    "seeds": list(SEEDS),
    "guidance_scale": 1.5,
    "teacher_guidance_scale": 1.0,
    "guidance_rescale": 0.0,
    "cfg_batch_mode": "batched",
    "clip_x0": True,
    "precision": "bf16",
}
PREPARATION_BOUNDARY = {
    "diagnostic_preparation_only": True,
    "gpu_execution_authorized": False,
    "training_launch_allowed": False,
    "checkpoint_promotion_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
}
SOURCE_BINDING_BOUNDARY = {
    "source_validation_only": True,
    "checkpoint_payload_hash_deferred_to_trusted_loader": True,
    "gpu_execution_authorized": False,
    "training_launch_allowed": False,
    "checkpoint_promotion_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
}
EXECUTION_BOUNDARY = {
    "matched_rollout_diagnostic_allowed": True,
    "gpu_execution_allowed": True,
    "checkpoint_evaluation_allowed": True,
    "published_generation_sampling_allowed": False,
    "training_launch_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_experiment_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
    "process_signaling_allowed": False,
}
CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "localizes_rollout_regression": True,
    "causal_factorization_attribution_allowed": False,
    "quality_advantage_claim_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "training_launch_allowed": False,
    "followup_experiment_launch_allowed": False,
    "checkpoint_promotion_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
}


def _is_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
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
        or not _is_hex(digest, length=64)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not _is_hex(revision, length=40)
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


def _absolute_posix(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute normalized POSIX path")
    return path.as_posix()


def _finite(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}")
    return number


def build_preparation(
    *,
    execution_git: Mapping[str, Any],
    quality_bridge_root: str = QUALITY_BRIDGE_ROOT,
    output_root: str = OUTPUT_ROOT,
) -> dict[str, Any]:
    git = _git(execution_git, label="factorization-regression preparation")
    quality_root = _absolute_posix(quality_bridge_root, label="quality bridge root")
    output = _absolute_posix(output_root, label="diagnostic output root")
    if quality_root != QUALITY_BRIDGE_ROOT or output != OUTPUT_ROOT:
        raise ValueError("factorization-regression roots differ from the frozen contract")
    return {
        "schema_version": SCHEMA_VERSION,
        "role": PREPARATION_ROLE,
        "status": "prepared",
        "scope": SCOPE,
        "git": git,
        "quality_bridge_git": copy.deepcopy(QUALITY_BRIDGE_GIT),
        "quality_bridge_root": quality_root,
        "output_root": output,
        "run_dirs": copy.deepcopy(RUN_DIRS),
        "protocol": copy.deepcopy(PROTOCOL),
        "scientific_objective": (
            "Localize whether a matched terminal distribution-quality regression is "
            "accompanied by repeatable CoFiTok-specific teacher-forced, reconstruction, "
            "high-frequency, or component-energy instability on the exact 100K EMA pair."
        ),
        "interpretation_limit": (
            "This observational checkpoint diagnostic cannot causally attribute a quality "
            "difference to a factorization-only loss or token layout."
        ),
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }


def validate_preparation(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str = OUTPUT_ROOT,
) -> dict[str, Any]:
    expected = build_preparation(
        execution_git={
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        },
        output_root=expected_output_root,
    )
    if dict(report) != expected:
        raise ValueError("factorization-regression preparation contract differs")
    return copy.deepcopy(expected)


def _validate_followup_training_exposure(
    value: Any,
    *,
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("quality-bridge follow-up training exposure is missing")
    expected_keys = {
        "source_report",
        "dataset",
        "dataset_identity_sha256",
        "train_image_count",
        "steps_per_method",
        "effective_batch_size",
        "images_seen_per_method",
        "full_data_equivalent_epochs",
        "historical_10pct_reference_equivalent_epochs",
        "full_to_reference_equivalent_epoch_ratio",
        "below_historical_reference_exposure",
        "insufficient_exposure_is_live_hypothesis",
        "causal_status",
        "quality_metric_comparison_allowed",
        "terminal_result_content_bound",
        "terminal_checkpoint_binding_verified",
    }
    if set(value) != expected_keys:
        raise ValueError("quality-bridge follow-up training exposure field set differs")
    source = _identity(value.get("source_report", {}), label="terminal training exposure")
    if source != _identity(expected_source, label="expected terminal training exposure"):
        raise ValueError("quality-bridge follow-up training exposure source differs")
    train_image_count = value.get("train_image_count")
    steps = value.get("steps_per_method")
    effective_batch = value.get("effective_batch_size")
    images_seen = value.get("images_seen_per_method")
    if (
        type(train_image_count) is not int
        or train_image_count < 1
        or type(steps) is not int
        or type(effective_batch) is not int
        or type(images_seen) is not int
    ):
        raise ValueError("quality-bridge follow-up training exposure counts differ")
    full_epochs = _finite(
        value.get("full_data_equivalent_epochs"),
        label="full-data equivalent epochs",
        positive=True,
    )
    reference_epochs = _finite(
        value.get("historical_10pct_reference_equivalent_epochs"),
        label="historical 10pct equivalent epochs",
        positive=True,
    )
    ratio = _finite(
        value.get("full_to_reference_equivalent_epoch_ratio"),
        label="full-to-reference equivalent epoch ratio",
        positive=True,
    )
    below_reference = full_epochs < reference_epochs
    if (
        value.get("dataset") != QUALITY_BRIDGE_DATASET
        or not _is_hex(value.get("dataset_identity_sha256"), length=64)
        or steps != QUALITY_BRIDGE_TARGET_STEPS
        or effective_batch != QUALITY_BRIDGE_EFFECTIVE_BATCH
        or images_seen != QUALITY_BRIDGE_IMAGES_SEEN
        or not math.isclose(
            full_epochs,
            QUALITY_BRIDGE_IMAGES_SEEN / train_image_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            ratio,
            full_epochs / reference_epochs,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or value.get("below_historical_reference_exposure") is not below_reference
        or value.get("insufficient_exposure_is_live_hypothesis") is not below_reference
        or value.get("causal_status") != "not_identified_by_exposure_alone"
        or value.get("quality_metric_comparison_allowed") is not False
        or value.get("terminal_result_content_bound") is not True
        or value.get("terminal_checkpoint_binding_verified") is not True
    ):
        raise ValueError("quality-bridge follow-up training exposure differs")
    return copy.deepcopy(dict(value))


def classify_followup_decision(report: Mapping[str, Any]) -> str:
    recommendation = report.get("recommended_next_stage")
    terminal = report.get("terminal_quality")
    sources = report.get("source_reports")
    checks = terminal.get("checks") if isinstance(terminal, Mapping) else None
    failed = terminal.get("failed_checks") if isinstance(terminal, Mapping) else None
    if (
        report.get("schema_version") != FOLLOWUP_DECISION_SCHEMA_VERSION
        or report.get("role") != FOLLOWUP_DECISION_ROLE
        or report.get("status") != "completed"
        or report.get("decision_builder_git") != FOLLOWUP_DECISION_BUILDER_GIT
        or report.get("quality_bridge_execution_git") != QUALITY_BRIDGE_GIT
        or report.get("authorization_boundary") != FOLLOWUP_AUTHORIZATION_BOUNDARY
        or not isinstance(recommendation, Mapping)
        or recommendation.get("execution_ready") is not False
        or recommendation.get("gpu_execution_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
        or recommendation.get("release_authorization_allowed") is not False
        or not isinstance(terminal, Mapping)
        or not isinstance(checks, list)
        or not isinstance(failed, list)
        or not isinstance(sources, Mapping)
        or set(sources)
        != {"quality_bridge_result", "milestones", "terminal_training_exposure"}
    ):
        raise ValueError("quality-bridge follow-up decision is malformed")
    indexed: dict[str, bool] = {}
    observed_failed: list[str] = []
    for row in checks:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("name"), str)
            or row["name"] in indexed
            or type(row.get("passed")) is not bool
        ):
            raise ValueError("quality-bridge follow-up checks are malformed")
        indexed[str(row["name"])] = bool(row["passed"])
        if row["passed"] is False:
            observed_failed.append(str(row["name"]))
    if set(indexed) != EXPECTED_CHECKS or observed_failed != failed:
        raise ValueError("quality-bridge follow-up check set differs")
    _identity(sources.get("quality_bridge_result", {}), label="quality bridge result")
    milestones = sources.get("milestones")
    if not isinstance(milestones, Mapping) or set(milestones) != {"50000", "100000"}:
        raise ValueError("quality-bridge follow-up milestone source set differs")
    for step, identity in milestones.items():
        _identity(identity, label=f"quality bridge milestone {step}")
    _validate_followup_training_exposure(
        report.get("training_exposure"),
        expected_source=sources.get("terminal_training_exposure", {}),
    )
    route = recommendation.get("id")
    if route != FOLLOWUP_DECISION_ID:
        return "not_selected"
    matched_failures = sorted(set(observed_failed) & MATCHED_QUALITY_CHECKS)
    trigger = recommendation.get("trigger")
    if (
        recommendation.get("category") != FOLLOWUP_DECISION_CATEGORY
        or not matched_failures
        or set(observed_failed) != set(matched_failures)
        or not isinstance(trigger, Mapping)
        or trigger.get("failed_checks") != matched_failures
    ):
        raise ValueError("matched factorization-quality route trigger differs")
    return "selected"


def validate_terminal_system_guard(
    report: Mapping[str, Any],
    *,
    expected_quality_result: Mapping[str, Any],
    expected_failed_checks: list[str],
) -> dict[str, Any]:
    sources = report.get("sources")
    evidence = report.get("evidence")
    policy = report.get("claim_policy")
    boundary = report.get("claim_boundary")
    quality_screen = evidence.get("quality_screen") if isinstance(evidence, Mapping) else None
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != TERMINAL_SYSTEM_GUARD_ROLE
        or report.get("status") not in {"pass", "hold"}
        or not isinstance(sources, Mapping)
        or _identity(
            sources.get("quality_bridge_result", {}),
            label="terminal-system quality result",
        )
        != _identity(expected_quality_result, label="expected quality result")
        or not isinstance(evidence, Mapping)
        or not isinstance(quality_screen, Mapping)
        or quality_screen.get("failed_checks") != expected_failed_checks
        or not isinstance(policy, Mapping)
        or policy.get("terminal_system_evidence_complete") is not True
        or policy.get("larger_training_launch_allowed") is not False
        or policy.get("release_authorization_allowed") is not False
        or policy.get("broad_generation_superiority_claim_allowed") is not False
        or not isinstance(boundary, Mapping)
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("gpu_execution_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("terminal-system guard does not bind completed non-authorizing evidence")
    return copy.deepcopy(dict(report))


def _validate_training_report(
    report: Mapping[str, Any],
    *,
    method: str,
    checkpoint_reference: Mapping[str, Any],
) -> dict[str, Any]:
    expected_mode = "fixed_basis" if method == "cofitok" else "dense_identity"
    expected_tokens = 8 if method == "cofitok" else 1
    config = report.get("config")
    latest = report.get("latest_checkpoint")
    git = report.get("git")
    if (
        report.get("training_complete") is not True
        or int(report.get("completed_steps", -1)) != CHECKPOINT_STEP
        or int(report.get("target_steps", -1)) != CHECKPOINT_STEP
        or not isinstance(config, Mapping)
        or config.get("data", {}).get("dataset") != PROTOCOL["dataset"]
        or int(config.get("runtime", {}).get("steps", -1)) != CHECKPOINT_STEP
        or config.get("model", {}).get("synthesis_mode") != expected_mode
        or int(config.get("model", {}).get("token_count", -1)) != expected_tokens
        or not isinstance(git, Mapping)
        or git.get("revision") != QUALITY_BRIDGE_GIT["revision"]
        or git.get("branch") != QUALITY_BRIDGE_GIT["branch"]
        or git.get("dirty") is not False
        or not isinstance(latest, Mapping)
        or int(latest.get("step", -1)) != CHECKPOINT_STEP
        or latest.get("checkpoint") != "checkpoint_step_00100000.pt"
        or latest.get("integrity_manifest")
        != "checkpoint_step_00100000.pt.integrity.json"
        or latest.get("checkpoint_sha256") != checkpoint_reference.get("checkpoint_sha256")
        or int(latest.get("checkpoint_bytes", -1))
        != int(checkpoint_reference.get("checkpoint_bytes", -2))
    ):
        raise ValueError(f"{method} 100K training report contract differs")
    return copy.deepcopy(dict(report))


def _validate_checkpoint_evaluation(
    report: Mapping[str, Any],
    *,
    method: str,
    training: Mapping[str, Any],
    checkpoint_reference: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = report.get("metrics")
    if (
        report.get("schema_version") != CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION
        or report.get("role") != "generation_checkpoint_evaluation_report"
        or report.get("status") != "completed"
        or report.get("weights") != PROTOCOL["weights"]
        or int(report.get("checkpoint_step", -1)) != CHECKPOINT_STEP
        or report.get("checkpoint_sha256") != checkpoint_reference.get("checkpoint_sha256")
        or report.get("config") != training.get("config")
        or report.get("git") != QUALITY_BRIDGE_GIT
        or not isinstance(metrics, Mapping)
        or int(metrics.get("evaluated_images", -1)) != 256
        or int(metrics.get("timestep", -1)) != 500
    ):
        raise ValueError(f"{method} terminal checkpoint evaluation differs")
    return copy.deepcopy(dict(report))


def build_source_binding(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    followup_decision: Mapping[str, Any],
    followup_decision_identity: Mapping[str, Any],
    terminal_system_guard: Mapping[str, Any],
    terminal_system_guard_identity: Mapping[str, Any],
    quality_result: Mapping[str, Any],
    quality_result_identity: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    checkpoint_evaluations: Mapping[str, Mapping[str, Any]],
    checkpoint_references: Mapping[str, Mapping[str, Any]],
    source_identities: Mapping[str, Mapping[str, Any]],
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    normalized_preparation = validate_preparation(
        preparation,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    if classify_followup_decision(followup_decision) != "selected":
        raise ValueError("factorization quality-regression route is not selected")
    decision_result = _identity(
        followup_decision["source_reports"]["quality_bridge_result"],
        label="follow-up quality result",
    )
    decision_exposure = _identity(
        followup_decision["source_reports"]["terminal_training_exposure"],
        label="follow-up terminal training exposure",
    )
    result_identity = _identity(quality_result_identity, label="quality result")
    if decision_result != result_identity:
        raise ValueError("follow-up decision and physical quality result differ")
    failed_checks = list(followup_decision["terminal_quality"]["failed_checks"])
    validate_terminal_system_guard(
        terminal_system_guard,
        expected_quality_result=result_identity,
        expected_failed_checks=failed_checks,
    )
    result_sources = quality_result.get("source_reports")
    if (
        quality_result.get("schema_version") != 1
        or quality_result.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or quality_result.get("status") != "completed"
        or not isinstance(result_sources, Mapping)
    ):
        raise ValueError("quality-bridge result contract differs")
    expected_source_keys = {
        "quality_result",
        "followup_decision",
        "terminal_training_exposure",
        "terminal_system_guard",
        "cofitok_training",
        "dense_training",
        "cofitok_checkpoint_evaluation",
        "dense_checkpoint_evaluation",
        "cofitok_checkpoint_integrity",
        "dense_checkpoint_integrity",
    }
    if set(source_identities) != expected_source_keys:
        raise ValueError("factorization-regression source identity set differs")
    normalized_sources = {
        name: _identity(identity, label=name)
        for name, identity in source_identities.items()
    }
    fixed_sources = {
        "quality_result": result_identity,
        "followup_decision": _identity(
            followup_decision_identity,
            label="follow-up decision",
        ),
        "terminal_training_exposure": decision_exposure,
        "terminal_system_guard": _identity(
            terminal_system_guard_identity,
            label="terminal-system guard",
        ),
    }
    for name, identity in fixed_sources.items():
        if normalized_sources[name] != identity:
            raise ValueError(f"{name} source identity differs")
    methods: dict[str, Any] = {}
    for method, result_training_key, result_eval_key in (
        ("cofitok", "cofitok_training", "cofitok_checkpoint_eval"),
        ("dense_identity", "dense_training", "dense_checkpoint_eval"),
    ):
        training = training_reports.get(method)
        evaluation = checkpoint_evaluations.get(method)
        reference = checkpoint_references.get(method)
        if not all(isinstance(value, Mapping) for value in (training, evaluation, reference)):
            raise ValueError(f"{method} source reports are missing")
        source_training_name = f"{method.split('_')[0]}_training"
        source_eval_name = f"{method.split('_')[0]}_checkpoint_evaluation"
        source_integrity_name = f"{method.split('_')[0]}_checkpoint_integrity"
        if normalized_sources[source_training_name] != _identity(
            result_sources.get(result_training_key, {}),
            label=result_training_key,
        ):
            raise ValueError(f"{method} training identity differs from quality result")
        if normalized_sources[source_eval_name] != _identity(
            result_sources.get(result_eval_key, {}),
            label=result_eval_key,
        ):
            raise ValueError(f"{method} checkpoint evaluation differs from quality result")
        normalized_reference = {
            "checkpoint": _absolute_posix(
                str(reference.get("checkpoint", "")),
                label=f"{method} checkpoint",
            ),
            "checkpoint_bytes": int(reference.get("checkpoint_bytes", -1)),
            "checkpoint_sha256": str(reference.get("checkpoint_sha256", "")),
            "checkpoint_step": int(reference.get("checkpoint_step", -1)),
            "integrity_manifest": _identity(
                reference.get("integrity_manifest", {}),
                label=f"{method} checkpoint integrity",
            ),
        }
        if (
            normalized_reference["checkpoint"]
            != f'{RUN_DIRS[method]}/checkpoint_step_00100000.pt'
            or normalized_reference["checkpoint_bytes"] < 1
            or not _is_hex(normalized_reference["checkpoint_sha256"], length=64)
            or normalized_reference["checkpoint_step"] != CHECKPOINT_STEP
            or normalized_reference["integrity_manifest"]
            != normalized_sources[source_integrity_name]
        ):
            raise ValueError(f"{method} checkpoint reference differs")
        normalized_training = _validate_training_report(
            training,
            method=method,
            checkpoint_reference=normalized_reference,
        )
        _validate_checkpoint_evaluation(
            evaluation,
            method=method,
            training=normalized_training,
            checkpoint_reference=normalized_reference,
        )
        methods[method] = {
            "training_report": normalized_sources[source_training_name],
            "terminal_checkpoint_evaluation": normalized_sources[source_eval_name],
            "checkpoint": normalized_reference,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "role": SOURCE_BINDING_ROLE,
        "status": "verified",
        "scope": SCOPE,
        "preparation": _identity(preparation_identity, label="preparation"),
        "sources": normalized_sources,
        "selected_route": {
            "id": FOLLOWUP_DECISION_ID,
            "category": FOLLOWUP_DECISION_CATEGORY,
            "failed_checks": sorted(set(failed_checks) & MATCHED_QUALITY_CHECKS),
        },
        "methods": methods,
        "protocol": copy.deepcopy(PROTOCOL),
        "authorization_boundary": copy.deepcopy(SOURCE_BINDING_BOUNDARY),
    }


def build_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    source_binding_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    execution_git: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    validate_preparation(
        preparation,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    if (
        source_binding.get("schema_version") != SCHEMA_VERSION
        or source_binding.get("role") != SOURCE_BINDING_ROLE
        or source_binding.get("status") != "verified"
        or source_binding.get("scope") != SCOPE
        or source_binding.get("preparation")
        != _identity(preparation_identity, label="preparation")
        or source_binding.get("protocol") != PROTOCOL
        or source_binding.get("authorization_boundary") != SOURCE_BINDING_BOUNDARY
    ):
        raise ValueError("factorization-regression source binding differs")
    standing = validate_standing_experiment_authorization(standing_authorization)
    if standing.get("preserved_safety_boundaries") != STANDING_AUTHORIZATION_SAFETY_BOUNDARIES:
        raise ValueError("standing authorization safety boundary differs")
    git = _git(execution_git, label="factorization-regression authorization")
    if git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("factorization-regression authorization Git differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "role": EXECUTION_AUTHORIZATION_ROLE,
        "status": "authorized",
        "scope": SCOPE,
        "git": git,
        "output_root": OUTPUT_ROOT,
        "preparation": _identity(preparation_identity, label="preparation"),
        "source_binding": _identity(source_binding_identity, label="source binding"),
        "standing_authorization": {
            "source": _identity(
                standing_authorization_identity,
                label="standing authorization",
            ),
            "validated_record": standing,
        },
        "protocol": copy.deepcopy(PROTOCOL),
        "methods": copy.deepcopy(dict(source_binding["methods"])),
        "authorization_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
    }


def validate_execution_authorization(
    report: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    expected = build_execution_authorization(**kwargs)
    if dict(report) != expected:
        raise ValueError("factorization-regression execution authorization differs")
    return copy.deepcopy(expected)


def _teacher_comparison(
    cofitok: Mapping[str, Any],
    dense: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cofitok_rows = cofitok.get("teacher_forced")
    dense_rows = dense.get("teacher_forced")
    if not isinstance(cofitok_rows, list) or not isinstance(dense_rows, list):
        raise ValueError("teacher-forced rollout rows are missing")
    cofitok_by_timestep = {int(row["timestep"]): row for row in cofitok_rows}
    dense_by_timestep = {int(row["timestep"]): row for row in dense_rows}
    if set(cofitok_by_timestep) != set(PROTOCOL["teacher_timesteps"]) or set(
        dense_by_timestep
    ) != set(PROTOCOL["teacher_timesteps"]):
        raise ValueError("teacher-forced timestep set differs")
    rows = []
    for timestep in PROTOCOL["teacher_timesteps"]:
        cofitok_row = cofitok_by_timestep[timestep]
        dense_row = dense_by_timestep[timestep]
        cofitok_epsilon = _finite(
            cofitok_row.get("epsilon_mse"),
            label=f"CoFiTok teacher epsilon at {timestep}",
            positive=True,
        )
        dense_epsilon = _finite(
            dense_row.get("epsilon_mse"),
            label=f"dense teacher epsilon at {timestep}",
            positive=True,
        )
        cofitok_x0 = _finite(
            cofitok_row.get("clipped_x0_mse"),
            label=f"CoFiTok teacher x0 at {timestep}",
            positive=True,
        )
        dense_x0 = _finite(
            dense_row.get("clipped_x0_mse"),
            label=f"dense teacher x0 at {timestep}",
            positive=True,
        )
        rows.append(
            {
                "timestep": timestep,
                "cofitok_epsilon_mse": cofitok_epsilon,
                "dense_epsilon_mse": dense_epsilon,
                "epsilon_mse_ratio": cofitok_epsilon / dense_epsilon,
                "cofitok_clipped_x0_mse": cofitok_x0,
                "dense_clipped_x0_mse": dense_x0,
                "clipped_x0_mse_ratio": cofitok_x0 / dense_x0,
            }
        )
    return rows


def build_diagnostic_report(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    source_binding_identity: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    checkpoint_evaluations: Mapping[str, Mapping[str, Any]],
    rollout_reports: Mapping[int, Mapping[str, Mapping[str, Any]]],
    qualification_reports: Mapping[int, Mapping[str, Any]],
    source_identities: Mapping[str, Mapping[str, Any]],
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    validate_preparation(
        preparation,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    if source_binding.get("role") != SOURCE_BINDING_ROLE:
        raise ValueError("diagnostic source binding differs")
    if execution_authorization.get("role") != EXECUTION_AUTHORIZATION_ROLE:
        raise ValueError("diagnostic execution authorization differs")
    if execution_authorization.get("source_binding") != _identity(
        source_binding_identity,
        label="source binding",
    ):
        raise ValueError("execution authorization does not bind the source report")
    if set(rollout_reports) != set(SEEDS) or set(qualification_reports) != set(SEEDS):
        raise ValueError("diagnostic seed set differs")
    evaluation_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    seed_results: dict[str, Any] = {}
    all_failed: list[set[str]] = []
    for seed in SEEDS:
        reports = rollout_reports[seed]
        if set(reports) != {"cofitok", "dense_identity"}:
            raise ValueError(f"rollout method set differs for seed {seed}")
        expected_protocol = {
            "num_images": PROTOCOL["num_images_per_seed"],
            "batch_size": PROTOCOL["batch_size"],
            "teacher_timesteps": PROTOCOL["teacher_timesteps"],
            "sample_steps": PROTOCOL["sample_steps"],
            "guidance_scale": PROTOCOL["guidance_scale"],
            "teacher_guidance_scale": PROTOCOL["teacher_guidance_scale"],
            "guidance_rescale": PROTOCOL["guidance_rescale"],
            "cfg_batch_mode": PROTOCOL["cfg_batch_mode"],
            "clip_x0": PROTOCOL["clip_x0"],
            "precision": PROTOCOL["precision"],
            "seed": seed,
        }
        for method, rollout in reports.items():
            checkpoint = source_binding["methods"][method]["checkpoint"]
            runtime = rollout.get("runtime")
            if (
                rollout.get("schema_version") != 1
                or rollout.get("status") != "completed"
                or rollout.get("git") != evaluation_git
                or rollout.get("weights") != PROTOCOL["weights"]
                or int(rollout.get("checkpoint_step", -1)) != CHECKPOINT_STEP
                or rollout.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
                or rollout.get("checkpoint") != checkpoint["checkpoint"]
                or rollout.get("config") != training_reports[method].get("config")
                or rollout.get("protocol") != expected_protocol
                or not isinstance(runtime, Mapping)
                or _finite(
                    runtime.get("elapsed_seconds"),
                    label=f"{method} seed {seed} elapsed time",
                    positive=True,
                )
                <= 0.0
                or int(runtime.get("cuda_peak_memory_bytes", -1)) < 1
            ):
                raise ValueError(f"{method} rollout report differs for seed {seed}")
        source_names = {
            "cofitok_training": "cofitok_training",
            "dense_training": "dense_training",
            "cofitok_checkpoint": "cofitok_diagnostic_checkpoint_evaluation",
            "dense_checkpoint": "dense_diagnostic_checkpoint_evaluation",
            "cofitok_rollout": f"cofitok_rollout_seed_{seed}",
            "dense_rollout": f"dense_rollout_seed_{seed}",
        }
        qualification_sources = {
            name: _identity(source_identities[source_name], label=source_name)
            for name, source_name in source_names.items()
        }
        expected_qualification = build_stability_qualification(
            cofitok_training=dict(training_reports["cofitok"]),
            dense_training=dict(training_reports["dense_identity"]),
            cofitok_checkpoint=dict(checkpoint_evaluations["cofitok"]),
            dense_checkpoint=dict(checkpoint_evaluations["dense_identity"]),
            cofitok_rollout=dict(reports["cofitok"]),
            dense_rollout=dict(reports["dense_identity"]),
            expected_weights=PROTOCOL["weights"],
            expected_evaluation_revision=expected_revision,
            expected_evaluation_branch=expected_branch,
            high_frequency_timesteps=DEFAULT_HIGH_FREQUENCY_TIMESTEPS,
        )
        expected_qualification["sources"] = qualification_sources
        qualification = qualification_reports[seed]
        if dict(qualification) != expected_qualification:
            raise ValueError(f"seed {seed} stability qualification is not reproducible")
        failed_gates = sorted(
            name
            for name, gate in qualification["gates"].items()
            if gate["passed"] is False
        )
        all_failed.append(set(failed_gates))
        seed_results[str(seed)] = {
            "status": qualification["status"],
            "failed_gates": failed_gates,
            "teacher_forced": _teacher_comparison(
                reports["cofitok"],
                reports["dense_identity"],
            ),
            "qualification_metrics": copy.deepcopy(qualification["metrics"]),
            "runtime": {
                method: copy.deepcopy(dict(reports[method]["runtime"]))
                for method in ("cofitok", "dense_identity")
            },
            "sources": {
                "cofitok_rollout": qualification_sources["cofitok_rollout"],
                "dense_rollout": qualification_sources["dense_rollout"],
                "qualification": _identity(
                    source_identities[f"qualification_seed_{seed}"],
                    label=f"qualification seed {seed}",
                ),
            },
        }
    union = sorted(set().union(*all_failed))
    common = sorted(set.intersection(*all_failed)) if all_failed else []
    if not union:
        decision_id = "factorization_rollout_regression_not_reproduced"
        next_evidence = (
            "Do not change training from this diagnostic alone. Reconcile terminal feature/"
            "support evidence and prepare a separately authorized bounded loss-or-layout "
            "ablation only if the source-bound decision still requires it."
        )
    elif common:
        decision_id = "consistent_factorization_rollout_regression_detected"
        next_evidence = (
            "Use the repeated failed gates to predeclare the smallest matched loss-or-layout "
            "ablation. A new source-compatible decision is required before any training."
        )
    else:
        decision_id = "seed_sensitive_factorization_rollout_regression_detected"
        next_evidence = (
            "Increase diagnostic seed coverage before designing a training intervention; "
            "no single-seed gate may authorize a recipe change."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": DIAGNOSTIC_REPORT_ROLE,
        "status": "completed",
        "scope": SCOPE,
        "protocol": copy.deepcopy(PROTOCOL),
        "git": evaluation_git,
        "decision": {
            "id": decision_id,
            "failed_gate_union": union,
            "failed_gate_intersection": common,
            "causal_factorization_attribution": False,
            "recommended_next_evidence": next_evidence,
            "next_stage_execution_allowed": False,
        },
        "seeds": seed_results,
        "sources": {
            "preparation": _identity(preparation_identity, label="preparation"),
            "source_binding": _identity(source_binding_identity, label="source binding"),
            "execution_authorization": _identity(
                execution_authorization_identity,
                label="execution authorization",
            ),
            **{
                name: _identity(identity, label=name)
                for name, identity in sorted(source_identities.items())
            },
        },
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
    }
