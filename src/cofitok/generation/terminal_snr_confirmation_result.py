"""Fail-closed result for the terminal-SNR frozen 10K confirmation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from cofitok.generation import terminal_snr_screen_result as screen_result
from cofitok.generation.exposure_capacity_authorization import identity, read_object
from cofitok.generation.exposure_capacity_decision import THRESHOLDS as SUPPORT_THRESHOLDS
from cofitok.generation.terminal_snr_confirmation import (
    EVALUATION_CONTRACT,
    SAMPLE_COUNT,
    validate_terminal_snr_confirmation_preparation_contract,
)
from cofitok.generation.terminal_snr_confirmation_arm import (
    validate_terminal_snr_confirmation_arm_validation,
)
from cofitok.generation.terminal_snr_confirmation_execution import (
    validate_terminal_snr_confirmation_launch_receipt_contract,
    validate_terminal_snr_confirmation_launch_receipt_physical,
)
from cofitok.generation.terminal_snr_screen import ARM_NAMES, SCREEN_THRESHOLDS


RESULT_SCHEMA = "cofitok_generation_terminal_snr_confirmation_result_v1"
RESULT_ROLE = "source_bound_terminal_snr_frozen_confirmation_qualification"
VALIDATION_SCHEMA = (
    "cofitok_generation_terminal_snr_confirmation_result_validation_v1"
)
VALIDATION_ROLE = "content_addressed_terminal_snr_confirmation_result_validation"
RESULT_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "large_capacity_readiness_preparation_allowed": "derived_from_result",
    "large_capacity_readiness_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
VALIDATION_BOUNDARY = {
    "receipt_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "large_capacity_readiness_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
CONFIRMATION_THRESHOLDS = {
    **SCREEN_THRESHOLDS,
    "both_methods_max_absolute_fid": SUPPORT_THRESHOLDS["max_absolute_fid"],
    "both_methods_min_absolute_precision": SUPPORT_THRESHOLDS["min_precision"],
    "both_methods_min_absolute_recall": SUPPORT_THRESHOLDS["min_recall"],
    "both_methods_min_absolute_top1": SUPPORT_THRESHOLDS["min_top1"],
    "both_methods_min_absolute_top5": SUPPORT_THRESHOLDS["min_top5"],
    "both_methods_min_predicted_class_fraction": SUPPORT_THRESHOLDS[
        "min_predicted_class_fraction"
    ],
    "both_methods_min_normalized_predicted_entropy": SUPPORT_THRESHOLDS[
        "min_normalized_predicted_entropy"
    ],
    "cofitok_max_relative_fid_regression_vs_dense": SUPPORT_THRESHOLDS[
        "max_fid_regression"
    ],
    "cofitok_max_precision_regression_vs_dense": SUPPORT_THRESHOLDS[
        "max_precision_regression"
    ],
    "cofitok_max_recall_regression_vs_dense": SUPPORT_THRESHOLDS[
        "max_recall_regression"
    ],
    "cofitok_max_top1_regression_vs_dense": SUPPORT_THRESHOLDS[
        "max_top1_regression"
    ],
    "cofitok_max_top5_regression_vs_dense": SUPPORT_THRESHOLDS[
        "max_top5_regression"
    ],
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    if (
        not isinstance(row.get("path"), str)
        or not row["path"]
        or not isinstance(row.get("bytes"), int)
        or isinstance(row.get("bytes"), bool)
        or row["bytes"] < 1
        or not _hex(row.get("sha256"), 64)
    ):
        raise ValueError(f"{name} identity is malformed")
    return {key: row[key] for key in ("path", "bytes", "sha256")}


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    if (
        not _hex(row.get("revision"), 40)
        or not _hex(row.get("tree"), 40)
        or not isinstance(row.get("branch"), str)
        or not row["branch"]
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} must identify one exact clean checkout")
    return copy.deepcopy(row)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite range")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _finite(value, name, minimum=0.0)
    if result > 1.0:
        raise ValueError(f"{name} exceeds one")
    return result


def _check(
    name: str, *, actual: float | int, operator: str, threshold: float | int
) -> dict[str, Any]:
    if operator == ">=":
        passed = actual >= threshold
    elif operator == "<=":
        passed = actual <= threshold
    elif operator == "==":
        passed = actual == threshold
    else:  # pragma: no cover
        raise ValueError(f"unknown terminal-SNR confirmation operator: {operator}")
    return {
        "name": name,
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "pass": bool(passed),
    }


def _summary(report: Mapping[str, Any], arm: str) -> dict[str, Any]:
    row = _object(report, f"{arm} confirmation validation")
    distribution = _object(row.get("distribution"), f"{arm} distribution")
    metrics = _object(distribution.get("metrics"), f"{arm} distribution metrics")
    class_fidelity = _object(row.get("class_fidelity"), f"{arm} class fidelity")
    class_metrics = _object(
        class_fidelity.get("metrics"), f"{arm} class fidelity metrics"
    )
    frozen = _object(row.get("frozen_training"), f"{arm} frozen training")
    screen_training = _object(frozen.get("screen_training"), f"{arm} screen training")
    screen_training_validation = _object(
        screen_training.get("validation"), f"{arm} screen training validation"
    )
    checkpoint_evaluation = _object(
        row.get("screen_checkpoint_evaluation"), f"{arm} checkpoint evaluation"
    )
    diagnostic = _object(
        checkpoint_evaluation.get("summary"), f"{arm} checkpoint diagnostics"
    )
    screen_rollout = _object(row.get("screen_rollout"), f"{arm} screen rollout")
    clipping = _object(
        screen_rollout.get("terminal_raw_x0_clipping"), f"{arm} terminal clipping"
    )
    sampling = _object(row.get("sampling"), f"{arm} sampling")
    preflight = _object(row.get("sampling_preflight"), f"{arm} sampling preflight")
    preflight_report = _identity(
        preflight.get("report"), f"{arm} sampling preflight report"
    )
    if (
        preflight.get("status") != "passed"
        or preflight.get("output_finite") is not True
        or int(preflight.get("checkpoint_step", -1))
        != int(EVALUATION_CONTRACT["checkpoint_step"])
        or int(preflight.get("peak_allocated_bytes", 0)) < 1
        or int(preflight.get("device_total_memory_bytes", 0)) < 1
    ):
        raise ValueError(f"{arm} sampling preflight evidence differs")
    result: dict[str, Any] = {
        "arm": arm,
        "condition": row.get("condition"),
        "method": row.get("method"),
        "endpoint_fraction": _finite(
            row.get("endpoint_fraction"), f"{arm} endpoint fraction", minimum=0.0
        ),
        "checkpoint": copy.deepcopy(frozen.get("checkpoint")),
        "validation_epsilon_mse": _finite(
            screen_training.get("validation_epsilon_mse"),
            f"{arm} validation epsilon MSE",
            minimum=0.0,
        ),
        "sample_count": int(sampling.get("sample_count", -1)),
        "sample_set_sha256": sampling.get("sample_set_sha256"),
        "screen_sample_set_sha256": row.get("screen_sample_set_sha256"),
        "sampling_preflight": copy.deepcopy(preflight),
        "sampling_preflight_report": preflight_report,
        "real_set": copy.deepcopy(
            _object(distribution.get("real_set"), f"{arm} real set")
        ),
        "classifier": copy.deepcopy(
            _object(class_fidelity.get("classifier"), f"{arm} classifier")
        ),
        "fid": _finite(metrics.get("fid"), f"{arm} FID", minimum=0.0),
        "inception_score_mean": _finite(
            metrics.get("inception_score_mean"), f"{arm} IS", minimum=0.0
        ),
        "precision": _fraction(metrics.get("precision"), f"{arm} precision"),
        "recall": _fraction(metrics.get("recall"), f"{arm} recall"),
        "class_top1": _fraction(
            class_metrics.get("top1_accuracy"), f"{arm} top1"
        ),
        "class_top5": _fraction(
            class_metrics.get("top5_accuracy"), f"{arm} top5"
        ),
        "predicted_class_fraction": _fraction(
            class_metrics.get("predicted_class_fraction"),
            f"{arm} predicted class fraction",
        ),
        "normalized_predicted_class_entropy": _fraction(
            class_metrics.get("normalized_predicted_class_entropy"),
            f"{arm} predicted class entropy",
        ),
        "terminal_raw_x0_clip_fraction": _fraction(
            clipping.get("raw_x0_clip_fraction"), f"{arm} terminal clipping"
        ),
        "training_runtime_environment_sha256": screen_training_validation.get(
            "runtime_environment_sha256"
        ),
        "sampling_runtime_environment_sha256": sampling.get(
            "runtime_environment_sha256"
        ),
        "distribution_runtime_environment_sha256": distribution.get(
            "runtime_environment_sha256"
        ),
        "class_fidelity_runtime_environment_sha256": class_fidelity.get(
            "runtime_environment_sha256"
        ),
        "execution_git": copy.deepcopy(row.get("execution_git")),
        "screen_execution_git": copy.deepcopy(row.get("screen_execution_git")),
    }
    if (
        result["sample_count"] != SAMPLE_COUNT
        or not _hex(result["sample_set_sha256"], 64)
        or not _hex(result["screen_sample_set_sha256"], 64)
        or result["sample_set_sha256"] == result["screen_sample_set_sha256"]
    ):
        raise ValueError(f"{arm} confirmation sample evidence differs")
    if row.get("method") == "cofitok":
        utilization = _object(diagnostic.get("utilization"), f"{arm} utilization")
        result["cofitok_diagnostics"] = {
            "ordered_rank_by_path_auc": int(
                diagnostic.get("ordered_rank_by_path_auc", -1)
            ),
            "zero_token_max_abs": _finite(
                diagnostic.get("zero_token_max_abs"),
                f"{arm} zero-token maximum",
                minimum=0.0,
            ),
            "shuffled_to_ordered_endpoint_ratio": _finite(
                diagnostic.get("shuffled_to_ordered_endpoint_ratio"),
                f"{arm} shuffle ratio",
                minimum=0.0,
            ),
            "coarse_token_energy_ratio": _finite(
                utilization.get("coarse_token_energy_ratio"),
                f"{arm} coarse energy",
                minimum=0.0,
            ),
            "tail_two_energy_ratio": _finite(
                utilization.get("tail_two_energy_ratio"),
                f"{arm} tail energy",
                minimum=0.0,
            ),
        }
    return result


def _absolute_and_matched_checks(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    endpoint_values: dict[str, dict[str, float]] = {}
    for method in ("cofitok", "dense_identity"):
        summary = _object(summaries[f"endpoint0975_{method}"], f"endpoint {method}")
        values = {
            "fid": float(summary["fid"]),
            "precision": float(summary["precision"]),
            "recall": float(summary["recall"]),
            "top1": float(summary["class_top1"]),
            "top5": float(summary["class_top5"]),
            "predicted_class_fraction": float(summary["predicted_class_fraction"]),
            "normalized_predicted_class_entropy": float(
                summary["normalized_predicted_class_entropy"]
            ),
        }
        endpoint_values[method] = values
        for metric, operator, threshold_name in (
            ("fid", "<=", "both_methods_max_absolute_fid"),
            ("precision", ">=", "both_methods_min_absolute_precision"),
            ("recall", ">=", "both_methods_min_absolute_recall"),
            ("top1", ">=", "both_methods_min_absolute_top1"),
            ("top5", ">=", "both_methods_min_absolute_top5"),
            (
                "predicted_class_fraction",
                ">=",
                "both_methods_min_predicted_class_fraction",
            ),
            (
                "normalized_predicted_class_entropy",
                ">=",
                "both_methods_min_normalized_predicted_entropy",
            ),
        ):
            checks.append(
                _check(
                    f"endpoint0975_{method}.{metric}_absolute",
                    actual=values[metric],
                    operator=operator,
                    threshold=CONFIRMATION_THRESHOLDS[threshold_name],
                )
            )
    cofitok = endpoint_values["cofitok"]
    dense = endpoint_values["dense_identity"]
    if dense["fid"] <= 0.0:
        raise ValueError("endpoint dense FID must be positive")
    matched = {
        "relative_fid_regression": cofitok["fid"] / dense["fid"] - 1.0,
        "precision_regression": dense["precision"] - cofitok["precision"],
        "recall_regression": dense["recall"] - cofitok["recall"],
        "top1_regression": dense["top1"] - cofitok["top1"],
        "top5_regression": dense["top5"] - cofitok["top5"],
    }
    for metric, threshold_name in (
        ("relative_fid_regression", "cofitok_max_relative_fid_regression_vs_dense"),
        ("precision_regression", "cofitok_max_precision_regression_vs_dense"),
        ("recall_regression", "cofitok_max_recall_regression_vs_dense"),
        ("top1_regression", "cofitok_max_top1_regression_vs_dense"),
        ("top5_regression", "cofitok_max_top5_regression_vs_dense"),
    ):
        checks.append(
            _check(
                f"endpoint0975_cofitok_vs_dense.{metric}",
                actual=matched[metric],
                operator="<=",
                threshold=CONFIRMATION_THRESHOLDS[threshold_name],
            )
        )
    return {
        "endpoint0975": endpoint_values,
        "endpoint0975_cofitok_vs_dense": matched,
    }, checks


def _evaluate(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    directional = screen_result._evaluate(summaries)
    support, support_checks = _absolute_and_matched_checks(summaries)
    checks = copy.deepcopy(directional["checks"]) + support_checks
    names = [str(row["name"]) for row in checks]
    if len(names) != len(set(names)):
        raise ValueError("terminal-SNR confirmation check names are duplicated")
    failed = [str(row["name"]) for row in checks if row.get("pass") is not True]
    return {
        "directional_comparisons": copy.deepcopy(directional["comparisons"]),
        "support_comparisons": support,
        "checks": checks,
        "failed_checks": failed,
        "confirmation_pass": not failed,
    }


def _cross_arm(
    summaries: Mapping[str, Mapping[str, Any]], runtime_environment_sha256: str
) -> dict[str, Any]:
    result = screen_result._cross_arm_evidence(
        summaries, runtime_environment_sha256
    )
    screen_execution_gits = {
        arm: _git(summaries[arm].get("screen_execution_git"), f"{arm} screen Git")
        for arm in ARM_NAMES
    }
    reference_screen_git = screen_execution_gits[ARM_NAMES[0]]
    if any(
        screen_execution_gits[arm] != reference_screen_git
        for arm in ARM_NAMES[1:]
    ):
        raise ValueError("terminal-SNR confirmation screen Git differs across arms")
    screen_sample_shas = {
        arm: summaries[arm].get("screen_sample_set_sha256") for arm in ARM_NAMES
    }
    if (
        len(set(screen_sample_shas.values())) != len(ARM_NAMES)
        or any(not _hex(value, 64) for value in screen_sample_shas.values())
        or set(screen_sample_shas.values())
        & set(result["sample_set_sha256_by_arm"].values())
    ):
        raise ValueError("terminal-SNR confirmation sample streams are not disjoint")
    result["screen_sample_set_sha256_by_arm"] = screen_sample_shas
    result["screen_execution_git"] = reference_screen_git
    result["confirmation_samples_disjoint_from_screen"] = True
    preflight_reports: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        summary = _object(summaries[arm], f"{arm} summary")
        preflight = _object(
            summary.get("sampling_preflight"), f"{arm} sampling preflight"
        )
        checkpoint = _object(summary.get("checkpoint"), f"{arm} checkpoint")
        execution = _git(summary.get("execution_git"), f"{arm} execution Git")
        screen_execution = _git(
            summary.get("screen_execution_git"), f"{arm} screen execution Git"
        )
        expected_execution_report_git = {
            "revision": execution["revision"],
            "branch": execution["branch"],
            "tracked_dirty": False,
        }
        expected_screen_report_git = {
            "revision": screen_execution["revision"],
            "branch": screen_execution["branch"],
            "tracked_dirty": False,
        }
        if (
            preflight.get("checkpoint_sha256") != checkpoint.get("sha256")
            or preflight.get("runtime_environment_sha256")
            != runtime_environment_sha256
            or preflight.get("execution_git") != expected_execution_report_git
            or preflight.get("checkpoint_screen_git")
            != expected_screen_report_git
            or preflight.get("source_git") is not None
            or preflight.get("output_finite") is not True
        ):
            raise ValueError(f"{arm} sampling preflight binding differs")
        preflight_reports[arm] = _identity(
            summary.get("sampling_preflight_report"),
            f"{arm} sampling preflight report",
        )
    if len({row["sha256"] for row in preflight_reports.values()}) != len(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation preflight reports are duplicated")
    result["sampling_preflight_reports"] = preflight_reports
    result["all_real_forward_preflights_passed"] = True
    return result


def build_terminal_snr_confirmation_result(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    arm_validations: Mapping[str, Mapping[str, Any]],
    arm_validation_identities: Mapping[str, Mapping[str, Any]],
    result_git: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_terminal_snr_confirmation_preparation_contract(preparation)
    prep_id = _identity(
        preparation_identity, "terminal-SNR confirmation preparation"
    )
    launch_id = _identity(
        launch_receipt_identity, "terminal-SNR confirmation launch receipt"
    )
    execution = _git(
        launch_receipt.get("execution_checkout"),
        "terminal-SNR confirmation execution checkout",
    )
    launch = validate_terminal_snr_confirmation_launch_receipt_contract(
        launch_receipt, expected_execution_checkout=execution
    )
    result_checkout = _git(result_git, "terminal-SNR confirmation result checkout")
    if result_checkout != execution:
        raise ValueError("terminal-SNR confirmation result Git differs from launch")
    launch_sources = _object(
        launch.get("source_evidence"), "terminal-SNR confirmation launch sources"
    )
    if (
        launch_sources.get("preparation") != prep_id
        or launch.get("screen_execution_checkout")
        != prepared.get("screen_execution_checkout")
        or launch.get("frozen_arms") != prepared.get("frozen_arms")
        or launch.get("evaluation_contract") != prepared.get("evaluation_contract")
        or launch.get("thresholds") != prepared.get("thresholds")
    ):
        raise ValueError("terminal-SNR confirmation launch binds another preparation")
    if set(arm_validations) != set(ARM_NAMES) or set(
        arm_validation_identities
    ) != set(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation result arm set differs")
    summaries: dict[str, dict[str, Any]] = {}
    normalized_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        validation = validate_terminal_snr_confirmation_arm_validation(
            arm_validations[arm]
        )
        arm_id = _identity(
            arm_validation_identities[arm], f"{arm} confirmation validation"
        )
        if validation.get("execution_git") != execution:
            raise ValueError(f"{arm} confirmation uses another checkout")
        sources = _object(validation.get("sources"), f"{arm} confirmation sources")
        if sources.get("launch_receipt") != launch_id:
            raise ValueError(f"{arm} confirmation binds another launch receipt")
        summaries[arm] = _summary(validation, arm)
        normalized_ids[arm] = arm_id
    runtime_environment_sha256 = str(launch["runtime_environment_sha256"])
    cross_arm = _cross_arm(summaries, runtime_environment_sha256)
    if cross_arm["screen_execution_git"] != prepared["screen_execution_checkout"]:
        raise ValueError("terminal-SNR confirmation screen Git differs from preparation")
    evaluated = _evaluate(summaries)
    passed = bool(evaluated["confirmation_pass"])
    report = {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "confirmation_pass" if passed else "hold",
        "terminal_status": "hold",
        "confirmation_pass": passed,
        "support_collapse_resolved": passed,
        "generation_advantage_proven": False,
        "decision": (
            "prepare_separate_large_capacity_readiness"
            if passed
            else "hold_terminal_snr_intervention"
        ),
        "result_git": result_checkout,
        "runtime_environment_sha256": runtime_environment_sha256,
        "source_evidence": {
            "preparation": prep_id,
            "launch_receipt": launch_id,
            "arm_validations": normalized_ids,
        },
        "validated_preparation": {
            "scientific_status": prepared["scientific_status"],
            "screen_execution_checkout": copy.deepcopy(
                prepared["screen_execution_checkout"]
            ),
            "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        },
        "thresholds": copy.deepcopy(CONFIRMATION_THRESHOLDS),
        "arm_summaries": summaries,
        "cross_arm_evidence": cross_arm,
        "directional_comparisons": evaluated["directional_comparisons"],
        "support_comparisons": evaluated["support_comparisons"],
        "checks": evaluated["checks"],
        "failed_checks": evaluated["failed_checks"],
        "next_stage": {
            "route": "large_capacity_readiness_preparation" if passed else "hold",
            "support_collapse_resolved": passed,
            "large_capacity_readiness_preparation_allowed": passed,
            "large_capacity_readiness_launch_allowed": False,
            "separate_source_bound_authorization_required": True,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "claim_policy": {
            "terminal_snr_intervention_confirmed": passed,
            "support_collapse_resolved": passed,
            "confirmation_is_large_capacity_qualification_only": True,
            "formal_generation_claim_allowed": False,
            "generation_advantage_proven": False,
            "full_300k_requires_separate_readiness_and_authorization": True,
        },
        "authorization_boundary": copy.deepcopy(RESULT_BOUNDARY),
    }
    return validate_terminal_snr_confirmation_result_contract(report)


def validate_terminal_snr_confirmation_result_contract(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation result")
    summaries = _object(
        row.get("arm_summaries"), "terminal-SNR confirmation arm summaries"
    )
    if set(summaries) != set(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation summary arm set differs")
    runtime_environment_sha256 = row.get("runtime_environment_sha256")
    cross_arm = _cross_arm(summaries, str(runtime_environment_sha256))
    evaluated = _evaluate(summaries)
    passed = bool(evaluated["confirmation_pass"])
    next_stage = _object(row.get("next_stage"), "terminal-SNR confirmation next stage")
    claim = _object(row.get("claim_policy"), "terminal-SNR confirmation claim policy")
    validated_preparation = _object(
        row.get("validated_preparation"),
        "terminal-SNR confirmation validated preparation",
    )
    preparation_evaluation = _object(
        validated_preparation.get("evaluation_contract"),
        "terminal-SNR confirmation preparation evaluation",
    )
    sources = _object(
        row.get("source_evidence"), "terminal-SNR confirmation result sources"
    )
    _identity(sources.get("preparation"), "terminal-SNR confirmation preparation")
    _identity(sources.get("launch_receipt"), "terminal-SNR confirmation launch")
    arm_ids = _object(sources.get("arm_validations"), "confirmation arm sources")
    if set(sources) != {"preparation", "launch_receipt", "arm_validations"}:
        raise ValueError("terminal-SNR confirmation result source set differs")
    if set(arm_ids) != set(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation source arm set differs")
    for arm in ARM_NAMES:
        _identity(arm_ids[arm], f"{arm} confirmation source")
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "operational_status",
            "scientific_status",
            "terminal_status",
            "confirmation_pass",
            "support_collapse_resolved",
            "generation_advantage_proven",
            "decision",
            "result_git",
            "runtime_environment_sha256",
            "source_evidence",
            "validated_preparation",
            "thresholds",
            "arm_summaries",
            "cross_arm_evidence",
            "directional_comparisons",
            "support_comparisons",
            "checks",
            "failed_checks",
            "next_stage",
            "claim_policy",
            "authorization_boundary",
        }
        or set(validated_preparation)
        != {
            "scientific_status",
            "screen_execution_checkout",
            "evaluation_contract",
        }
        or validated_preparation.get("scientific_status")
        != "terminal_snr_confirmation_prepared"
        or _git(
            validated_preparation.get("screen_execution_checkout"),
            "terminal-SNR confirmation preparation screen Git",
        )
        != cross_arm.get("screen_execution_git")
        or int(preparation_evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or preparation_evaluation.get("arms") != list(ARM_NAMES)
        or preparation_evaluation.get("frozen_checkpoint_training_allowed") is not False
        or preparation_evaluation != EVALUATION_CONTRACT
        or set(next_stage)
        != {
            "route",
            "support_collapse_resolved",
            "large_capacity_readiness_preparation_allowed",
            "large_capacity_readiness_launch_allowed",
            "separate_source_bound_authorization_required",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
        }
        or set(claim)
        != {
            "terminal_snr_intervention_confirmed",
            "support_collapse_resolved",
            "confirmation_is_large_capacity_qualification_only",
            "formal_generation_claim_allowed",
            "generation_advantage_proven",
            "full_300k_requires_separate_readiness_and_authorization",
        }
        or row.get("schema_version") != RESULT_SCHEMA
        or row.get("role") != RESULT_ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("scientific_status")
        != ("confirmation_pass" if passed else "hold")
        or row.get("terminal_status") != "hold"
        or row.get("confirmation_pass") is not passed
        or row.get("support_collapse_resolved") is not passed
        or row.get("generation_advantage_proven") is not False
        or row.get("decision")
        != (
            "prepare_separate_large_capacity_readiness"
            if passed
            else "hold_terminal_snr_intervention"
        )
        or row.get("thresholds") != CONFIRMATION_THRESHOLDS
        or row.get("cross_arm_evidence") != cross_arm
        or row.get("directional_comparisons")
        != evaluated["directional_comparisons"]
        or row.get("support_comparisons") != evaluated["support_comparisons"]
        or row.get("checks") != evaluated["checks"]
        or row.get("failed_checks") != evaluated["failed_checks"]
        or row.get("authorization_boundary") != RESULT_BOUNDARY
        or next_stage.get("route")
        != ("large_capacity_readiness_preparation" if passed else "hold")
        or next_stage.get("support_collapse_resolved") is not passed
        or next_stage.get("large_capacity_readiness_preparation_allowed") is not passed
        or next_stage.get("large_capacity_readiness_launch_allowed") is not False
        or next_stage.get("separate_source_bound_authorization_required") is not True
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or claim.get("terminal_snr_intervention_confirmed") is not passed
        or claim.get("support_collapse_resolved") is not passed
        or claim.get("confirmation_is_large_capacity_qualification_only") is not True
        or claim.get("formal_generation_claim_allowed") is not False
        or claim.get("generation_advantage_proven") is not False
        or claim.get("full_300k_requires_separate_readiness_and_authorization")
        is not True
    ):
        raise ValueError("terminal-SNR confirmation result contract differs")
    _git(row.get("result_git"), "terminal-SNR confirmation result Git")
    return copy.deepcopy(row)


def _load_bound(identity_row: Mapping[str, Any], name: str) -> dict[str, Any]:
    expected = _identity(identity_row, name)
    actual = identity(expected["path"])
    if actual != expected:
        raise ValueError(f"{name} physical identity differs")
    return read_object(expected["path"], name=name)


def replay_terminal_snr_confirmation_result(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = validate_terminal_snr_confirmation_result_contract(report)
    sources = _object(row["source_evidence"], "confirmation result sources")
    preparation = _load_bound(sources["preparation"], "confirmation preparation")
    launch = _load_bound(sources["launch_receipt"], "confirmation launch receipt")
    validate_terminal_snr_confirmation_launch_receipt_physical(
        launch, expected_execution_checkout=row["result_git"]
    )
    arm_ids = _object(sources["arm_validations"], "confirmation arm sources")
    arms = {
        arm: _load_bound(arm_ids[arm], f"{arm} confirmation validation")
        for arm in ARM_NAMES
    }
    return build_terminal_snr_confirmation_result(
        preparation=preparation,
        preparation_identity=sources["preparation"],
        launch_receipt=launch,
        launch_receipt_identity=sources["launch_receipt"],
        arm_validations=arms,
        arm_validation_identities=arm_ids,
        result_git=row["result_git"],
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_terminal_snr_confirmation_validation_receipt(
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_terminal_snr_confirmation_result_contract(result)
    replayed = replay_terminal_snr_confirmation_result(validated)
    if replayed != validated:
        raise ValueError("terminal-SNR confirmation differs from physical replay")
    result_id = _identity(result_identity, "terminal-SNR confirmation result")
    validator = _git(validator_git, "terminal-SNR confirmation validator")
    basis = {
        "result": result_id,
        "result_git": validated["result_git"],
        "validator_git": validator,
        "confirmation_pass": validated["confirmation_pass"],
        "failed_checks": validated["failed_checks"],
        "source_evidence": validated["source_evidence"],
    }
    return {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "scientific_status": validated["scientific_status"],
        "confirmation_pass": validated["confirmation_pass"],
        "support_collapse_resolved": validated["support_collapse_resolved"],
        "result": result_id,
        "result_git": copy.deepcopy(validated["result_git"]),
        "validator_git": validator,
        "failed_checks": copy.deepcopy(validated["failed_checks"]),
        "source_evidence": copy.deepcopy(validated["source_evidence"]),
        "validation_basis_sha256": _canonical_sha256(basis),
        "generation_advantage_proven": False,
        "authorization_boundary": copy.deepcopy(VALIDATION_BOUNDARY),
    }


def validate_terminal_snr_confirmation_validation_receipt(
    receipt: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(receipt, "terminal-SNR confirmation validation")
    expected = build_terminal_snr_confirmation_validation_receipt(
        result=result,
        result_identity=result_identity,
        validator_git=validator_git,
    )
    if row != expected:
        raise ValueError("terminal-SNR confirmation validation receipt differs")
    return copy.deepcopy(row)


__all__ = [
    "CONFIRMATION_THRESHOLDS",
    "RESULT_BOUNDARY",
    "RESULT_ROLE",
    "RESULT_SCHEMA",
    "VALIDATION_BOUNDARY",
    "VALIDATION_ROLE",
    "VALIDATION_SCHEMA",
    "build_terminal_snr_confirmation_result",
    "build_terminal_snr_confirmation_validation_receipt",
    "replay_terminal_snr_confirmation_result",
    "validate_terminal_snr_confirmation_result_contract",
    "validate_terminal_snr_confirmation_validation_receipt",
]
