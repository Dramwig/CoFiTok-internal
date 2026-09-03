"""Source-bound scientific routing after the bounded 110K exposure result.

The decision produced here never authorizes a GPU job or full training.  It
only determines whether the additional exposure cleared the scaling-grade
non-collapse checks, whether a separately authorized 250M capacity screen is
scientifically justified, or whether the route must remain held.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from cofitok.generation.exposure_capacity import validate_preparation
from cofitok.generation.exposure_capacity_result import (
    RESULT_ROLE,
    RESULT_SCHEMA,
    validate_validation_receipt,
)
from cofitok.generation.stability_qualification import (
    build_stability_qualification,
)


DECISION_SCHEMA = "cofitok_generation_exposure_capacity_scientific_decision_v1"
DECISION_ROLE = "source_bound_exposure_capacity_scientific_route"
METHODS = ("cofitok", "dense_identity")
SOURCE_REPORT_NAMES = frozenset(
    {
        "cofitok_training",
        "dense_training",
        "cofitok_checkpoint",
        "dense_checkpoint",
        "cofitok_rollout",
        "dense_rollout",
    }
)
THRESHOLDS = {
    "max_absolute_fid": 100.0,
    "max_fid_regression": 0.05,
    "min_precision": 0.10,
    "min_recall": 0.10,
    "max_precision_regression": 0.05,
    "max_recall_regression": 0.05,
    "min_top1": 0.01,
    "min_top5": 0.05,
    "min_predicted_class_fraction": 0.25,
    "min_normalized_predicted_entropy": 0.50,
    "max_top1_regression": 0.05,
    "max_top5_regression": 0.05,
}
AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite domain")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _finite(value, name, minimum=0.0)
    if result > 1.0:
        raise ValueError(f"{name} is outside [0, 1]")
    return result


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    path = row.get("path")
    size = row.get("bytes")
    digest = row.get("sha256")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} path is missing")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"{name} byte count is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} SHA256 is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    revision = row.get("revision")
    tree = row.get("tree")
    branch = row.get("branch")
    for label, digest in (("revision", revision), ("tree", tree)):
        if (
            not isinstance(digest, str)
            or len(digest) != 40
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} {label} is malformed")
    if not isinstance(branch, str) or not branch or row.get("tracked_dirty") is not False:
        raise ValueError(f"{name} must identify one exact clean checkout")
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def _method_metrics(
    *,
    distribution: Mapping[str, Any],
    class_fidelity: Mapping[str, Any],
    label: str,
    exposure_result: bool,
) -> dict[str, float]:
    distribution_row = _object(distribution, f"{label} distribution metrics")
    class_row = _object(class_fidelity, f"{label} class-fidelity metrics")
    fid_key = "frechet_inception_distance" if exposure_result else "fid"
    return {
        "fid": _finite(distribution_row.get(fid_key), f"{label} FID", minimum=0.0),
        "precision": _fraction(
            distribution_row.get("precision"), f"{label} precision"
        ),
        "recall": _fraction(distribution_row.get("recall"), f"{label} recall"),
        "top1": _fraction(class_row.get("top1_accuracy"), f"{label} top1"),
        "top5": _fraction(class_row.get("top5_accuracy"), f"{label} top5"),
        "predicted_class_fraction": _fraction(
            class_row.get("predicted_class_fraction"),
            f"{label} predicted-class fraction",
        ),
        "normalized_predicted_class_entropy": _fraction(
            class_row.get("normalized_predicted_class_entropy"),
            f"{label} normalized predicted-class entropy",
        ),
    }


def _exposure_metrics(result: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    quality = _object(result.get("quality"), "exposure result quality")
    class_fidelity = _object(
        result.get("class_fidelity"), "exposure result class fidelity"
    )
    if set(quality) != set(METHODS) or set(class_fidelity) != set(METHODS):
        raise ValueError("exposure result method metrics are incomplete")
    return {
        method: _method_metrics(
            distribution=_object(quality[method], f"exposure {method} quality").get(
                "metrics", {}
            ),
            class_fidelity=_object(
                class_fidelity[method], f"exposure {method} class fidelity"
            ).get("metrics", {}),
            label=f"exposure {method}",
            exposure_result=True,
        )
        for method in METHODS
    }


def _source_metrics(result: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    if (
        int(result.get("schema_version", -1)) != 1
        or result.get("role") != "stability_full_data_quality_bridge_result"
        or result.get("status") != "completed"
        or result.get("stage") != "stability_quality_bridge"
    ):
        raise ValueError("100K quality-bridge result contract differs")
    terminal = _object(result.get("terminal"), "100K quality-bridge terminal")
    methods = _object(terminal.get("methods"), "100K quality-bridge methods")
    qualification = _object(
        terminal.get("class_fidelity"), "100K class-fidelity qualification"
    )
    class_metrics = _object(
        qualification.get("metrics"), "100K class-fidelity metrics"
    )
    if set(methods) != set(METHODS) or not set(METHODS).issubset(class_metrics):
        raise ValueError("100K quality-bridge method metrics are incomplete")
    return {
        method: _method_metrics(
            distribution=methods[method],
            class_fidelity=class_metrics[method],
            label=f"100K {method}",
            exposure_result=False,
        )
        for method in METHODS
    }


def _expected_raw_source_identities(
    result: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    training = _object(result.get("training"), "exposure result training")
    mechanism = _object(result.get("mechanism"), "exposure result mechanism")
    rollout = _object(result.get("rollout"), "exposure result rollout")
    expected = {}
    for method, prefix in (("cofitok", "cofitok"), ("dense_identity", "dense")):
        expected[f"{prefix}_training"] = _identity(
            _object(training.get(method), f"{method} training").get("report"),
            f"{method} training report",
        )
        expected[f"{prefix}_checkpoint"] = _identity(
            _object(mechanism.get(method), f"{method} mechanism").get("report"),
            f"{method} checkpoint report",
        )
        expected[f"{prefix}_rollout"] = _identity(
            _object(rollout.get(method), f"{method} rollout").get("report"),
            f"{method} rollout report",
        )
    return expected


def _check(
    name: str,
    passed: bool,
    observed: Any,
    comparison: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
    }


def _quality_checks(metrics: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    cofitok = metrics["cofitok"]
    dense = metrics["dense_identity"]
    checks = [
        _check(
            "cofitok_absolute_fid",
            cofitok["fid"] <= THRESHOLDS["max_absolute_fid"],
            cofitok["fid"],
            "<=",
            THRESHOLDS["max_absolute_fid"],
        ),
        _check(
            "matched_fid_tolerance",
            cofitok["fid"] <= dense["fid"] * (1.0 + THRESHOLDS["max_fid_regression"]),
            cofitok["fid"] / max(dense["fid"], 1e-12) - 1.0,
            "<=",
            THRESHOLDS["max_fid_regression"],
        ),
        _check(
            "cofitok_precision_floor",
            cofitok["precision"] >= THRESHOLDS["min_precision"],
            cofitok["precision"],
            ">=",
            THRESHOLDS["min_precision"],
        ),
        _check(
            "cofitok_recall_floor",
            cofitok["recall"] >= THRESHOLDS["min_recall"],
            cofitok["recall"],
            ">=",
            THRESHOLDS["min_recall"],
        ),
        _check(
            "matched_precision_tolerance",
            dense["precision"] - cofitok["precision"]
            <= THRESHOLDS["max_precision_regression"],
            dense["precision"] - cofitok["precision"],
            "<=",
            THRESHOLDS["max_precision_regression"],
        ),
        _check(
            "matched_recall_tolerance",
            dense["recall"] - cofitok["recall"]
            <= THRESHOLDS["max_recall_regression"],
            dense["recall"] - cofitok["recall"],
            "<=",
            THRESHOLDS["max_recall_regression"],
        ),
    ]
    for method in METHODS:
        row = metrics[method]
        for metric, threshold_name in (
            ("top1", "min_top1"),
            ("top5", "min_top5"),
            ("predicted_class_fraction", "min_predicted_class_fraction"),
            (
                "normalized_predicted_class_entropy",
                "min_normalized_predicted_entropy",
            ),
        ):
            checks.append(
                _check(
                    f"{method}_{metric}_floor",
                    row[metric] >= THRESHOLDS[threshold_name],
                    row[metric],
                    ">=",
                    THRESHOLDS[threshold_name],
                )
            )
    for metric, threshold_name in (
        ("top1", "max_top1_regression"),
        ("top5", "max_top5_regression"),
    ):
        checks.append(
            _check(
                f"cofitok_{metric}_regression_vs_dense",
                dense[metric] - cofitok[metric] <= THRESHOLDS[threshold_name],
                dense[metric] - cofitok[metric],
                "<=",
                THRESHOLDS[threshold_name],
            )
        )
    return checks


def build_decision(
    *,
    exposure_result: Mapping[str, Any],
    exposure_result_identity: Mapping[str, Any],
    validation_receipt: Mapping[str, Any],
    validation_receipt_identity: Mapping[str, Any],
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    quality_bridge_result: Mapping[str, Any],
    quality_bridge_result_identity: Mapping[str, Any],
    raw_source_reports: Mapping[str, Mapping[str, Any]],
    raw_source_identities: Mapping[str, Mapping[str, Any]],
    decision_git: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        exposure_result.get("schema_version") != RESULT_SCHEMA
        or exposure_result.get("role") != RESULT_ROLE
        or exposure_result.get("status") != "completed"
        or exposure_result.get("operational_status") != "pass"
        or exposure_result.get("terminal_status") != "hold"
        or exposure_result.get("generation_advantage_proven") is not False
    ):
        raise ValueError("exposure decision requires the canonical completed result")
    result_identity = _identity(exposure_result_identity, "exposure result")
    receipt_identity = _identity(
        validation_receipt_identity, "exposure result validation receipt"
    )
    validated_receipt = validate_validation_receipt(
        validation_receipt,
        result=exposure_result,
        result_identity=result_identity,
    )
    prepared = validate_preparation(dict(preparation))
    prepared_identity = _identity(preparation_identity, "exposure preparation")
    if exposure_result.get("preparation") != prepared_identity:
        raise ValueError("exposure result binds another preparation")
    quality_identity = _identity(
        quality_bridge_result_identity, "100K quality-bridge result"
    )
    prepared_sources = _object(preparation.get("sources"), "preparation sources")
    if _identity(
        prepared_sources.get("quality_bridge_result"),
        "prepared quality-bridge result",
    ) != quality_identity:
        raise ValueError("decision quality-bridge result differs from preparation")
    if (
        set(raw_source_reports) != SOURCE_REPORT_NAMES
        or set(raw_source_identities) != SOURCE_REPORT_NAMES
    ):
        raise ValueError("exposure decision raw source set differs")
    expected_raw = _expected_raw_source_identities(exposure_result)
    normalized_raw_identities = {
        name: _identity(raw_source_identities[name], f"raw source {name}")
        for name in sorted(SOURCE_REPORT_NAMES)
    }
    if normalized_raw_identities != expected_raw:
        raise ValueError("exposure decision raw sources differ from the validated result")
    execution_git = _git(
        exposure_result.get("execution_checkout"), "exposure execution checkout"
    )
    normalized_decision_git = _git(decision_git, "exposure scientific decision")
    stability = build_stability_qualification(
        cofitok_training=dict(raw_source_reports["cofitok_training"]),
        dense_training=dict(raw_source_reports["dense_training"]),
        cofitok_checkpoint=dict(raw_source_reports["cofitok_checkpoint"]),
        dense_checkpoint=dict(raw_source_reports["dense_checkpoint"]),
        cofitok_rollout=dict(raw_source_reports["cofitok_rollout"]),
        dense_rollout=dict(raw_source_reports["dense_rollout"]),
        expected_weights="ema",
        expected_evaluation_revision=str(execution_git["revision"]),
        expected_evaluation_branch=str(execution_git["branch"]),
    )
    source_metrics = _source_metrics(quality_bridge_result)
    exposure_metrics = _exposure_metrics(exposure_result)
    checks = _quality_checks(exposure_metrics)
    checks.append(
        _check(
            "mechanism_and_rollout_stability",
            stability.get("status") == "pass",
            stability.get("status"),
            "==",
            "pass",
        )
    )
    failed = [row["name"] for row in checks if row["passed"] is not True]
    quality_pass = not failed
    cofitok = exposure_metrics["cofitok"]
    dense = exposure_metrics["dense_identity"]
    shared_collapse_signals = {
        "both_fid_above_scaling_floor": (
            cofitok["fid"] > THRESHOLDS["max_absolute_fid"]
            and dense["fid"] > THRESHOLDS["max_absolute_fid"]
        ),
        "both_recall_below_scaling_floor": (
            cofitok["recall"] < THRESHOLDS["min_recall"]
            and dense["recall"] < THRESHOLDS["min_recall"]
        ),
        "both_top1_below_scaling_floor": (
            cofitok["top1"] < THRESHOLDS["min_top1"]
            and dense["top1"] < THRESHOLDS["min_top1"]
        ),
        "both_top5_below_scaling_floor": (
            cofitok["top5"] < THRESHOLDS["min_top5"]
            and dense["top5"] < THRESHOLDS["min_top5"]
        ),
    }
    shared_collapse = any(shared_collapse_signals.values())
    matched_specific_failures = {
        row["name"]
        for row in checks
        if row["passed"] is not True
        and row["name"]
        in {
            "matched_fid_tolerance",
            "matched_precision_tolerance",
            "matched_recall_tolerance",
            "cofitok_top1_regression_vs_dense",
            "cofitok_top5_regression_vs_dense",
            "mechanism_and_rollout_stability",
        }
    }
    if quality_pass:
        route = "exposure_qualified"
        reason = "additional_exposure_cleared_all_scaling_grade_noncollapse_checks"
    elif shared_collapse and not matched_specific_failures and stability.get("status") == "pass":
        route = "capacity_screen"
        reason = "shared_support_or_class_collapse_persisted_after_additional_exposure"
    else:
        route = "hold"
        reason = "exposure_evidence_does_not_support_capacity_as_a_clean_single_variable_test"
    deltas = {
        method: {
            metric: exposure_metrics[method][metric] - source_metrics[method][metric]
            for metric in exposure_metrics[method]
        }
        for method in METHODS
    }
    next_stage = {
        "route": route,
        "reason": reason,
        "exposure_qualification_passed": route == "exposure_qualified",
        "capacity_screen_preparation_allowed": route == "capacity_screen",
        "large_capacity_readiness_preparation_allowed": route == "exposure_qualified",
        "gpu_execution_allowed": False,
        "training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "required_next_evidence": (
            "A separate source-bound large-capacity readiness decision."
            if route == "exposure_qualified"
            else (
                "A separate capacity-screen preparation and exact execution authorization."
                if route == "capacity_screen"
                else "A new matched intervention justified by the named failed checks."
            )
        ),
    }
    return {
        "schema_version": DECISION_SCHEMA,
        "role": DECISION_ROLE,
        "status": "completed",
        "scientific_status": route,
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "decision": route,
        "decision_git": normalized_decision_git,
        "source_evidence": {
            "exposure_result": result_identity,
            "exposure_result_validation_receipt": receipt_identity,
            "validated_receipt": {
                "status": validated_receipt["status"],
                "validation_basis_sha256": validated_receipt[
                    "validation_basis_sha256"
                ],
                "execution_checkout": validated_receipt["execution_checkout"],
                "validator_git": validated_receipt["validator_git"],
            },
            "preparation": prepared_identity,
            "quality_bridge_result": quality_identity,
            "raw_reports": normalized_raw_identities,
            "execution_git": execution_git,
        },
        "thresholds": copy.deepcopy(THRESHOLDS),
        "source_100k_metrics": source_metrics,
        "exposure_110k_metrics": exposure_metrics,
        "exposure_minus_source": deltas,
        "checks": checks,
        "failed_checks": failed,
        "shared_collapse": {
            "present": shared_collapse,
            "signals": shared_collapse_signals,
        },
        "cofitok_specific_or_mechanism_failures": sorted(matched_specific_failures),
        "stability_qualification": stability,
        "next_stage": next_stage,
        "validated_preparation": {
            "decision": preparation["decision"],
            "route_id": prepared["route_id"],
        },
        "claim_policy": {
            "formal_generation_claim_allowed": False,
            "cross_stage_numeric_ranking_allowed": False,
            "generation_advantage_proven": False,
            "new_execution_authorization_required": True,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_decision_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(report, "exposure/capacity scientific decision")
    route = row.get("decision")
    next_stage = _object(row.get("next_stage"), "decision next stage")
    if (
        row.get("schema_version") != DECISION_SCHEMA
        or row.get("role") != DECISION_ROLE
        or row.get("status") != "completed"
        or route not in {"exposure_qualified", "capacity_screen", "hold"}
        or row.get("scientific_status") != route
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("thresholds") != THRESHOLDS
        or row.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or next_stage.get("route") != route
        or next_stage.get("gpu_execution_allowed") is not False
        or next_stage.get("training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or next_stage.get("capacity_screen_preparation_allowed")
        is not (route == "capacity_screen")
        or next_stage.get("large_capacity_readiness_preparation_allowed")
        is not (route == "exposure_qualified")
    ):
        raise ValueError("exposure/capacity scientific decision contract differs")
    checks = row.get("checks")
    failed = row.get("failed_checks")
    if not isinstance(checks, list) or not checks or not isinstance(failed, list):
        raise ValueError("exposure/capacity decision checks are incomplete")
    names: set[str] = set()
    computed_failed: list[str] = []
    for check in checks:
        item = _object(check, "decision check")
        name = item.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("exposure/capacity decision check names differ")
        if item.get("passed") not in {True, False}:
            raise ValueError(f"decision check lacks a boolean result: {name}")
        names.add(name)
        if item["passed"] is False:
            computed_failed.append(name)
    if failed != computed_failed:
        raise ValueError("exposure/capacity decision failed-check list differs")
    if route == "exposure_qualified" and computed_failed:
        raise ValueError("qualified exposure decision retains failed checks")
    if route == "capacity_screen":
        shared = _object(row.get("shared_collapse"), "shared-collapse decision")
        if shared.get("present") is not True or row.get(
            "cofitok_specific_or_mechanism_failures"
        ) != []:
            raise ValueError("capacity-screen selection is not scientifically isolated")
    _git(row.get("decision_git"), "exposure scientific decision")
    return copy.deepcopy(row)


__all__ = [
    "AUTHORIZATION_BOUNDARY",
    "DECISION_ROLE",
    "DECISION_SCHEMA",
    "SOURCE_REPORT_NAMES",
    "THRESHOLDS",
    "build_decision",
    "validate_decision_contract",
]
