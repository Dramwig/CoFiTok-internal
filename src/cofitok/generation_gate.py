from __future__ import annotations

import hashlib
import json
import math
from typing import Any


GENERATION_GATE_SCHEMA_VERSION = 1

_STAGE_DECISIONS = {
    "scaling": "promote_to_full_imagenet256",
    "full": "large_scale_generation_ready",
}

_COMMON_REQUIRED_GATES = frozenset(
    {
        "training_complete",
        "matched_training_revision",
        "matched_training_protocol",
        "training_cost_accounting",
        "matched_generation_protocol",
        "generation_metrics_complete",
        "matched_real_set_provenance",
        "matched_evaluator_code_provenance",
        "matched_evaluator_runtime_environment",
        "distribution_metric_ranges",
        "matched_sampling_provenance",
        "formal_sampling_protocol",
        "matched_sampling_code_provenance",
        "matched_sampling_runtime_environment",
        "checkpoint_evaluation_provenance",
        "matched_checkpoint_evaluator_code_provenance",
        "fid_within_tolerance",
        "absolute_fid_quality",
        "endpoint_within_tolerance",
        "ordered_prefix_path",
        "restricted_synthesis_contract",
        "shuffle_mismatch",
    }
)

REQUIRED_GENERATION_GATES = {
    "scaling": _COMMON_REQUIRED_GATES,
    "full": _COMMON_REQUIRED_GATES
    | {
        "full_precision_recall_quality",
        "full_training_checkpoint_integrity",
    },
}

REQUIRED_GENERATION_GATE_THRESHOLDS = {
    "scaling": {
        "min_samples": ("min", 10_000.0),
        "max_fid_regression": ("max", 0.05),
        "max_absolute_fid": ("max", 100.0),
        "max_endpoint_regression": ("max", 0.05),
    },
    "full": {
        "min_samples": ("min", 50_000.0),
        "max_fid_regression": ("max", 0.05),
        "max_absolute_fid": ("max", 20.0),
        "max_endpoint_regression": ("max", 0.05),
        "min_precision": ("min", 0.30),
        "min_recall": ("min", 0.30),
        "max_precision_regression": ("max", 0.05),
        "max_recall_regression": ("max", 0.05),
    },
}


def generation_gate_identity_sha256(gate: dict[str, Any]) -> str:
    canonical = json.dumps(
        gate,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _finite_number(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"generation gate {name} is not numeric") from error
    if not math.isfinite(numeric):
        raise ValueError(f"generation gate {name} is not finite")
    return numeric


def _validate_thresholds(gate: dict[str, Any], *, stage: str) -> dict[str, float]:
    raw = gate.get("thresholds")
    if not isinstance(raw, dict):
        raise ValueError("generation gate thresholds are missing")
    validated: dict[str, float] = {}
    for name, (direction, boundary) in REQUIRED_GENERATION_GATE_THRESHOLDS[stage].items():
        value = _finite_number(raw.get(name), name=f"threshold {name}")
        if (direction == "max" and value > boundary) or (
            direction == "min" and value < boundary
        ):
            raise ValueError(
                f"generation gate threshold {name}={value:g} is weaker than "
                f"the required {direction} boundary {boundary:g}"
            )
        validated[name] = value
    return validated


def _validate_summary(gate: dict[str, Any], *, stage: str, thresholds: dict[str, float]) -> None:
    summary = gate.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("generation gate summary is missing")
    cofitok_fid = _finite_number(summary.get("cofitok_fid"), name="summary cofitok_fid")
    dense_fid = _finite_number(summary.get("dense_fid"), name="summary dense_fid")
    if cofitok_fid < 0.0 or dense_fid <= 0.0:
        raise ValueError("generation gate FID values are outside their valid range")
    if cofitok_fid > thresholds["max_absolute_fid"]:
        raise ValueError("generation gate summary violates the absolute FID threshold")
    if cofitok_fid > dense_fid * (1.0 + thresholds["max_fid_regression"]):
        raise ValueError("generation gate summary violates the relative FID threshold")

    cofitok_endpoint = _finite_number(
        summary.get("cofitok_endpoint_mse"), name="summary cofitok_endpoint_mse"
    )
    dense_endpoint = _finite_number(
        summary.get("dense_endpoint_mse"), name="summary dense_endpoint_mse"
    )
    if cofitok_endpoint < 0.0 or dense_endpoint <= 0.0:
        raise ValueError("generation gate endpoint MSE values are outside their valid range")
    if cofitok_endpoint > dense_endpoint * (1.0 + thresholds["max_endpoint_regression"]):
        raise ValueError("generation gate summary violates the endpoint MSE threshold")

    if int(summary.get("ordered_rank", -1)) != 1 or int(summary.get("order_count", 0)) < 10:
        raise ValueError("generation gate summary does not prove ordered-prefix rank 1")

    if stage == "full":
        cofitok_precision = _finite_number(
            summary.get("cofitok_precision"), name="summary cofitok_precision"
        )
        dense_precision = _finite_number(
            summary.get("dense_precision"), name="summary dense_precision"
        )
        cofitok_recall = _finite_number(
            summary.get("cofitok_recall"), name="summary cofitok_recall"
        )
        dense_recall = _finite_number(
            summary.get("dense_recall"), name="summary dense_recall"
        )
        if any(
            value < 0.0 or value > 1.0
            for value in (cofitok_precision, dense_precision, cofitok_recall, dense_recall)
        ):
            raise ValueError("generation gate precision/recall values are outside [0, 1]")
        if cofitok_precision < thresholds["min_precision"]:
            raise ValueError("generation gate summary violates the minimum precision threshold")
        if cofitok_recall < thresholds["min_recall"]:
            raise ValueError("generation gate summary violates the minimum recall threshold")
        if cofitok_precision < dense_precision - thresholds["max_precision_regression"]:
            raise ValueError("generation gate summary violates the relative precision threshold")
        if cofitok_recall < dense_recall - thresholds["max_recall_regression"]:
            raise ValueError("generation gate summary violates the relative recall threshold")


def _validate_scientific_gate_evidence(
    gate: dict[str, Any],
    *,
    stage: str,
    thresholds: dict[str, float],
    indexed: dict[str, dict[str, Any]],
) -> None:
    summary = gate["summary"]

    def evidence(name: str) -> dict[str, Any]:
        raw = indexed[name].get("evidence")
        if not isinstance(raw, dict):
            raise ValueError(f"generation gate check {name} has no evidence")
        return raw

    def require_same(name: str, left: Any, right: Any) -> None:
        left_value = _finite_number(left, name=name)
        right_value = _finite_number(right, name=name)
        if left_value != right_value:
            raise ValueError(f"generation gate {name} evidence differs from its summary")

    fid = evidence("fid_within_tolerance")
    require_same("cofitok_fid", fid.get("cofitok_fid"), summary.get("cofitok_fid"))
    require_same("dense_fid", fid.get("dense_fid"), summary.get("dense_fid"))
    require_same("max_fid_regression", fid.get("max_regression"), thresholds["max_fid_regression"])

    absolute_fid = evidence("absolute_fid_quality")
    require_same(
        "absolute cofitok_fid",
        absolute_fid.get("cofitok_fid"),
        summary.get("cofitok_fid"),
    )
    require_same(
        "max_absolute_fid",
        absolute_fid.get("max_absolute_fid"),
        thresholds["max_absolute_fid"],
    )

    endpoint = evidence("endpoint_within_tolerance")
    require_same(
        "cofitok_endpoint_mse",
        endpoint.get("cofitok_endpoint_mse"),
        summary.get("cofitok_endpoint_mse"),
    )
    require_same(
        "dense_endpoint_mse",
        endpoint.get("dense_endpoint_mse"),
        summary.get("dense_endpoint_mse"),
    )
    require_same(
        "max_endpoint_regression",
        endpoint.get("max_regression"),
        thresholds["max_endpoint_regression"],
    )

    ordered = evidence("ordered_prefix_path")
    if int(ordered.get("rank", -1)) != int(summary.get("ordered_rank", -1)):
        raise ValueError("generation gate ordered-prefix rank evidence differs from its summary")
    if int(ordered.get("order_count", 0)) != int(summary.get("order_count", 0)):
        raise ValueError("generation gate order-count evidence differs from its summary")
    zero_token = _finite_number(
        evidence("restricted_synthesis_contract").get("zero_token_max_abs"),
        name="zero_token_max_abs",
    )
    if zero_token != 0.0:
        raise ValueError("generation gate does not prove exact zero-token synthesis")
    shuffle_ratio = _finite_number(
        evidence("shuffle_mismatch").get("shuffled_to_ordered_endpoint_ratio"),
        name="shuffled_to_ordered_endpoint_ratio",
    )
    if shuffle_ratio <= 1.0:
        raise ValueError("generation gate does not prove shuffled-token mismatch")

    if stage == "full":
        quality = evidence("full_precision_recall_quality")
        if quality.get("enforced") is not True:
            raise ValueError("full generation gate does not enforce precision/recall quality")
        for name in ("min_precision", "min_recall", "max_precision_regression", "max_recall_regression"):
            require_same(name, quality.get(name), thresholds[name])
        for name in ("cofitok_precision", "dense_precision", "cofitok_recall", "dense_recall"):
            require_same(name, quality.get(name), summary.get(name))


def validate_generation_gate_authorization(
    gate: dict[str, Any], *, expected_stage: str
) -> dict[str, Any]:
    if expected_stage not in _STAGE_DECISIONS:
        raise ValueError(f"unsupported generation gate stage: {expected_stage}")
    if int(gate.get("schema_version", -1)) != GENERATION_GATE_SCHEMA_VERSION:
        raise ValueError("generation gate schema version is unsupported")
    if gate.get("stage") != expected_stage:
        raise ValueError(f"expected {expected_stage} generation gate")
    expected_decision = _STAGE_DECISIONS[expected_stage]
    if gate.get("status") != "pass" or gate.get("decision") != expected_decision:
        raise ValueError(f"{expected_stage} gate did not authorize {expected_decision}")

    rows = gate.get("gates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("generation gate checks are missing")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("generation gate check is malformed")
        name = str(row.get("name", ""))
        if not name:
            raise ValueError("generation gate check has no name")
        if name in indexed:
            raise ValueError(f"generation gate contains duplicate check: {name}")
        if row.get("passed") is not True:
            raise ValueError(f"generation gate contains a failed check: {name}")
        indexed[name] = row
    missing = sorted(REQUIRED_GENERATION_GATES[expected_stage] - indexed.keys())
    if missing:
        raise ValueError("generation gate lacks required checks: " + ", ".join(missing))

    thresholds = _validate_thresholds(gate, stage=expected_stage)
    _validate_summary(gate, stage=expected_stage, thresholds=thresholds)
    _validate_scientific_gate_evidence(
        gate,
        stage=expected_stage,
        thresholds=thresholds,
        indexed=indexed,
    )
    return {
        "schema_version": GENERATION_GATE_SCHEMA_VERSION,
        "stage": expected_stage,
        "decision": expected_decision,
        "gate_count": len(rows),
        "validated_thresholds": thresholds,
    }
