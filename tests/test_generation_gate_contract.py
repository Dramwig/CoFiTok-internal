from __future__ import annotations

import copy

import pytest

from cofitok.generation_gate import (
    REQUIRED_GENERATION_GATES,
    validate_generation_gate_authorization,
)


def _gate(stage: str = "scaling") -> dict:
    full = stage == "full"
    summary = {
        "cofitok_fid": 19.5 if full else 90.0,
        "dense_fid": 19.0 if full else 89.0,
        "cofitok_endpoint_mse": 0.104,
        "dense_endpoint_mse": 0.1,
        "cofitok_precision": 0.35,
        "dense_precision": 0.36,
        "cofitok_recall": 0.34,
        "dense_recall": 0.35,
        "ordered_rank": 1,
        "order_count": 24,
    }
    thresholds = {
        "min_samples": 50_000 if full else 10_000,
        "max_fid_regression": 0.05,
        "max_absolute_fid": 20.0 if full else 100.0,
        "max_endpoint_regression": 0.05,
        "min_precision": 0.30,
        "min_recall": 0.30,
        "max_precision_regression": 0.05,
        "max_recall_regression": 0.05,
    }

    def evidence(name: str) -> dict:
        if name == "fid_within_tolerance":
            return {
                "cofitok_fid": summary["cofitok_fid"],
                "dense_fid": summary["dense_fid"],
                "max_regression": thresholds["max_fid_regression"],
            }
        if name == "absolute_fid_quality":
            return {
                "cofitok_fid": summary["cofitok_fid"],
                "max_absolute_fid": thresholds["max_absolute_fid"],
            }
        if name == "endpoint_within_tolerance":
            return {
                "cofitok_endpoint_mse": summary["cofitok_endpoint_mse"],
                "dense_endpoint_mse": summary["dense_endpoint_mse"],
                "max_regression": thresholds["max_endpoint_regression"],
            }
        if name == "ordered_prefix_path":
            return {"rank": summary["ordered_rank"], "order_count": summary["order_count"]}
        if name == "restricted_synthesis_contract":
            return {"zero_token_max_abs": 0.0}
        if name == "shuffle_mismatch":
            return {"shuffled_to_ordered_endpoint_ratio": 1.2}
        if name == "full_precision_recall_quality":
            return {
                "enforced": True,
                **{key: thresholds[key] for key in (
                    "min_precision",
                    "min_recall",
                    "max_precision_regression",
                    "max_recall_regression",
                )},
                **{key: summary[key] for key in (
                    "cofitok_precision",
                    "dense_precision",
                    "cofitok_recall",
                    "dense_recall",
                )},
            }
        return {}

    return {
        "schema_version": 1,
        "stage": stage,
        "status": "pass",
        "decision": (
            "large_scale_generation_ready" if full else "promote_to_full_imagenet256"
        ),
        "thresholds": thresholds,
        "gates": [
            {"name": name, "passed": True, "evidence": evidence(name)}
            for name in sorted(REQUIRED_GENERATION_GATES[stage])
        ],
        "summary": summary,
    }


def test_scaling_gate_authorizes_only_the_locked_contract() -> None:
    evidence = validate_generation_gate_authorization(_gate(), expected_stage="scaling")

    assert evidence["decision"] == "promote_to_full_imagenet256"
    assert evidence["validated_thresholds"]["max_absolute_fid"] == 100.0


@pytest.mark.parametrize(
    ("threshold", "value"),
    (("min_samples", 9_999), ("max_absolute_fid", 100.01), ("max_fid_regression", 0.051)),
)
def test_scaling_gate_rejects_weakened_thresholds(threshold: str, value: float) -> None:
    gate = _gate()
    gate["thresholds"][threshold] = value

    with pytest.raises(ValueError, match="weaker"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_scaling_gate_rejects_missing_required_check() -> None:
    gate = _gate()
    gate["gates"] = [
        row for row in gate["gates"] if row["name"] != "absolute_fid_quality"
    ]

    with pytest.raises(ValueError, match="absolute_fid_quality"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_scaling_gate_recomputes_summary_thresholds() -> None:
    gate = _gate()
    gate["summary"]["cofitok_fid"] = 100.1

    with pytest.raises(ValueError, match="absolute FID"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_full_gate_rejects_weakened_precision_floor() -> None:
    gate = copy.deepcopy(_gate("full"))
    gate["thresholds"]["min_precision"] = 0.29

    with pytest.raises(ValueError, match="min_precision"):
        validate_generation_gate_authorization(gate, expected_stage="full")
