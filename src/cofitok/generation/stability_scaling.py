from __future__ import annotations

from copy import deepcopy
from typing import Any


ROBUST_METRIC_FIELDS = (
    "tail_two_energy_ratio",
    "max_single_token_energy_ratio",
    "endpoint_ratio",
    "validation_ratio",
    "peak_predicted_x0_high_frequency_ratio",
    "reconstruction_ratio",
    "cofitok_reconstruction_amplification",
    "dense_reconstruction_amplification",
)
AUTHORIZED_NEXT_STAGES = {
    "matched_5k": "authorize_fresh_matched_5k",
    "fresh_matched_50k_preparation": "authorize_fresh_matched_50k_preparation",
}


def _failed_gates(report: dict[str, Any]) -> list[str]:
    return sorted(
        name
        for name, gate in report.get("gates", {}).items()
        if gate.get("passed") is not True
    )


def _protocol_without_seed(protocol: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(protocol)
    normalized.get("rollout", {}).pop("seed", None)
    return normalized


def build_stability_scaling_decision(
    *,
    screening_report: dict[str, Any],
    robust_reports: list[dict[str, Any]],
    min_robust_reports: int = 2,
    min_robust_images: int = 64,
    next_stage: str = "fresh_matched_50k_preparation",
) -> dict[str, Any]:
    if next_stage not in AUTHORIZED_NEXT_STAGES:
        raise ValueError(
            "next_stage must be one of: "
            + ", ".join(sorted(AUTHORIZED_NEXT_STAGES))
        )
    issues: list[str] = []
    if int(screening_report.get("schema_version", 0)) < 2:
        issues.append("screening report schema must be at least 2")
    screening_failed = _failed_gates(screening_report)
    unexpected_screening_failures = sorted(
        set(screening_failed) - {"reconstruction_regression"}
    )
    if unexpected_screening_failures:
        issues.append(
            "screening has failures beyond reconstruction: "
            + ", ".join(unexpected_screening_failures)
        )
    if len(robust_reports) < min_robust_reports:
        issues.append(
            f"requires at least {min_robust_reports} robust reports, "
            f"received {len(robust_reports)}"
        )

    screening_identity = screening_report.get("identity")
    screening_checkpoint_protocol = {
        key: screening_report.get("protocol", {}).get(key)
        for key in (
            "weights",
            "checkpoint_step",
            "checkpoint_evaluated_images",
            "checkpoint_timestep",
            "high_frequency_timesteps",
        )
    }
    robust_rows = []
    robust_seeds: list[int] = []
    normalized_protocol: dict[str, Any] | None = None
    for index, report in enumerate(robust_reports):
        label = f"robust report {index}"
        if int(report.get("schema_version", 0)) < 2:
            issues.append(f"{label} schema must be at least 2")
        if report.get("status") != "pass" or _failed_gates(report):
            issues.append(f"{label} did not pass every qualification gate")
        if report.get("identity") != screening_identity:
            issues.append(f"{label} checkpoint identity differs from screening")
        if report.get("pair_contract", {}).get("valid") is not True:
            issues.append(f"{label} pair contract is not valid")

        protocol = report.get("protocol", {})
        checkpoint_protocol = {
            key: protocol.get(key) for key in screening_checkpoint_protocol
        }
        if checkpoint_protocol != screening_checkpoint_protocol:
            issues.append(f"{label} checkpoint evaluation protocol differs")
        rollout_protocol = protocol.get("rollout", {})
        num_images = int(rollout_protocol.get("num_images", 0))
        if num_images < min_robust_images:
            issues.append(
                f"{label} has {num_images} rollout images; "
                f"requires at least {min_robust_images}"
            )
        seed_value = rollout_protocol.get("seed")
        if not isinstance(seed_value, int):
            issues.append(f"{label} rollout seed is not an integer")
            seed = -1
        else:
            seed = seed_value
            robust_seeds.append(seed)

        current_normalized = _protocol_without_seed(protocol)
        if normalized_protocol is None:
            normalized_protocol = current_normalized
        elif current_normalized != normalized_protocol:
            issues.append(f"{label} protocol differs beyond the rollout seed")

        metrics = report.get("metrics", {})
        robust_rows.append(
            {
                "seed": seed,
                "num_images": num_images,
                **{field: float(metrics[field]) for field in ROBUST_METRIC_FIELDS},
            }
        )

    if len(set(robust_seeds)) != len(robust_seeds):
        issues.append("robust rollout seeds must be unique")

    metric_ranges = {}
    if robust_rows:
        for field in ROBUST_METRIC_FIELDS:
            values = [float(row[field]) for row in robust_rows]
            metric_ranges[field] = {
                "minimum": min(values),
                "maximum": max(values),
                "mean": sum(values) / len(values),
            }

    passed = not issues
    return {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "decision": (
            AUTHORIZED_NEXT_STAGES[next_stage]
            if passed
            else "hold_for_stability_correction"
        ),
        "authorized_next_stage": next_stage if passed else None,
        "requirements": {
            "min_robust_reports": min_robust_reports,
            "min_robust_images": min_robust_images,
            "unique_rollout_seeds": True,
            "allowed_screening_failures": ["reconstruction_regression"],
            "all_robust_gates_must_pass": True,
        },
        "screening": {
            "status": screening_report.get("status"),
            "failed_gates": screening_failed,
            "rollout_protocol": screening_report.get("protocol", {}).get("rollout"),
        },
        "identity": screening_identity,
        "robust_rows": robust_rows,
        "robust_metric_ranges": metric_ranges,
        "issues": issues,
    }
