from __future__ import annotations

import math
from typing import Any


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_100k_cross_protocol_reconciliation"
INTERMEDIATE_METRIC_SCHEMA_VERSION = 1
INTERMEDIATE_METRIC_ROLE = "generation_100k_cross_protocol_subset_metrics"
REQUIRED_ROUTE = "reconcile_100k_cross_protocol_evidence"

CLAIM_BOUNDARY = {
    "report_is_promotion_gate": False,
    "generation_advantage_proven": False,
    "quality_or_generation_advantage_claim_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "gpu_execution_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
}


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is missing or non-numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def metric_summary(metrics: dict[str, Any], *, label: str) -> dict[str, float]:
    fid = _finite(metrics.get("frechet_inception_distance"), label=f"{label} FID")
    inception_mean = _finite(
        metrics.get("inception_score_mean"), label=f"{label} Inception Score"
    )
    inception_std = _finite(
        metrics.get("inception_score_std"), label=f"{label} Inception Score std"
    )
    if fid < 0.0 or inception_mean <= 0.0 or inception_std < 0.0:
        raise ValueError(f"{label} metrics are outside their domains")
    return {
        "frechet_inception_distance": fid,
        "inception_score_mean": inception_mean,
        "inception_score_std": inception_std,
    }


def lower_fid_winner(
    cofitok_fid: float,
    dense_fid: float,
    *,
    tie_tolerance: float = 1.0e-12,
) -> str:
    cofitok = _finite(cofitok_fid, label="CoFiTok FID")
    dense = _finite(dense_fid, label="dense FID")
    if abs(cofitok - dense) <= tie_tolerance:
        return "tie"
    return "cofitok" if cofitok < dense else "dense_identity"


def build_reconciliation_comparison(
    *,
    cofitok_ddim50_2048: dict[str, Any],
    dense_ddim50_2048: dict[str, Any],
    cofitok_ddim100_2048: dict[str, Any],
    dense_ddim100_2048: dict[str, Any],
    cofitok_ddim100_10000: dict[str, Any],
    dense_ddim100_10000: dict[str, Any],
) -> dict[str, Any]:
    rows = {
        "ddim50_2048": {
            "cofitok": metric_summary(
                cofitok_ddim50_2048, label="CoFiTok DDIM-50/2048"
            ),
            "dense_identity": metric_summary(
                dense_ddim50_2048, label="dense DDIM-50/2048"
            ),
        },
        "ddim100_2048": {
            "cofitok": metric_summary(
                cofitok_ddim100_2048, label="CoFiTok DDIM-100/2048"
            ),
            "dense_identity": metric_summary(
                dense_ddim100_2048, label="dense DDIM-100/2048"
            ),
        },
        "ddim100_10000": {
            "cofitok": metric_summary(
                cofitok_ddim100_10000, label="CoFiTok DDIM-100/10000"
            ),
            "dense_identity": metric_summary(
                dense_ddim100_10000, label="dense DDIM-100/10000"
            ),
        },
    }
    for row in rows.values():
        row["lower_fid_method"] = lower_fid_winner(
            row["cofitok"]["frechet_inception_distance"],
            row["dense_identity"]["frechet_inception_distance"],
        )

    winner_50 = rows["ddim50_2048"]["lower_fid_method"]
    winner_100_2048 = rows["ddim100_2048"]["lower_fid_method"]
    winner_100_10000 = rows["ddim100_10000"]["lower_fid_method"]
    if winner_100_2048 == winner_100_10000 and winner_50 != winner_100_2048:
        explanation = "sampler_step_effect_dominates_observed_ranking_reversal"
    elif winner_50 == winner_100_2048 and winner_100_2048 != winner_100_10000:
        explanation = "sample_count_effect_dominates_observed_ranking_reversal"
    elif winner_50 == winner_100_2048 == winner_100_10000:
        explanation = "no_cross_protocol_ranking_reversal_after_replay"
    else:
        explanation = "sampler_step_and_sample_count_effects_remain_entangled"

    per_method: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        fid_50_2048 = rows["ddim50_2048"][method]["frechet_inception_distance"]
        fid_100_2048 = rows["ddim100_2048"][method]["frechet_inception_distance"]
        fid_100_10000 = rows["ddim100_10000"][method][
            "frechet_inception_distance"
        ]
        per_method[method] = {
            "sampler_step_effect_at_2048": {
                "fid_delta_ddim100_minus_ddim50": fid_100_2048 - fid_50_2048,
                "ddim100_strictly_better": fid_100_2048 < fid_50_2048,
            },
            "sample_count_effect_at_ddim100": {
                "fid_delta_10000_minus_2048": fid_100_10000 - fid_100_2048,
                "10000_strictly_better": fid_100_10000 < fid_100_2048,
            },
        }

    return {
        "protocol_rows": rows,
        "per_method_effects": per_method,
        "ranking_reversal_explanation": explanation,
        "sampler_step_changes_matched_ranking": winner_50 != winner_100_2048,
        "sample_count_changes_matched_ranking": winner_100_2048 != winner_100_10000,
        "terminal_protocol_controls_quality_status": True,
        "terminal_quality_status": "hold",
        "generation_advantage_proven": False,
    }


def validate_decision_route(decision: dict[str, Any]) -> None:
    stage = decision.get("recommended_next_stage")
    boundary = decision.get("authorization_boundary")
    if (
        decision.get("status") != "completed"
        or not isinstance(stage, dict)
        or stage.get("id") != REQUIRED_ROUTE
        or stage.get("execution_ready") is not False
        or stage.get("gpu_execution_allowed") is not False
        or stage.get("full_300k_launch_allowed") is not False
        or stage.get("release_authorization_allowed") is not False
        or not isinstance(boundary, dict)
        or boundary.get("recommended_stage_execution_allowed") is not False
        or boundary.get("full_training_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("release_authorization_allowed") is not False
    ):
        raise ValueError("cross-protocol reconciliation decision route differs")


def validate_quality_hold(result: dict[str, Any]) -> None:
    screen = result.get("quality_screen")
    boundary = result.get("authorization_boundary")
    if (
        result.get("status") != "completed"
        or not isinstance(screen, dict)
        or screen.get("status") != "hold"
        or not isinstance(boundary, dict)
        or boundary.get("full_training_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("release_authorization_allowed") is not False
        or boundary.get("report_is_promotion_gate") is not False
    ):
        raise ValueError("quality bridge terminal hold boundary differs")
