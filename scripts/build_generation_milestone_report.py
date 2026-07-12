from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a matched, non-claim generation-quality milestone report."
    )
    parser.add_argument("--cofitok-generation", required=True)
    parser.add_argument("--dense-generation", required=True)
    parser.add_argument("--cofitok-checkpoint-eval", required=True)
    parser.add_argument("--dense-checkpoint-eval", required=True)
    parser.add_argument("--milestone-step", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, default=2048)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") not in {"completed", "pass"}:
        raise ValueError(f"report is incomplete: {path}")
    return payload


def _finite_metric(report: dict[str, Any], name: str) -> float:
    value = float(report["metrics"][name])
    if not math.isfinite(value):
        raise ValueError(f"generation metric {name} is not finite")
    return value


def _matched_sampling_protocol(sampling: dict[str, Any]) -> dict[str, Any]:
    ignored = {"prefix_budgets", "sample_set_digest"}
    return {key: value for key, value in sampling.items() if key not in ignored}


def _method_row(
    generation: dict[str, Any],
    checkpoint_eval: dict[str, Any],
    *,
    milestone_step: int,
    expected_samples: int,
) -> dict[str, Any]:
    provenance = generation["sample_provenance"]
    if int(generation["counts"]["generated_image_count"]) != expected_samples:
        raise ValueError("milestone generated sample count mismatch")
    if int(provenance["checkpoint_step"]) != milestone_step:
        raise ValueError("generation checkpoint step does not match milestone")
    if int(checkpoint_eval["checkpoint_step"]) != milestone_step:
        raise ValueError("mechanism checkpoint step does not match milestone")
    if provenance["checkpoint_sha256"] != checkpoint_eval["checkpoint_sha256"]:
        raise ValueError("generation and mechanism evaluation use different checkpoint bytes")
    if provenance["weights"] != "ema" or checkpoint_eval["weights"] != "ema":
        raise ValueError("milestone evaluation must use EMA weights")

    ordered = checkpoint_eval["metrics"]["orders"]["ordered"]
    return {
        "checkpoint": provenance["checkpoint"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_step": milestone_step,
        "sample_set_sha256": provenance["sample_set_sha256"],
        "selected_prefix_budget": int(provenance["selected_prefix_budget"]),
        "sample_count": expected_samples,
        "fid": _finite_metric(generation, "frechet_inception_distance"),
        "inception_score": _finite_metric(generation, "inception_score_mean"),
        "endpoint_clean_mse": float(ordered["endpoint_clean_mse"]),
        "prefix_path_mse_auc": float(ordered["prefix_path_mse_auc"]),
        "ordered_rank_by_path_auc": int(
            checkpoint_eval["metrics"]["ordered_rank_by_path_auc"]
        ),
        "order_count": int(checkpoint_eval["metrics"]["order_count"]),
        "zero_token_max_abs": float(checkpoint_eval["metrics"]["zero_token_max_abs"]),
        "shuffled_to_ordered_endpoint_ratio": float(
            checkpoint_eval["metrics"]["shuffled_to_ordered_endpoint_ratio"]
        ),
        "sampling": provenance["sampling"],
    }


def build_report(
    *,
    cofitok_generation: dict[str, Any],
    dense_generation: dict[str, Any],
    cofitok_checkpoint_eval: dict[str, Any],
    dense_checkpoint_eval: dict[str, Any],
    milestone_step: int,
    expected_samples: int,
) -> dict[str, Any]:
    if milestone_step < 1 or expected_samples < 1:
        raise ValueError("milestone-step and expected-samples must be positive")
    cofitok = _method_row(
        cofitok_generation,
        cofitok_checkpoint_eval,
        milestone_step=milestone_step,
        expected_samples=expected_samples,
    )
    dense = _method_row(
        dense_generation,
        dense_checkpoint_eval,
        milestone_step=milestone_step,
        expected_samples=expected_samples,
    )
    if _matched_sampling_protocol(cofitok["sampling"]) != _matched_sampling_protocol(
        dense["sampling"]
    ):
        raise ValueError("CoFiTok and dense milestone sampling protocols are not matched")
    if cofitok["selected_prefix_budget"] <= dense["selected_prefix_budget"]:
        raise ValueError("milestone rows do not identify K-token CoFiTok and dense control")

    fid_relative_change = (cofitok["fid"] - dense["fid"]) / max(dense["fid"], 1e-12)
    endpoint_relative_change = (
        cofitok["endpoint_clean_mse"] - dense["endpoint_clean_mse"]
    ) / max(dense["endpoint_clean_mse"], 1e-12)
    alerts = []
    if fid_relative_change > 0.25:
        alerts.append("cofitok_fid_more_than_25pct_above_dense")
    if cofitok["ordered_rank_by_path_auc"] != 1:
        alerts.append("cofitok_ordered_prefix_not_rank1")
    if cofitok["zero_token_max_abs"] != 0.0:
        alerts.append("cofitok_zero_token_contract_failed")
    if cofitok["shuffled_to_ordered_endpoint_ratio"] <= 1.0:
        alerts.append("cofitok_shuffle_mismatch_not_detected")

    return {
        "schema_version": 1,
        "status": "completed",
        "role": "training_quality_trend_only",
        "claim_policy": {
            "formal_generation_claim_allowed": False,
            "reason": "2,048-sample DDIM-50 milestones are early-warning diagnostics, not final 50K evaluation.",
        },
        "milestone_step": milestone_step,
        "expected_samples": expected_samples,
        "methods": {"cofitok": cofitok, "dense_identity": dense},
        "matched_comparison": {
            "fid_relative_change": fid_relative_change,
            "endpoint_clean_mse_relative_change": endpoint_relative_change,
        },
        "quality_alerts": alerts,
        "quality_alert": bool(alerts),
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        cofitok_generation=_read(args.cofitok_generation),
        dense_generation=_read(args.dense_generation),
        cofitok_checkpoint_eval=_read(args.cofitok_checkpoint_eval),
        dense_checkpoint_eval=_read(args.dense_checkpoint_eval),
        milestone_step=args.milestone_step,
        expected_samples=args.expected_samples,
    )
    write_json_report(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
