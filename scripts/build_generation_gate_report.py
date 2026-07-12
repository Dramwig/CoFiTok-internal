from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ImageNet-256 10% generation promotion gate.")
    parser.add_argument("--cofitok-training", required=True)
    parser.add_argument("--dense-training", required=True)
    parser.add_argument("--cofitok-generation", required=True)
    parser.add_argument("--dense-generation", required=True)
    parser.add_argument("--cofitok-checkpoint-eval", required=True)
    parser.add_argument("--dense-checkpoint-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["scaling", "full"], default="scaling")
    parser.add_argument("--min-samples", type=int, default=10_000)
    parser.add_argument("--max-fid-regression", type=float, default=0.05)
    parser.add_argument("--max-absolute-fid", type=float, default=100.0)
    parser.add_argument("--max-endpoint-regression", type=float, default=0.05)
    parser.add_argument("--allow-fail", action="store_true")
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _quality_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    names = (
        "frechet_inception_distance",
        "inception_score_mean",
        "inception_score_std",
        "precision",
        "recall",
    )
    values: dict[str, float | None] = {}
    for name in names:
        raw = report.get("metrics", {}).get(name)
        value = float(raw) if raw is not None else None
        values[name] = value if value is not None and math.isfinite(value) else None
    return values


def _gate(name: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _sampling_protocol(provenance: dict[str, Any]) -> dict[str, Any]:
    ignored = {"prefix_budgets"}
    return {
        key: value
        for key, value in provenance["sampling"].items()
        if key not in ignored
    }


def _training_checkpoint_integrity_matches(
    training: dict[str, Any],
    sample_provenance: dict[str, Any],
) -> bool:
    latest = training.get("latest_checkpoint", {})
    return (
        int(latest.get("step", -1)) == int(training.get("target_steps", -2))
        and int(latest.get("checkpoint_bytes", 0)) > 0
        and len(str(latest.get("checkpoint_sha256", ""))) == 64
        and latest.get("checkpoint_sha256") == sample_provenance.get("checkpoint_sha256")
        and str(latest.get("integrity_manifest", "")).endswith(".integrity.json")
    )


def build_report(
    *,
    cofitok_training: dict[str, Any],
    dense_training: dict[str, Any],
    cofitok_generation: dict[str, Any],
    dense_generation: dict[str, Any],
    cofitok_checkpoint: dict[str, Any],
    dense_checkpoint: dict[str, Any],
    min_samples: int,
    max_fid_regression: float,
    max_endpoint_regression: float,
    stage: str = "scaling",
    max_absolute_fid: float = 100.0,
) -> dict[str, Any]:
    if stage not in {"scaling", "full"}:
        raise ValueError("stage must be scaling or full")
    if not math.isfinite(max_absolute_fid) or max_absolute_fid <= 0.0:
        raise ValueError("max_absolute_fid must be finite and positive")
    cofitok_quality = _quality_metrics(cofitok_generation)
    dense_quality = _quality_metrics(dense_generation)
    cofitok_fid = cofitok_quality["frechet_inception_distance"]
    dense_fid = dense_quality["frechet_inception_distance"]
    cofitok_endpoint = float(
        cofitok_checkpoint["metrics"]["orders"]["ordered"]["endpoint_clean_mse"]
    )
    dense_endpoint = float(
        dense_checkpoint["metrics"]["orders"]["ordered"]["endpoint_clean_mse"]
    )
    cofitok_checkpoint_sha = str(cofitok_checkpoint["checkpoint_sha256"])
    dense_checkpoint_sha = str(dense_checkpoint["checkpoint_sha256"])
    generated_counts = (
        int(cofitok_generation["counts"]["generated_image_count"]),
        int(dense_generation["counts"]["generated_image_count"]),
    )
    evaluator_pair = (
        cofitok_generation["implementation"],
        dense_generation["implementation"],
    )
    cofitok_provenance = cofitok_generation["sample_provenance"]
    dense_provenance = dense_generation["sample_provenance"]
    cofitok_sampling = _sampling_protocol(cofitok_provenance)
    dense_sampling = _sampling_protocol(dense_provenance)
    cofitok_model_config = cofitok_training["config"]["model"]
    dense_model_config = dense_training["config"]["model"]
    cofitok_expected_shape = [
        int(cofitok_model_config["image_channels"]),
        int(cofitok_model_config["image_size"]),
        int(cofitok_model_config["image_size"]),
    ]
    dense_expected_shape = [
        int(dense_model_config["image_channels"]),
        int(dense_model_config["image_size"]),
        int(dense_model_config["image_size"]),
    ]
    parameter_gap = (
        int(cofitok_training["parameter_count"]) - int(dense_training["parameter_count"])
    ) / int(dense_training["parameter_count"])
    matched_sections = all(
        cofitok_training["config"][section] == dense_training["config"][section]
        for section in ("data", "diffusion", "runtime", "optimization")
    )
    gates = [
        _gate(
            "training_complete",
            all(
                report.get("training_complete") is True
                and report.get("completed_steps") == report.get("target_steps")
                and report.get("git", {}).get("dirty") is False
                for report in (cofitok_training, dense_training)
            ),
            {
                "cofitok_steps": cofitok_training.get("completed_steps"),
                "dense_steps": dense_training.get("completed_steps"),
                "cofitok_revision": cofitok_training.get("git", {}).get("revision"),
                "dense_revision": dense_training.get("git", {}).get("revision"),
            },
        ),
        _gate(
            "matched_training_protocol",
            matched_sections and abs(parameter_gap) <= 0.02,
            {"matched_sections": matched_sections, "relative_parameter_gap": parameter_gap},
        ),
        _gate(
            "matched_generation_protocol",
            generated_counts[0] == generated_counts[1]
            and min(generated_counts) >= min_samples
            and evaluator_pair[0] == evaluator_pair[1]
            and cofitok_generation["parameters"] == dense_generation["parameters"],
            {
                "generated_counts": list(generated_counts),
                "cofitok_evaluator": evaluator_pair[0],
                "dense_evaluator": evaluator_pair[1],
            },
        ),
        _gate(
            "generation_metrics_complete",
            all(value is not None for value in cofitok_quality.values())
            and all(value is not None for value in dense_quality.values()),
            {"cofitok": cofitok_quality, "dense": dense_quality},
        ),
        _gate(
            "matched_sampling_provenance",
            cofitok_sampling == dense_sampling
            and cofitok_provenance["weights"] == dense_provenance["weights"] == "ema"
            and int(cofitok_provenance["checkpoint_step"])
            == int(cofitok_training["target_steps"])
            and int(dense_provenance["checkpoint_step"])
            == int(dense_training["target_steps"])
            and int(cofitok_provenance["selected_prefix_budget"])
            == int(cofitok_training["config"]["model"]["token_count"])
            and int(dense_provenance["selected_prefix_budget"])
            == int(dense_training["config"]["model"]["token_count"])
            and len(str(cofitok_provenance["checkpoint_sha256"])) == 64
            and len(str(dense_provenance["checkpoint_sha256"])) == 64
            and len(str(cofitok_provenance.get("sample_set_sha256", ""))) == 64
            and len(str(dense_provenance.get("sample_set_sha256", ""))) == 64
            and cofitok_sampling.get("random_stream", {}).get("prefix_budgets_share_stream")
            is True
            and cofitok_sampling.get("random_stream", {}).get("batch_size_invariant") is True
            and cofitok_sampling.get("image_shape") == cofitok_expected_shape
            and dense_sampling.get("image_shape") == dense_expected_shape,
            {
                "protocols_match": cofitok_sampling == dense_sampling,
                "cofitok_checkpoint_step": cofitok_provenance["checkpoint_step"],
                "dense_checkpoint_step": dense_provenance["checkpoint_step"],
                "cofitok_checkpoint_sha256": cofitok_provenance["checkpoint_sha256"],
                "dense_checkpoint_sha256": dense_provenance["checkpoint_sha256"],
                "cofitok_prefix_budget": cofitok_provenance["selected_prefix_budget"],
                "dense_prefix_budget": dense_provenance["selected_prefix_budget"],
                "cofitok_image_shape": cofitok_sampling.get("image_shape"),
                "dense_image_shape": dense_sampling.get("image_shape"),
                "cofitok_sample_set_sha256": cofitok_provenance.get("sample_set_sha256"),
                "dense_sample_set_sha256": dense_provenance.get("sample_set_sha256"),
            },
        ),
        _gate(
            "checkpoint_evaluation_provenance",
            int(cofitok_checkpoint["checkpoint_step"])
            == int(cofitok_provenance["checkpoint_step"])
            and int(dense_checkpoint["checkpoint_step"])
            == int(dense_provenance["checkpoint_step"])
            and cofitok_checkpoint_sha == cofitok_provenance["checkpoint_sha256"]
            and dense_checkpoint_sha == dense_provenance["checkpoint_sha256"],
            {
                "cofitok_hash_matches": cofitok_checkpoint_sha
                == cofitok_provenance["checkpoint_sha256"],
                "dense_hash_matches": dense_checkpoint_sha
                == dense_provenance["checkpoint_sha256"],
                "cofitok_step": cofitok_checkpoint["checkpoint_step"],
                "dense_step": dense_checkpoint["checkpoint_step"],
            },
        ),
        _gate(
            "full_training_checkpoint_integrity",
            stage != "full"
            or (
                _training_checkpoint_integrity_matches(cofitok_training, cofitok_provenance)
                and _training_checkpoint_integrity_matches(dense_training, dense_provenance)
            ),
            {
                "enforced": stage == "full",
                "cofitok_latest": cofitok_training.get("latest_checkpoint"),
                "dense_latest": dense_training.get("latest_checkpoint"),
            },
        ),
        _gate(
            "fid_within_tolerance",
            cofitok_fid is not None
            and dense_fid is not None
            and dense_fid > 0.0
            and cofitok_fid <= dense_fid * (1.0 + max_fid_regression),
            {
                "cofitok_fid": cofitok_fid,
                "dense_fid": dense_fid,
                "relative_change": (
                    cofitok_fid / dense_fid - 1.0
                    if cofitok_fid is not None and dense_fid is not None and dense_fid > 0.0
                    else None
                ),
                "max_regression": max_fid_regression,
            },
        ),
        _gate(
            "absolute_fid_quality",
            cofitok_fid is not None and cofitok_fid <= max_absolute_fid,
            {"cofitok_fid": cofitok_fid, "max_absolute_fid": max_absolute_fid},
        ),
        _gate(
            "endpoint_within_tolerance",
            cofitok_endpoint <= dense_endpoint * (1.0 + max_endpoint_regression),
            {
                "cofitok_endpoint_mse": cofitok_endpoint,
                "dense_endpoint_mse": dense_endpoint,
                "relative_change": cofitok_endpoint / dense_endpoint - 1.0,
                "max_regression": max_endpoint_regression,
            },
        ),
        _gate(
            "ordered_prefix_path",
            int(cofitok_checkpoint["metrics"]["ordered_rank_by_path_auc"]) == 1
            and int(cofitok_checkpoint["metrics"]["order_count"]) >= 10,
            {
                "rank": cofitok_checkpoint["metrics"]["ordered_rank_by_path_auc"],
                "order_count": cofitok_checkpoint["metrics"]["order_count"],
            },
        ),
        _gate(
            "restricted_synthesis_contract",
            float(cofitok_checkpoint["metrics"]["zero_token_max_abs"]) == 0.0,
            {"zero_token_max_abs": cofitok_checkpoint["metrics"]["zero_token_max_abs"]},
        ),
        _gate(
            "shuffle_mismatch",
            float(cofitok_checkpoint["metrics"]["shuffled_to_ordered_endpoint_ratio"]) > 1.0,
            {
                "shuffled_to_ordered_endpoint_ratio": cofitok_checkpoint["metrics"][
                    "shuffled_to_ordered_endpoint_ratio"
                ]
            },
        ),
    ]
    passed = all(gate["passed"] for gate in gates)
    pass_decision = (
        "promote_to_full_imagenet256"
        if stage == "scaling"
        else "large_scale_generation_ready"
    )
    return {
        "schema_version": 1,
        "stage": stage,
        "status": "pass" if passed else "fail",
        "decision": pass_decision if passed else "hold",
        "thresholds": {
            "min_samples": min_samples,
            "max_fid_regression": max_fid_regression,
            "max_absolute_fid": max_absolute_fid,
            "max_endpoint_regression": max_endpoint_regression,
        },
        "gates": gates,
        "summary": {
            "cofitok_fid": cofitok_fid,
            "dense_fid": dense_fid,
            "cofitok_endpoint_mse": cofitok_endpoint,
            "dense_endpoint_mse": dense_endpoint,
            "cofitok_inception_score": cofitok_quality["inception_score_mean"],
            "dense_inception_score": dense_quality["inception_score_mean"],
            "cofitok_precision": cofitok_quality["precision"],
            "dense_precision": dense_quality["precision"],
            "cofitok_recall": cofitok_quality["recall"],
            "dense_recall": dense_quality["recall"],
            "ordered_rank": cofitok_checkpoint["metrics"]["ordered_rank_by_path_auc"],
            "order_count": cofitok_checkpoint["metrics"]["order_count"],
        },
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        cofitok_training=_read(args.cofitok_training),
        dense_training=_read(args.dense_training),
        cofitok_generation=_read(args.cofitok_generation),
        dense_generation=_read(args.dense_generation),
        cofitok_checkpoint=_read(args.cofitok_checkpoint_eval),
        dense_checkpoint=_read(args.dense_checkpoint_eval),
        min_samples=args.min_samples,
        max_fid_regression=args.max_fid_regression,
        max_endpoint_regression=args.max_endpoint_regression,
        stage=args.stage,
        max_absolute_fid=args.max_absolute_fid,
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")
    if report["status"] != "pass" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
