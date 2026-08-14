from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.generation import sampling_protocol_contract
from cofitok.reporting import file_sha256, write_json_report


MILESTONE_REPORT_SCHEMA_VERSION = 2
SOURCE_REPORT_NAMES = {
    "cofitok_generation",
    "dense_generation",
    "cofitok_checkpoint_eval",
    "dense_checkpoint_eval",
}
SOURCE_PROFILES = {
    "full": {
        "cofitok": "imagenet256_full_cofitok_k8_300k",
        "dense": "imagenet256_full_dense_300k",
    },
    "stability_full": {
        "cofitok": (
            "stability_full_300k_ema_teacher/"
            "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
        ),
        "dense": (
            "stability_full_300k_ema_teacher/"
            "dense_rollout_x0_u2_ema_teacher"
        ),
    },
    "quality_bridge": {
        "cofitok": (
            "stability_full_data_100k_base128_quality_bridge_v1/"
            "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
        ),
        "dense": (
            "stability_full_data_100k_base128_quality_bridge_v1/"
            "dense_rollout_x0_u2_ema_teacher"
        ),
    },
    "capacity_scaling": {
        "cofitok": (
            "stability_full_data_100k_capacity_probe_250m_10k_v1/"
            "base256_cofitok"
        ),
        "dense": (
            "stability_full_data_100k_capacity_probe_250m_10k_v1/"
            "base256_dense_identity"
        ),
    },
}


def expected_source_report_suffixes(
    step: int,
    *,
    source_profile: str = "full",
) -> dict[str, str]:
    profile = SOURCE_PROFILES.get(source_profile)
    if profile is None:
        raise ValueError("unsupported milestone source profile")
    step_tag = f"step_{step:08d}"
    return {
        "cofitok_generation": (
            f"{profile['cofitok']}/milestones/"
            f"{step_tag}/samples_2048_ddim50_cfg15/metrics/"
            "generation_metrics_report.json"
        ),
        "dense_generation": (
            f"{profile['dense']}/milestones/"
            f"{step_tag}/samples_2048_ddim50_cfg15/metrics/"
            "generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            f"{profile['cofitok']}/milestones/"
            f"{step_tag}/checkpoint_eval/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            f"{profile['dense']}/milestones/"
            f"{step_tag}/checkpoint_eval/checkpoint_evaluation_report.json"
        ),
    }


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
    parser.add_argument(
        "--source-profile",
        choices=tuple(SOURCE_PROFILES),
        default="full",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") not in {"completed", "pass"}:
        raise ValueError(f"report is incomplete: {path}")
    return payload


def source_report_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"milestone source report does not exist: {source}")
    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _validate_source_report_identities(
    source_reports: dict[str, dict[str, Any]],
) -> None:
    if set(source_reports) != SOURCE_REPORT_NAMES:
        raise ValueError("milestone source-report set is incomplete")
    for name, identity in source_reports.items():
        path = str(identity.get("path", ""))
        sha256 = str(identity.get("sha256", ""))
        if not path or int(identity.get("bytes", 0)) < 1 or len(sha256) != 64:
            raise ValueError(f"milestone source-report identity is invalid: {name}")


def verify_milestone_source_reports(
    report: dict[str, Any],
    *,
    source_profile: str | None = None,
) -> dict[str, Any]:
    source_reports = report.get("source_reports")
    if not isinstance(source_reports, dict):
        raise ValueError("milestone report is missing source-report identities")
    _validate_source_report_identities(source_reports)
    expected_step = int(report.get("milestone_step", -1))
    if expected_step < 1:
        raise ValueError("milestone report step is invalid")
    observed_profile = str(report.get("source_profile", "full"))
    expected_profile = source_profile or observed_profile
    if observed_profile != expected_profile:
        raise ValueError("milestone source profile differs")
    for name, expected_suffix in expected_source_report_suffixes(
        expected_step,
        source_profile=expected_profile,
    ).items():
        if not str(source_reports[name]["path"]).replace("\\", "/").endswith(
            expected_suffix
        ):
            raise ValueError(f"milestone source-report path is not authoritative: {name}")
    verified = {}
    for name, expected in source_reports.items():
        actual = source_report_identity(expected["path"])
        if actual != expected:
            raise ValueError(f"milestone source report changed after binding: {name}")
        verified[name] = actual
    return {
        "status": "verified",
        "source_profile": expected_profile,
        "source_reports": verified,
    }


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
    if provenance.get("checkpoint_integrity_manifest") != checkpoint_eval.get(
        "checkpoint_integrity_manifest"
    ):
        raise ValueError("generation and mechanism evaluation use different integrity manifests")
    if provenance["weights"] != "ema" or checkpoint_eval["weights"] != "ema":
        raise ValueError("milestone evaluation must use EMA weights")
    sampling_contract = sampling_protocol_contract(
        provenance["sampling"],
        stage="milestone",
        expected_num_train_timesteps=1_000,
    )
    if sampling_contract["valid"] is not True:
        raise ValueError(
            "milestone sampling protocol is invalid: "
            + ", ".join(sampling_contract["issues"])
        )
    if provenance["sampling"].get("image_shape") != [3, 256, 256]:
        raise ValueError("milestone sampling image shape must be RGB 256x256")

    ordered = checkpoint_eval["metrics"]["orders"]["ordered"]
    row = {
        "checkpoint": provenance["checkpoint"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_integrity_manifest": provenance["checkpoint_integrity_manifest"],
        "checkpoint_step": milestone_step,
        "weights": provenance["weights"],
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
    for name in (
        "fid",
        "inception_score",
        "endpoint_clean_mse",
        "prefix_path_mse_auc",
        "zero_token_max_abs",
        "shuffled_to_ordered_endpoint_ratio",
    ):
        if not math.isfinite(float(row[name])):
            raise ValueError(f"milestone method metric is not finite: {name}")
    return row


def build_report(
    *,
    cofitok_generation: dict[str, Any],
    dense_generation: dict[str, Any],
    cofitok_checkpoint_eval: dict[str, Any],
    dense_checkpoint_eval: dict[str, Any],
    source_reports: dict[str, dict[str, Any]],
    milestone_step: int,
    expected_samples: int,
    source_profile: str = "full",
) -> dict[str, Any]:
    if milestone_step < 1 or expected_samples < 1:
        raise ValueError("milestone-step and expected-samples must be positive")
    _validate_source_report_identities(source_reports)
    if source_profile not in SOURCE_PROFILES:
        raise ValueError("unsupported milestone source profile")
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
        "schema_version": MILESTONE_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "role": "training_quality_trend_only",
        "source_profile": source_profile,
        "claim_policy": {
            "formal_generation_claim_allowed": False,
            "reason": "2,048-sample DDIM-50 milestones are early-warning diagnostics, not final 50K evaluation.",
        },
        "milestone_step": milestone_step,
        "expected_samples": expected_samples,
        "source_reports": source_reports,
        "methods": {"cofitok": cofitok, "dense_identity": dense},
        "matched_comparison": {
            "fid_relative_change": fid_relative_change,
            "endpoint_clean_mse_relative_change": endpoint_relative_change,
        },
        "quality_alerts": alerts,
        "quality_alert": bool(alerts),
    }


def validate_milestone_report(
    report: dict[str, Any],
    *,
    expected_step: int,
    source_verification: dict[str, Any] | None = None,
    expected_source_profile: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if report.get("schema_version") != MILESTONE_REPORT_SCHEMA_VERSION:
        raise ValueError("milestone report schema is unsupported")
    if report.get("status") != "completed":
        raise ValueError("milestone report is incomplete")
    if report.get("role") != "training_quality_trend_only":
        raise ValueError("milestone report role is invalid")
    claim_policy = report.get("claim_policy", {})
    if claim_policy.get("formal_generation_claim_allowed") is not False:
        raise ValueError("milestone report cannot authorize a formal generation claim")
    if int(report.get("milestone_step", -1)) != expected_step:
        raise ValueError("milestone report step mismatch")
    if int(report.get("expected_samples", -1)) != 2_048:
        raise ValueError("milestone report sample count mismatch")
    source_profile = str(report.get("source_profile", "full"))
    if source_profile not in SOURCE_PROFILES:
        raise ValueError("milestone source profile is unsupported")
    if (
        expected_source_profile is not None
        and source_profile != expected_source_profile
    ):
        raise ValueError("milestone source profile mismatch")
    source_reports = report.get("source_reports")
    if not isinstance(source_reports, dict):
        raise ValueError("milestone report is missing source-report identities")
    _validate_source_report_identities(source_reports)
    for name, expected_suffix in expected_source_report_suffixes(
        expected_step,
        source_profile=source_profile,
    ).items():
        if not str(source_reports[name]["path"]).replace("\\", "/").endswith(
            expected_suffix
        ):
            raise ValueError(f"milestone source-report path is not authoritative: {name}")
    if source_verification is not None:
        if (
            source_verification.get("status") != "verified"
            or source_verification.get("source_profile", "full") != source_profile
            or source_verification.get("source_reports") != source_reports
        ):
            raise ValueError("milestone source-report verification differs")

    methods = report.get("methods", {})
    if set(methods) != {"cofitok", "dense_identity"}:
        raise ValueError("milestone method pair is invalid")
    for method, expected_budget in (("cofitok", 8), ("dense_identity", 1)):
        row = methods[method]
        if (
            int(row.get("checkpoint_step", -1)) != expected_step
            or int(row.get("sample_count", -1)) != 2_048
            or int(row.get("selected_prefix_budget", -1)) != expected_budget
            or row.get("weights") != "ema"
        ):
            raise ValueError(f"milestone {method} checkpoint/sample identity differs")
        for field in ("checkpoint_sha256", "sample_set_sha256"):
            if len(str(row.get(field, ""))) != 64:
                raise ValueError(f"milestone {method} {field} is malformed")
        if not str(row.get("checkpoint_integrity_manifest", "")).endswith(
            ".integrity.json"
        ):
            raise ValueError(f"milestone {method} integrity manifest is malformed")
        contract = sampling_protocol_contract(
            row.get("sampling", {}),
            stage="milestone",
            expected_num_train_timesteps=1_000,
        )
        if contract["valid"] is not True:
            raise ValueError(
                f"milestone {method} sampling protocol is invalid: "
                + ", ".join(contract["issues"])
            )
        if row["sampling"].get("image_shape") != [3, 256, 256]:
            raise ValueError(f"milestone {method} image shape differs")
        for field in (
            "fid",
            "inception_score",
            "endpoint_clean_mse",
            "prefix_path_mse_auc",
            "zero_token_max_abs",
            "shuffled_to_ordered_endpoint_ratio",
        ):
            if not math.isfinite(float(row.get(field, math.nan))):
                raise ValueError(f"milestone {method} metric is not finite: {field}")
    if _matched_sampling_protocol(methods["cofitok"]["sampling"]) != (
        _matched_sampling_protocol(methods["dense_identity"]["sampling"])
    ):
        raise ValueError("milestone method sampling protocols differ")

    cofitok = methods["cofitok"]
    dense = methods["dense_identity"]
    fid_relative_change = (cofitok["fid"] - dense["fid"]) / max(
        dense["fid"], 1e-12
    )
    endpoint_relative_change = (
        cofitok["endpoint_clean_mse"] - dense["endpoint_clean_mse"]
    ) / max(dense["endpoint_clean_mse"], 1e-12)
    comparison = report.get("matched_comparison", {})
    if not math.isclose(
        float(comparison.get("fid_relative_change", math.nan)),
        fid_relative_change,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(comparison.get("endpoint_clean_mse_relative_change", math.nan)),
        endpoint_relative_change,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("milestone matched comparison was not recomputed correctly")

    expected_alerts = []
    if fid_relative_change > 0.25:
        expected_alerts.append("cofitok_fid_more_than_25pct_above_dense")
    if int(cofitok.get("ordered_rank_by_path_auc", -1)) != 1:
        expected_alerts.append("cofitok_ordered_prefix_not_rank1")
    if float(cofitok.get("zero_token_max_abs", math.nan)) != 0.0:
        expected_alerts.append("cofitok_zero_token_contract_failed")
    if float(cofitok.get("shuffled_to_ordered_endpoint_ratio", math.nan)) <= 1.0:
        expected_alerts.append("cofitok_shuffle_mismatch_not_detected")
    if report.get("quality_alerts") != expected_alerts:
        raise ValueError("milestone quality alerts differ from source metrics")
    if report.get("quality_alert") is not bool(expected_alerts):
        raise ValueError("milestone quality-alert flag differs")
    return {
        "status": "verified",
        "source_profile": source_profile,
        "quality_alerts": expected_alerts,
        "cofitok_fid": float(cofitok["fid"]),
        "dense_fid": float(dense["fid"]),
        "source_report_sha256": {
            name: identity["sha256"] for name, identity in source_reports.items()
        },
    }, [f"milestone_{expected_step}:{alert}" for alert in expected_alerts]


def main() -> None:
    args = parse_args()
    source_paths = {
        "cofitok_generation": Path(args.cofitok_generation),
        "dense_generation": Path(args.dense_generation),
        "cofitok_checkpoint_eval": Path(args.cofitok_checkpoint_eval),
        "dense_checkpoint_eval": Path(args.dense_checkpoint_eval),
    }
    report = build_report(
        cofitok_generation=_read(source_paths["cofitok_generation"]),
        dense_generation=_read(source_paths["dense_generation"]),
        cofitok_checkpoint_eval=_read(source_paths["cofitok_checkpoint_eval"]),
        dense_checkpoint_eval=_read(source_paths["dense_checkpoint_eval"]),
        source_reports={
            name: source_report_identity(path) for name, path in source_paths.items()
        },
        milestone_step=args.milestone_step,
        expected_samples=args.expected_samples,
        source_profile=args.source_profile,
    )
    write_json_report(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
