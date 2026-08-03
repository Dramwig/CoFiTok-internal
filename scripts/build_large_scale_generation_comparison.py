from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation import sampling_protocol_contract
from cofitok.generation_class_fidelity import (
    validate_class_fidelity_qualification,
)
from cofitok.generation_cost import training_cost_summary
from cofitok.gpu_contention import validate_gpu_contention_evidence
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA
from cofitok.reporting import file_sha256, write_json_report, write_text_report


EXTERNAL_ALIASES = {"d_ar", "mar", "retok"}
EXTERNAL_METHODS = {"d_ar": "D-AR", "mar": "MAR", "retok": "ReTok"}
COMPARISON_REPORT_SCHEMA_VERSION = 8
MATCHED_TRAINING_BUDGET_BASIS = "matched_steps_and_training_images"
MATCHED_TRAINING_PROTOCOL_NOTE = (
    "Matched dataset, resolution, shared backbone contract, optimizer schedule, "
    "effective batch, steps, images seen, and evaluator; wall-clock, GPU-hours, "
    "and FLOPs are not equalized budgets."
)
SOURCE_REPORT_PROFILES = {
    "full": {
        "cofitok_training": (
            "imagenet256_full_cofitok_k8_300k/training_report.json"
        ),
        "dense_training": "imagenet256_full_dense_300k/training_report.json",
        "cofitok_generation": (
            "imagenet256_full_cofitok_k8_300k/samples_50k_ddim250_cfg15/metrics/"
            "generation_metrics_report.json"
        ),
        "dense_generation": (
            "imagenet256_full_dense_300k/samples_50k_ddim250_cfg15/metrics/"
            "generation_metrics_report.json"
        ),
        "final_gate": (
            "artifacts/reports/generation/imagenet256_full_matched_300k/"
            "final_generation_gate.json"
        ),
        "training_contention": "generation_full_matched_300k_monitor.json",
    },
    "stability_full": {
        "cofitok_training": (
            "stability_full_300k_ema_teacher/"
            "cofitok_rgbtail3_rollout_x0_u2_ema_teacher/training_report.json"
        ),
        "dense_training": (
            "stability_full_300k_ema_teacher/"
            "dense_rollout_x0_u2_ema_teacher/training_report.json"
        ),
        "cofitok_generation": (
            "stability_full_300k_ema_teacher/"
            "cofitok_rgbtail3_rollout_x0_u2_ema_teacher/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_generation": (
            "stability_full_300k_ema_teacher/"
            "dense_rollout_x0_u2_ema_teacher/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "final_gate": (
            "stability_full_300k_ema_teacher/reports/final_generation_gate.json"
        ),
        "training_contention": (
            "stability_full_300k_ema_teacher/pair_monitor.json"
        ),
        "class_fidelity_qualification": (
            "stability_full_300k_ema_teacher/reports/class_fidelity/"
            "qualification_report.json"
        ),
    },
}


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_report_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"comparison source report does not exist: {source}")
    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _validate_source_report_identities(
    source_reports: dict[str, dict[str, Any]],
    *,
    source_profile: str = "full",
) -> None:
    expected = SOURCE_REPORT_PROFILES.get(source_profile)
    if expected is None or set(source_reports) != set(expected):
        raise ValueError("comparison source-report set is incomplete")
    for name, expected_suffix in expected.items():
        identity = source_reports[name]
        path = str(identity.get("path", "")).replace("\\", "/")
        sha256 = str(identity.get("sha256", ""))
        if (
            not path.endswith(expected_suffix)
            or int(identity.get("bytes", 0)) < 1
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"comparison source-report identity is invalid: {name}")


def verify_comparison_source_reports(report: dict[str, Any]) -> dict[str, Any]:
    source_reports = report.get("source_reports")
    if not isinstance(source_reports, dict):
        raise ValueError("comparison report is missing source-report identities")
    source_profile = str(report.get("source_profile", "full"))
    _validate_source_report_identities(
        source_reports,
        source_profile=source_profile,
    )
    verified = {}
    for name, expected in source_reports.items():
        actual = source_report_identity(expected["path"])
        if actual != expected:
            raise ValueError(f"comparison source report changed after binding: {name}")
        verified[name] = actual
    return {
        "status": "verified",
        "source_profile": source_profile,
        "source_reports": verified,
    }


def _finite_metric(report: dict[str, Any], key: str) -> float:
    value = float(report["metrics"][key])
    if not math.isfinite(value):
        raise ValueError(f"generation metric {key} is not finite")
    return value


def _distribution_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "fid": _finite_metric(report, "frechet_inception_distance"),
        "inception_score": _finite_metric(report, "inception_score_mean"),
        "precision": _finite_metric(report, "precision"),
        "recall": _finite_metric(report, "recall"),
    }
    if (
        metrics["fid"] < 0.0
        or metrics["inception_score"] <= 0.0
        or not 0.0 <= metrics["precision"] <= 1.0
        or not 0.0 <= metrics["recall"] <= 1.0
    ):
        raise ValueError("generation distribution metrics are out of range")
    return metrics


def _matched_row(
    method: str,
    training: dict[str, Any],
    generation: dict[str, Any],
    training_contention: dict[str, Any],
    class_fidelity_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = generation["sample_provenance"]
    sampling = provenance["sampling"]
    sampling_progress = provenance["sampling_progress"]
    real_set = generation.get("real_set", {})
    real_set_sha = str(real_set.get("sha256", ""))
    evaluator_environment_sha = str(
        generation.get("runtime_environment_sha256", "")
    )
    if (
        real_set.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
        or len(real_set_sha) != 64
        or any(character not in "0123456789abcdef" for character in real_set_sha)
        or int(real_set.get("image_count", -1)) < 1
    ):
        raise ValueError(f"{method} real-set provenance is invalid")
    if len(evaluator_environment_sha) != 64:
        raise ValueError(f"{method} evaluator environment SHA256 is malformed")
    training_cost = training_cost_summary(training)
    if training_cost["valid"] is not True:
        raise ValueError(f"{method} training cost accounting is invalid")
    sampling_elapsed_seconds = float(sampling_progress["cumulative_elapsed_seconds"])
    sample_count = int(generation["counts"]["generated_image_count"])
    if not math.isfinite(sampling_elapsed_seconds) or sampling_elapsed_seconds <= 0.0:
        raise ValueError(f"{method} sampling elapsed time is invalid")
    if sampling_progress.get("status") != "completed":
        raise ValueError(f"{method} sampling progress is incomplete")
    if int(sampling_progress.get("completed_samples", -1)) != sample_count:
        raise ValueError(f"{method} sampling progress count does not match metrics")
    distribution_metrics = _distribution_metrics(generation)
    class_metrics = class_fidelity_metrics or {}
    return {
        "method": method,
        "comparison_tier": "matched_training_direct",
        "directly_comparable_to_cofitok": True,
        "dataset": training["config"]["data"]["dataset"],
        "resolution": int(training["config"]["model"]["image_size"]),
        "training_steps": int(training["target_steps"]),
        "parameter_count": int(training["parameter_count"]),
        "effective_batch_size": training_cost["effective_batch_size"],
        "training_images_seen": training_cost["samples_seen"],
        "training_elapsed_seconds": training_cost["elapsed_seconds"],
        "training_images_per_second": training_cost["images_per_second"],
        "training_time_measurement": training_contention["measurement"],
        "training_budget_basis": MATCHED_TRAINING_BUDGET_BASIS,
        "compute_matched_claim_allowed": False,
        "training_cost_fields_role": "measured_outcomes",
        "training_wall_clock_directly_comparable": training_contention[
            "direct_comparison_allowed"
        ],
        "training_throughput_directly_comparable": training_contention[
            "direct_comparison_allowed"
        ],
        "training_wall_clock_comparison_reason": training_contention["reason"],
        "peak_vram_bytes": training_cost["peak_vram_bytes"],
        "sample_count": sample_count,
        "sample_batch_size": int(sampling["batch_size"]),
        "sampling_elapsed_seconds": sampling_elapsed_seconds,
        "sampling_images_per_second": sample_count / sampling_elapsed_seconds,
        "sampling_invocations": int(sampling_progress["invocation"]),
        **distribution_metrics,
        "evaluator": generation["implementation"],
        "weights": provenance["weights"],
        "sampling_protocol_schema": sampling["protocol_schema"],
        "sampling_inference_api": sampling["inference_api"],
        "sampler": sampling["sampler"],
        "num_train_timesteps": int(sampling["num_train_timesteps"]),
        "sample_steps": int(sampling["sample_steps"]),
        "actual_timesteps": sampling["actual_timesteps"],
        "guidance_scale": float(sampling["guidance_scale"]),
        "guidance_rescale": float(sampling["guidance_rescale"]),
        "cfg_batch_mode": sampling["cfg_batch_mode"],
        "eta": float(sampling["eta"]),
        "clip_x0": sampling["clip_x0"],
        "sampling_precision": sampling["precision"],
        "sampling_seed": int(sampling["seed"]),
        "sampling_start_index": int(sampling["start_index"]),
        "class_schedule": sampling["class_schedule"],
        "prefix_budgets": sampling["prefix_budgets"],
        "sampling_random_stream": sampling["random_stream"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "sample_set_sha256": provenance["sample_set_sha256"],
        "real_set_digest_schema": real_set.get("digest_schema"),
        "real_set_sha256": real_set_sha,
        "real_image_count": int(real_set.get("image_count", -1)),
        "evaluator_runtime_environment_sha256": evaluator_environment_sha,
        "class_fidelity_sample_count": class_metrics.get("sample_count"),
        "class_top1_accuracy": class_metrics.get("top1_accuracy"),
        "class_top5_accuracy": class_metrics.get("top5_accuracy"),
        "class_mean_target_probability": class_metrics.get(
            "mean_target_probability"
        ),
        "class_target_negative_log_likelihood": class_metrics.get(
            "target_negative_log_likelihood"
        ),
        "class_predicted_class_fraction": class_metrics.get(
            "predicted_class_fraction"
        ),
        "class_normalized_predicted_entropy": class_metrics.get(
            "normalized_predicted_class_entropy"
        ),
        "protocol_note": MATCHED_TRAINING_PROTOCOL_NOTE,
    }


def training_budget_policy(
    matched_rows: list[dict[str, Any]],
    training_contention: dict[str, Any],
) -> dict[str, Any]:
    if len(matched_rows) != 2:
        raise ValueError("training budget policy requires exactly two matched rows")
    matched_fields = {
        "dataset": "dataset",
        "resolution": "resolution",
        "effective_batch_size": "effective_batch_size",
        "optimizer_steps": "training_steps",
        "training_images_seen": "training_images_seen",
    }
    matched_values: dict[str, Any] = {}
    for policy_field, row_field in matched_fields.items():
        values = [row.get(row_field) for row in matched_rows]
        if values[0] != values[1]:
            raise ValueError(
                f"matched training rows differ on budget axis: {policy_field}"
            )
        matched_values[policy_field] = values[0]
    return {
        "basis": MATCHED_TRAINING_BUDGET_BASIS,
        "matched_axis_values": matched_values,
        "equal_wall_clock_budget": False,
        "equal_gpu_hours_budget": False,
        "equal_training_flops_budget": False,
        "compute_matched_claim_allowed": False,
        "direct_quality_comparison_allowed": True,
        "training_cost_fields_role": "measured_outcomes",
        "cost_efficiency_ranking_allowed": training_contention[
            "direct_comparison_allowed"
        ],
        "cost_efficiency_ranking_reason": training_contention["reason"],
    }


def _external_rows(official: dict[str, Any]) -> list[dict[str, Any]]:
    if official.get("schema_version") != 1:
        raise ValueError("unsupported official related-method table schema")
    rows = official.get("rows", [])
    aliases = {str(row.get("alias")) for row in rows}
    if aliases != EXTERNAL_ALIASES:
        raise ValueError(f"official related-method aliases do not match: {sorted(aliases)}")
    output = []
    for row in rows:
        alias = str(row.get("alias"))
        if row.get("method") != EXTERNAL_METHODS[alias]:
            raise ValueError(f"official baseline method identity mismatch: {alias}")
        if row.get("status") != "completed_eval_only_50k":
            raise ValueError(f"official baseline is incomplete: {row.get('alias')}")
        if row.get("paper_table_role") != "secondary related-method only":
            raise ValueError(f"official baseline has an unsafe table role: {row.get('alias')}")
        if row.get("dataset") != "imagenet_256" or int(row.get("resolution", 0)) != 256:
            raise ValueError(f"official baseline dataset mismatch: {row.get('alias')}")
        if int(row.get("sample_count", 0)) != 50_000:
            raise ValueError(f"official baseline sample count mismatch: {row.get('alias')}")
        metrics = {
            name: float(row[name])
            for name in ("fid", "inception_score", "precision", "recall")
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"official baseline metrics are non-finite: {row.get('alias')}")
        if (
            metrics["fid"] < 0.0
            or metrics["inception_score"] <= 0.0
            or not 0.0 <= metrics["precision"] <= 1.0
            or not 0.0 <= metrics["recall"] <= 1.0
        ):
            raise ValueError(f"official baseline metrics are out of range: {row.get('alias')}")
        metrics_txt = str(row.get("metrics_txt", ""))
        if not PurePosixPath(metrics_txt).is_absolute() or not metrics_txt.endswith(".txt"):
            raise ValueError(f"official baseline metrics source is invalid: {row.get('alias')}")
        output.append(
            {
                "alias": alias,
                "method": row["method"],
                "comparison_tier": "official_pretrained_contextual",
                "directly_comparable_to_cofitok": False,
                "dataset": row["dataset"],
                "resolution": int(row["resolution"]),
                "training_steps": None,
                "parameter_count": None,
                "sample_count": int(row["sample_count"]),
                **metrics,
                "evaluator": {
                    "package": "ADM TensorFlow evaluation graph",
                    "version": "pinned baseline protocol",
                },
                "sample_steps": None,
                "guidance_scale": None,
                "checkpoint_sha256": None,
                "sample_set_sha256": None,
                "protocol_note": row["protocol"],
                "source_metrics": metrics_txt,
                "source_npz": row.get("npz"),
                "source_kind": row.get("source_kind"),
                "source_status": row["status"],
                "paper_table_role": row["paper_table_role"],
            }
        )
    return output


def _gate_evidence(final_gate: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [gate for gate in final_gate.get("gates", []) if gate.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"final gate is missing unique evidence for {name}")
    if matches[0].get("passed") is not True:
        raise ValueError(f"final gate evidence did not pass for {name}")
    return matches[0].get("evidence", {})


def build_report(
    *,
    cofitok_training: dict[str, Any],
    dense_training: dict[str, Any],
    cofitok_generation: dict[str, Any],
    dense_generation: dict[str, Any],
    final_gate: dict[str, Any],
    official_related: dict[str, Any],
    training_contention: dict[str, Any],
    class_fidelity_qualification: dict[str, Any] | None = None,
    official_source_path: str,
    official_source_sha256: str,
    source_reports: dict[str, dict[str, Any]],
    source_profile: str = "full",
) -> dict[str, Any]:
    if len(official_source_sha256) != 64:
        raise ValueError("official related-method source SHA256 is malformed")
    if final_gate.get("stage") != "full":
        raise ValueError("large-scale comparison requires a full-stage gate report")
    _validate_source_report_identities(
        source_reports,
        source_profile=source_profile,
    )
    contention = validate_gpu_contention_evidence(training_contention)
    class_fidelity = None
    if class_fidelity_qualification is not None:
        provenance_contract = final_gate.get("provenance_contract")
        provenance_contract = (
            provenance_contract if isinstance(provenance_contract, dict) else {}
        )
        class_fidelity = validate_class_fidelity_qualification(
            class_fidelity_qualification,
            expected_stage="full",
            expected_revision=provenance_contract.get("evaluation_revision"),
            expected_branch=provenance_contract.get("evaluation_branch"),
            require_pass=False,
        )
        matches = [
            row
            for row in final_gate.get("gates", [])
            if row.get("name") == "class_conditional_fidelity"
        ]
        if (
            len(matches) != 1
            or matches[0].get("passed") is not class_fidelity["valid"]
            or matches[0].get("evidence", {}).get("qualification")
            != class_fidelity_qualification
        ):
            raise ValueError(
                "final gate class-fidelity evidence does not match its source"
            )
    elif source_profile == "stability_full":
        raise ValueError(
            "stability-full comparison requires class-fidelity qualification"
        )
    formal_contracts = {
        "cofitok": sampling_protocol_contract(
            cofitok_generation["sample_provenance"]["sampling"],
            stage="full",
            expected_num_train_timesteps=int(
                cofitok_training["config"]["diffusion"]["num_train_timesteps"]
            ),
        ),
        "dense_identity": sampling_protocol_contract(
            dense_generation["sample_provenance"]["sampling"],
            stage="full",
            expected_num_train_timesteps=int(
                dense_training["config"]["diffusion"]["num_train_timesteps"]
            ),
        ),
    }
    for method, contract in formal_contracts.items():
        if contract["valid"] is not True:
            raise ValueError(
                f"{method} formal sampling protocol is invalid: "
                + ", ".join(contract["issues"])
            )
    matched = [
        _matched_row(
            "CoFiTok K=8",
            cofitok_training,
            cofitok_generation,
            contention,
            (
                class_fidelity["metrics"]["cofitok"]
                if class_fidelity is not None
                else None
            ),
        ),
        _matched_row(
            "Dense identity",
            dense_training,
            dense_generation,
            contention,
            (
                class_fidelity["metrics"]["dense_identity"]
                if class_fidelity is not None
                else None
            ),
        ),
    ]
    if matched[0]["evaluator"] != matched[1]["evaluator"]:
        raise ValueError("matched methods used different evaluator implementations")
    if matched[0]["sample_count"] != matched[1]["sample_count"]:
        raise ValueError("matched methods used different sample counts")
    if matched[0]["sample_count"] != 50_000:
        raise ValueError("matched methods did not use the formal 50K sample count")
    if any(row["weights"] != "ema" for row in matched):
        raise ValueError("matched methods did not use EMA weights")
    if (
        matched[0]["real_set_digest_schema"] != matched[1]["real_set_digest_schema"]
        or matched[0]["real_set_sha256"] != matched[1]["real_set_sha256"]
        or matched[0]["real_image_count"] != matched[1]["real_image_count"]
    ):
        raise ValueError("matched methods used different real sets")
    if (
        matched[0]["evaluator_runtime_environment_sha256"]
        != matched[1]["evaluator_runtime_environment_sha256"]
    ):
        raise ValueError("matched methods used different evaluator environments")
    if matched[1]["fid"] <= 0.0:
        raise ValueError("dense FID must be positive")
    gate_summary = final_gate.get("summary", {})
    for key, row in (("cofitok_fid", matched[0]), ("dense_fid", matched[1])):
        if not math.isclose(
            float(gate_summary.get(key, math.nan)),
            row["fid"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"final gate {key} does not match generation metrics")
    sampling_evidence = _gate_evidence(final_gate, "matched_sampling_provenance")
    formal_protocol_evidence = _gate_evidence(final_gate, "formal_sampling_protocol")
    if (
        formal_protocol_evidence.get("stage") != "full"
        or formal_protocol_evidence.get("cofitok") != formal_contracts["cofitok"]
        or formal_protocol_evidence.get("dense_identity")
        != formal_contracts["dense_identity"]
    ):
        raise ValueError("final gate formal sampling protocol does not match sources")
    expected_provenance = (
        (
            "cofitok",
            matched[0],
        ),
        (
            "dense",
            matched[1],
        ),
    )
    for prefix, row in expected_provenance:
        if sampling_evidence.get(f"{prefix}_checkpoint_sha256") != row["checkpoint_sha256"]:
            raise ValueError(f"final gate {prefix} checkpoint hash does not match")
        if sampling_evidence.get(f"{prefix}_sample_set_sha256") != row["sample_set_sha256"]:
            raise ValueError(f"final gate {prefix} sample-set hash does not match")
    if class_fidelity is not None:
        class_contract = class_fidelity["sampling_contract"]
        if (
            class_contract["cofitok_checkpoint_sha256"]
            != matched[0]["checkpoint_sha256"]
            or class_contract["dense_checkpoint_sha256"]
            != matched[1]["checkpoint_sha256"]
            or class_contract["cofitok_sample_set_sha256"]
            != matched[0]["sample_set_sha256"]
            or class_contract["dense_sample_set_sha256"]
            != matched[1]["sample_set_sha256"]
        ):
            raise ValueError(
                "class-fidelity qualification does not match comparison samples"
            )
    real_set_evidence = _gate_evidence(final_gate, "matched_real_set_provenance")
    for method, row in (("cofitok", matched[0]), ("dense_identity", matched[1])):
        gate_real_set = real_set_evidence.get(method, {})
        if (
            gate_real_set.get("digest_schema") != row["real_set_digest_schema"]
            or gate_real_set.get("sha256") != row["real_set_sha256"]
            or int(gate_real_set.get("image_count", -1)) != row["real_image_count"]
        ):
            raise ValueError(f"final gate {method} real-set hash does not match")
    external = _external_rows(official_related)
    budget_policy = training_budget_policy(matched, contention)
    ready = (
        final_gate.get("status") == "pass"
        and final_gate.get("decision") == "large_scale_generation_ready"
        and (
            source_profile != "stability_full"
            or (
                class_fidelity is not None
                and class_fidelity["valid"] is True
            )
        )
    )
    return {
        "schema_version": COMPARISON_REPORT_SCHEMA_VERSION,
        "status": "ready" if ready else "hold",
        "source_profile": source_profile,
        "final_gate": {
            "status": final_gate.get("status"),
            "decision": final_gate.get("decision"),
        },
        "comparison_policy": {
            "primary_direct_tier": "matched_training_direct",
            "external_context_tier": "official_pretrained_contextual",
            "cross_tier_numeric_ranking_allowed": False,
            "training_wall_clock_direct_comparison_allowed": contention[
                "direct_comparison_allowed"
            ],
            "training_wall_clock_comparison_reason": contention["reason"],
            "reason": (
                "External rows use official pretrained checkpoints and the ADM TensorFlow "
                "evaluator; CoFiTok and dense use matched training plus torch-fidelity."
            ),
        },
        "training_budget_policy": budget_policy,
        "official_context_source": {
            "path": official_source_path,
            "sha256": official_source_sha256,
            "schema_version": official_related.get("schema_version"),
        },
        "source_reports": source_reports,
        "training_contention": contention,
        "class_fidelity": (
            {
                "status": class_fidelity["status"],
                "valid": class_fidelity["valid"],
                "classifier": class_fidelity["classifier"],
                "thresholds": class_fidelity["thresholds"],
                "sampling_contract": class_fidelity["sampling_contract"],
                "claim_boundary": class_fidelity["claim_boundary"],
            }
            if class_fidelity is not None
            else None
        ),
        "matched_training_rows": matched,
        "official_context_rows": external,
        "matched_summary": {
            "cofitok_minus_dense_fid": matched[0]["fid"] - matched[1]["fid"],
            "cofitok_relative_fid": matched[0]["fid"] / matched[1]["fid"] - 1.0,
            "cofitok_minus_dense_class_top1": (
                matched[0]["class_top1_accuracy"]
                - matched[1]["class_top1_accuracy"]
                if class_fidelity is not None
                else None
            ),
            "cofitok_minus_dense_class_top5": (
                matched[0]["class_top5_accuracy"]
                - matched[1]["class_top5_accuracy"]
                if class_fidelity is not None
                else None
            ),
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Large-Scale Generation Comparison",
        "",
        f"Readiness: `{report['status']}`.",
        "",
        "## Matched steps/images training (direct quality comparison)",
        "",
        "| method | params | steps | eff. batch | train images | train h (raw) | train img/s (raw) | VRAM GiB | samples | sample batch | sample h | sample img/s | FID | IS | precision | recall | class top-1 | class top-5 | class coverage | class entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["matched_training_rows"]:
        lines.append(
            "| {method} | {params} | {steps} | {batch} | {train_images} | {hours} | {throughput} | {vram} | {samples} | {sample_batch} | {sample_hours} | {sample_throughput} | {fid} | {iscore} | {precision} | {recall} | {class_top1} | {class_top5} | {class_coverage} | {class_entropy} |".format(
                method=row["method"],
                params=row["parameter_count"],
                steps=row["training_steps"],
                batch=row["effective_batch_size"],
                train_images=row["training_images_seen"],
                hours=_fmt(row["training_elapsed_seconds"] / 3600.0),
                throughput=_fmt(row["training_images_per_second"]),
                vram=_fmt(row["peak_vram_bytes"] / (1024**3)),
                samples=row["sample_count"],
                sample_batch=row["sample_batch_size"],
                sample_hours=_fmt(row["sampling_elapsed_seconds"] / 3600.0),
                sample_throughput=_fmt(row["sampling_images_per_second"]),
                fid=_fmt(row["fid"]),
                iscore=_fmt(row["inception_score"]),
                precision=_fmt(row["precision"]),
                recall=_fmt(row["recall"]),
                class_top1=_fmt(row["class_top1_accuracy"]),
                class_top5=_fmt(row["class_top5_accuracy"]),
                class_coverage=_fmt(row["class_predicted_class_fraction"]),
                class_entropy=_fmt(row["class_normalized_predicted_entropy"]),
            )
        )
    lines.extend(
        [
            "",
            (
                "Quality is compared under matched dataset, resolution, effective "
                "batch, optimizer steps, and training images. This is not an equal "
                "wall-clock, GPU-hours, or FLOPs budget; training time, throughput, "
                "and peak VRAM are measured outcomes."
            ),
            "",
            (
                "Training wall-clock and throughput are directly comparable."
                if report["comparison_policy"][
                    "training_wall_clock_direct_comparison_allowed"
                ]
                else (
                    "Training wall-clock and throughput are raw observations only; "
                    "do not interpret them as a model-efficiency ranking "
                    f"(`{report['comparison_policy']['training_wall_clock_comparison_reason']}`)."
                )
            ),
            "",
            "## Official pretrained context (not a direct ranking)",
            "",
            "| method | protocol | samples | FID | IS | precision | recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["official_context_rows"]:
        lines.append(
            "| {method} | {protocol} | {samples} | {fid} | {iscore} | {precision} | {recall} |".format(
                method=row["method"],
                protocol=row["protocol_note"],
                samples=row["sample_count"],
                fid=_fmt(row["fid"]),
                iscore=_fmt(row["inception_score"]),
                precision=_fmt(row["precision"]),
                recall=_fmt(row["recall"]),
            )
        )
    lines.extend(
        [
            "",
            "Do not rank across the two panels: checkpoint source, training budget, and evaluator differ.",
            "",
        ]
    )
    return "\n".join(lines)


def render_csv(report: dict[str, Any]) -> str:
    fields = [
        "comparison_tier",
        "directly_comparable_to_cofitok",
        "method",
        "dataset",
        "resolution",
        "parameter_count",
        "training_steps",
        "effective_batch_size",
        "training_images_seen",
        "training_elapsed_seconds",
        "training_images_per_second",
        "training_time_measurement",
        "training_budget_basis",
        "compute_matched_claim_allowed",
        "training_cost_fields_role",
        "training_wall_clock_directly_comparable",
        "training_throughput_directly_comparable",
        "training_wall_clock_comparison_reason",
        "peak_vram_bytes",
        "sample_count",
        "real_image_count",
        "sample_batch_size",
        "sampling_elapsed_seconds",
        "sampling_images_per_second",
        "sampling_invocations",
        "weights",
        "sampling_protocol_schema",
        "sampler",
        "num_train_timesteps",
        "sample_steps",
        "guidance_scale",
        "guidance_rescale",
        "cfg_batch_mode",
        "eta",
        "clip_x0",
        "sampling_precision",
        "sampling_seed",
        "sampling_start_index",
        "class_schedule",
        "fid",
        "inception_score",
        "precision",
        "recall",
        "class_fidelity_sample_count",
        "class_top1_accuracy",
        "class_top5_accuracy",
        "class_mean_target_probability",
        "class_target_negative_log_likelihood",
        "class_predicted_class_fraction",
        "class_normalized_predicted_entropy",
        "real_set_digest_schema",
        "real_set_sha256",
        "evaluator_runtime_environment_sha256",
        "protocol_note",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["matched_training_rows"])
    writer.writerows(report["official_context_rows"])
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final two-tier generation comparison.")
    parser.add_argument("--cofitok-training", required=True)
    parser.add_argument("--dense-training", required=True)
    parser.add_argument("--cofitok-generation", required=True)
    parser.add_argument("--dense-generation", required=True)
    parser.add_argument("--final-gate", required=True)
    parser.add_argument("--class-fidelity-qualification")
    parser.add_argument("--official-related", required=True)
    parser.add_argument("--training-contention", required=True)
    parser.add_argument(
        "--source-profile",
        choices=sorted(SOURCE_REPORT_PROFILES),
        default="full",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_paths = {
        "cofitok_training": Path(args.cofitok_training),
        "dense_training": Path(args.dense_training),
        "cofitok_generation": Path(args.cofitok_generation),
        "dense_generation": Path(args.dense_generation),
        "final_gate": Path(args.final_gate),
        "training_contention": Path(args.training_contention),
    }
    if args.class_fidelity_qualification:
        if args.source_profile != "stability_full":
            raise ValueError(
                "--class-fidelity-qualification requires stability_full source profile"
            )
        source_paths["class_fidelity_qualification"] = Path(
            args.class_fidelity_qualification
        )
    if (
        args.source_profile == "stability_full"
        and "class_fidelity_qualification" not in source_paths
    ):
        raise ValueError(
            "stability-full comparison requires --class-fidelity-qualification"
        )
    report = build_report(
        cofitok_training=_read(source_paths["cofitok_training"]),
        dense_training=_read(source_paths["dense_training"]),
        cofitok_generation=_read(source_paths["cofitok_generation"]),
        dense_generation=_read(source_paths["dense_generation"]),
        final_gate=_read(source_paths["final_gate"]),
        official_related=_read(args.official_related),
        training_contention=_read(args.training_contention),
        class_fidelity_qualification=(
            _read(args.class_fidelity_qualification)
            if args.class_fidelity_qualification
            else None
        ),
        official_source_path=Path(args.official_related).resolve().as_posix(),
        official_source_sha256=file_sha256(args.official_related),
        source_reports={
            name: source_report_identity(path) for name, path in source_paths.items()
        },
        source_profile=args.source_profile,
    )
    output_dir = Path(args.output_dir)
    write_json_report(output_dir / "large_scale_generation_comparison.json", report)
    write_text_report(output_dir / "large_scale_generation_comparison.md", render_markdown(report))
    write_text_report(output_dir / "large_scale_generation_comparison.csv", render_csv(report))
    print(output_dir)


if __name__ == "__main__":
    main()
