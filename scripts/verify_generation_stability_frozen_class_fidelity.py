from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from cofitok.generation_gate import validate_generation_gate_authorization
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.reporting import file_sha256


ROLE = "generation_class_fidelity_qualification"
REPORT_ROLE = "generation_class_fidelity_report"
CLASSIFIER = {
    "name": "torchvision_resnet50_imagenet1k_v2",
    "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
    "weights_bytes": 102540417,
    "weights_sha256": (
        "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
    ),
    "num_classes": 1000,
    "categories_sha256": (
        "62fff941ecff3f19de9128c6ca9c2097807c6b7bafc5552d7589a219431221ed"
    ),
    "preprocessing": {
        "resize_size": [232],
        "crop_size": [224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bilinear",
        "antialias": True,
    },
}
THRESHOLDS = {
    "min_top1": 0.01,
    "min_top5": 0.05,
    "min_predicted_class_fraction": 0.25,
    "min_normalized_predicted_entropy": 0.50,
    "max_top1_regression": 0.05,
    "max_top5_regression": 0.05,
}
METRIC_FIELDS = {
    "sample_count",
    "num_classes",
    "top1_correct",
    "top5_correct",
    "top1_accuracy",
    "top5_accuracy",
    "mean_target_probability",
    "target_negative_log_likelihood",
    "requested_class_count",
    "requested_count_min",
    "requested_count_max",
    "predicted_class_count",
    "predicted_class_fraction",
    "predicted_class_entropy",
    "normalized_predicted_class_entropy",
}


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _hex_digest(value: Any, *, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(
        character in "0123456789abcdef" for character in text
    )


def _git(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and _hex_digest(value.get("revision"), length=40)
        and bool(str(value.get("branch", "")))
        and value.get("tracked_dirty") is False
    )


def _classifier(value: Any) -> bool:
    return isinstance(value, dict) and all(
        value.get(name) == expected for name, expected in CLASSIFIER.items()
    )


def _finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"class-fidelity {name} is malformed") from error
    if not math.isfinite(number):
        raise ValueError(f"class-fidelity {name} is not finite")
    return number


def _runtime_environment_sha256(value: Any) -> str:
    if not isinstance(value, dict):
        raise ValueError("class-fidelity runtime environment is missing")
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def _verify_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict) or set(metrics) != METRIC_FIELDS:
        raise ValueError("class-fidelity metrics are incomplete")
    sample_count = int(metrics["sample_count"])
    num_classes = int(metrics["num_classes"])
    top1 = int(metrics["top1_correct"])
    top5 = int(metrics["top5_correct"])
    requested_count = int(metrics["requested_class_count"])
    requested_min = int(metrics["requested_count_min"])
    requested_max = int(metrics["requested_count_max"])
    predicted_count = int(metrics["predicted_class_count"])
    if (
        sample_count != 10_000
        or num_classes != 1000
        or not 0 <= top1 <= top5 <= sample_count
        or requested_count != num_classes
        or requested_min != 10
        or requested_max != 10
        or not 1 <= predicted_count <= num_classes
    ):
        raise ValueError("class-fidelity metric counts differ from the scaling contract")
    bounded = (
        "top1_accuracy",
        "top5_accuracy",
        "mean_target_probability",
        "predicted_class_fraction",
        "normalized_predicted_class_entropy",
    )
    for name in bounded:
        value = _finite(metrics[name], name=name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"class-fidelity metric {name} is outside [0, 1]")
    entropy = _finite(metrics["predicted_class_entropy"], name="entropy")
    nll = _finite(metrics["target_negative_log_likelihood"], name="NLL")
    if entropy < 0.0 or entropy > math.log(num_classes) + 1e-12 or nll < 0.0:
        raise ValueError("class-fidelity entropy or NLL is outside its domain")
    expected = {
        "top1_accuracy": top1 / sample_count,
        "top5_accuracy": top5 / sample_count,
        "predicted_class_fraction": predicted_count / num_classes,
        "normalized_predicted_class_entropy": entropy / math.log(num_classes),
    }
    for name, value in expected.items():
        if not math.isclose(float(metrics[name]), value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"class-fidelity metric {name} is inconsistent")
    return metrics


def _expected_checks(
    cofitok: dict[str, Any],
    dense: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for method, metrics in (("cofitok", cofitok), ("dense_identity", dense)):
        for metric_name, threshold_name in (
            ("top1_accuracy", "min_top1"),
            ("top5_accuracy", "min_top5"),
            ("predicted_class_fraction", "min_predicted_class_fraction"),
            (
                "normalized_predicted_class_entropy",
                "min_normalized_predicted_entropy",
            ),
        ):
            observed = float(metrics[metric_name])
            threshold = THRESHOLDS[threshold_name]
            checks.append(
                {
                    "name": f"{method}_{metric_name}",
                    "status": "pass" if observed >= threshold else "hold",
                    "observed": observed,
                    "threshold": threshold,
                    "comparison": ">=",
                }
            )
    for metric_name, threshold_name in (
        ("top1_accuracy", "max_top1_regression"),
        ("top5_accuracy", "max_top5_regression"),
    ):
        observed = float(dense[metric_name]) - float(cofitok[metric_name])
        threshold = THRESHOLDS[threshold_name]
        checks.append(
            {
                "name": f"cofitok_{metric_name}_regression_vs_dense",
                "status": "pass" if observed <= threshold else "hold",
                "observed": observed,
                "threshold": threshold,
                "comparison": "<=",
            }
        )
    return checks


def _timesteps() -> list[int]:
    return sorted({round(index * 999 / 99) for index in range(100)}, reverse=True)


def _verify_sampling(sampling: Any) -> dict[str, Any]:
    if not isinstance(sampling, dict):
        raise ValueError("class-fidelity sampling protocol is missing")
    expected = {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "inference_api": {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        },
        "sampler": "ddim",
        "num_samples": 10_000,
        "start_index": 0,
        "sample_steps": 100,
        "num_train_timesteps": 1000,
        "actual_timesteps": _timesteps(),
        "image_shape": [3, 256, 256],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "class_schedule": "balanced_modulo",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
    }
    if any(sampling.get(name) != value for name, value in expected.items()):
        raise ValueError("class-fidelity sampling protocol is not formal scaling EMA")
    stream = sampling.get("random_stream")
    if not isinstance(stream, dict) or any(
        stream.get(name) is not True
        for name in (
            "prefix_budgets_share_stream",
            "batch_size_invariant",
            "resume_index_invariant",
        )
    ):
        raise ValueError("class-fidelity random-stream contract differs")
    return sampling


def _gate_sampling(gate: dict[str, Any]) -> dict[str, Any]:
    rows = gate.get("gates")
    if not isinstance(rows, list):
        raise ValueError("promotion gate checks are missing")
    for row in rows:
        if isinstance(row, dict) and row.get("name") == "matched_sampling_provenance":
            evidence = row.get("evidence")
            if isinstance(evidence, dict):
                return evidence
    raise ValueError("promotion gate matched sampling provenance is missing")


def _verify_raw(
    report: dict[str, Any],
    *,
    prefix_budget: int,
    evaluator_git: dict[str, Any],
    sampling_git: dict[str, Any],
    runtime_sha256: str,
) -> dict[str, Any]:
    sample = report.get("sample_provenance")
    parameters = report.get("parameters")
    environment = report.get("runtime_environment")
    if (
        report.get("schema_version") != 2
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("protocol") != "torchvision_imagenet_class_fidelity"
        or report.get("git") != evaluator_git
        or report.get("runtime_environment_sha256") != runtime_sha256
        or _runtime_environment_sha256(environment) != runtime_sha256
        or not _classifier(report.get("classifier"))
        or not isinstance(sample, dict)
        or sample.get("git") != sampling_git
        or sample.get("weights") != "ema"
        or sample.get("selected_prefix_budget") != prefix_budget
        or not _hex_digest(sample.get("checkpoint_sha256"), length=64)
        or not _hex_digest(sample.get("sample_set_sha256"), length=64)
        or not isinstance(parameters, dict)
        or parameters.get("num_classes") != 1000
        or parameters.get("sample_count") != 10_000
        or parameters.get("target_from_filename")
        != "int(zero_based_png_stem) mod 1000"
    ):
        raise ValueError("frozen class-fidelity raw report contract differs")
    sampling = dict(_verify_sampling(sample.get("sampling")))
    if sampling.pop("prefix_budgets", None) != [prefix_budget]:
        raise ValueError("frozen class-fidelity prefix budget differs")
    _verify_metrics(report.get("metrics"))
    return report


def verify_frozen_class_fidelity_qualification(
    report: dict[str, Any],
    *,
    report_path: Path,
    expected_report_sha256: str,
    promotion_gate_path: Path,
    expected_evaluator_revision: str,
    expected_evaluator_branch: str,
) -> dict[str, Any]:
    report_path = report_path.resolve()
    promotion_gate_path = promotion_gate_path.resolve()
    if file_sha256(report_path) != expected_report_sha256:
        raise ValueError("frozen class-fidelity qualification SHA256 differs")
    if _read(report_path) != report:
        raise ValueError("frozen class-fidelity qualification payload differs")
    evaluator_git = {
        "revision": expected_evaluator_revision,
        "branch": expected_evaluator_branch,
        "tracked_dirty": False,
    }
    if (
        report.get("schema_version") != 2
        or report.get("role") != ROLE
        or report.get("status") != "pass"
        or report.get("stage") != "scaling"
        or report.get("git") != evaluator_git
        or not _classifier(report.get("classifier"))
        or report.get("thresholds") != THRESHOLDS
    ):
        raise ValueError("frozen class-fidelity qualification did not pass exactly")
    boundary = report.get("claim_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("class_conditional_quality_evaluated") is not True
        or boundary.get("unconditional_distribution_quality_evaluated") is not False
        or boundary.get("standalone_generation_quality_claim_allowed") is not False
        or boundary.get("full_training_launch_allowed") is not False
        or boundary.get("release_authorization_allowed") is not False
    ):
        raise ValueError("frozen class-fidelity claim boundary differs")

    contract = report.get("sampling_contract")
    if not isinstance(contract, dict):
        raise ValueError("frozen class-fidelity sampling contract is missing")
    sampling = _verify_sampling(contract.get("sampling"))
    sampling_git = contract.get("sampling_git")
    runtime_sha256 = str(contract.get("evaluator_runtime_environment_sha256", ""))
    identity_fields = (
        "cofitok_checkpoint_sha256",
        "dense_checkpoint_sha256",
        "cofitok_sample_set_sha256",
        "dense_sample_set_sha256",
    )
    if (
        not _git(sampling_git)
        or contract.get("evaluator_git") != evaluator_git
        or not _hex_digest(runtime_sha256, length=64)
        or any(
            not _hex_digest(contract.get(name), length=64)
            for name in identity_fields
        )
        or contract.get("cofitok_prefix_budget") != 8
        or contract.get("dense_prefix_budget") != 1
        or contract.get("weights") != "ema"
        or contract.get("sample_count_per_method") != 10_000
    ):
        raise ValueError("frozen class-fidelity cross-revision contract differs")

    sources = report.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"cofitok", "dense_identity"}:
        raise ValueError("frozen class-fidelity source set differs")
    raw: dict[str, dict[str, Any]] = {}
    for name, prefix in (("cofitok", 8), ("dense_identity", 1)):
        descriptor = sources[name]
        if not isinstance(descriptor, dict):
            raise ValueError(f"frozen class-fidelity {name} source is malformed")
        path = Path(str(descriptor.get("path", "")))
        if _identity(path) != descriptor:
            raise ValueError(f"frozen class-fidelity {name} source changed")
        raw[name] = _verify_raw(
            _read(path),
            prefix_budget=prefix,
            evaluator_git=evaluator_git,
            sampling_git=sampling_git,
            runtime_sha256=runtime_sha256,
        )
    metrics = report.get("metrics")
    difference_fields = (
        "top1_accuracy",
        "top5_accuracy",
        "mean_target_probability",
        "predicted_class_fraction",
        "normalized_predicted_class_entropy",
    )
    expected_difference = {
        name: float(raw["cofitok"]["metrics"][name])
        - float(raw["dense_identity"]["metrics"][name])
        for name in difference_fields
    }
    if (
        not isinstance(metrics, dict)
        or set(metrics) != {"cofitok", "dense_identity", "cofitok_minus_dense"}
        or metrics.get("cofitok") != raw["cofitok"].get("metrics")
        or metrics.get("dense_identity") != raw["dense_identity"].get("metrics")
        or metrics.get("cofitok_minus_dense") != expected_difference
    ):
        raise ValueError("frozen class-fidelity qualification metrics differ from raw reports")
    for name, raw_name in (("cofitok", "cofitok"), ("dense", "dense_identity")):
        sample = raw[raw_name]["sample_provenance"]
        if (
            contract.get(f"{name}_checkpoint_sha256")
            != sample.get("checkpoint_sha256")
            or contract.get(f"{name}_sample_set_sha256")
            != sample.get("sample_set_sha256")
        ):
            raise ValueError("frozen class-fidelity qualification sample identity differs")
    checks = report.get("checks")
    expected_checks = _expected_checks(
        raw["cofitok"]["metrics"],
        raw["dense_identity"]["metrics"],
    )
    if checks != expected_checks or any(
        row["status"] != "pass" for row in expected_checks
    ):
        raise ValueError("frozen class-fidelity qualification checks did not pass")

    gate_identity = _identity(promotion_gate_path)
    gate = _read(promotion_gate_path)
    verify_generation_gate_source_reports(gate)
    validate_generation_gate_authorization(gate, expected_stage="scaling")
    provenance = gate.get("provenance_contract")
    if (
        gate.get("source_profile") != "stability_scaling"
        or not isinstance(provenance, dict)
        or sampling_git
        != {
            "revision": provenance.get("evaluation_revision"),
            "branch": provenance.get("evaluation_branch"),
            "tracked_dirty": False,
        }
    ):
        raise ValueError("frozen class-fidelity sampling Git differs from the promotion gate")
    gate_sampling = _gate_sampling(gate)
    for name in (
        "cofitok_checkpoint_sha256",
        "dense_checkpoint_sha256",
        "cofitok_sample_set_sha256",
        "dense_sample_set_sha256",
    ):
        if contract.get(name) != gate_sampling.get(name):
            raise ValueError("frozen class-fidelity sample identity differs from the gate")

    if file_sha256(report_path) != expected_report_sha256 or _read(report_path) != report:
        raise ValueError("frozen class-fidelity qualification changed while verifying")
    for name, descriptor in sources.items():
        if _identity(Path(descriptor["path"])) != descriptor:
            raise ValueError(f"frozen class-fidelity {name} source changed while verifying")
    if (
        _identity(promotion_gate_path) != gate_identity
        or _read(promotion_gate_path) != gate
    ):
        raise ValueError("frozen class-fidelity promotion gate changed while verifying")
    return {
        "report": _identity(report_path),
        "promotion_gate": gate_identity,
        "evaluator_git": evaluator_git,
        "sampling_git": sampling_git,
        "classifier": report["classifier"],
        "thresholds": THRESHOLDS,
        "metrics": {
            "cofitok": metrics["cofitok"],
            "dense_identity": metrics["dense_identity"],
        },
        "class_fidelity_passed": True,
        "supplemental_non_authorizing": True,
        "required_for_full_training_launch": True,
        "full_training_launch_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify cross-revision class fidelity over the frozen formal scaling "
            "sample sets as a mandatory non-authorizing full-launch prerequisite."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--expected-evaluator-revision", required=True)
    parser.add_argument("--expected-evaluator-branch", required=True)
    args = parser.parse_args()
    evidence = verify_frozen_class_fidelity_qualification(
        _read(args.report),
        report_path=args.report,
        expected_report_sha256=args.expected_report_sha256,
        promotion_gate_path=args.promotion_gate,
        expected_evaluator_revision=args.expected_evaluator_revision,
        expected_evaluator_branch=args.expected_evaluator_branch,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
