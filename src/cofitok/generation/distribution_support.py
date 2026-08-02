from __future__ import annotations

import math
from typing import Any

from cofitok.environment import runtime_environment_sha256
from cofitok.generation.protocol import sampling_protocol_contract
from cofitok.generation_gate import (
    STABILITY_SCALING_MAX_PRECISION_REGRESSION,
    STABILITY_SCALING_MAX_RECALL_REGRESSION,
    STABILITY_SCALING_MIN_PRECISION,
    STABILITY_SCALING_MIN_RECALL,
)
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA


STABILITY_DISTRIBUTION_SUPPORT_SCHEMA_VERSION = 1
STABILITY_DISTRIBUTION_SUPPORT_ROLE = (
    "generation_stability_distribution_support_qualification"
)


def _finite_metric(report: dict[str, Any], name: str) -> float | None:
    raw = report.get("metrics", {}).get(name)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _integer(value: Any, *, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _git_identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and len(str(value.get("revision", ""))) == 40
        and all(
            character in "0123456789abcdef"
            for character in str(value.get("revision", ""))
        )
        and bool(str(value.get("branch", "")))
        and value.get("tracked_dirty") is False
    )


def _runtime_identity(payload: dict[str, Any]) -> dict[str, Any]:
    environment = payload.get("runtime_environment")
    declared = payload.get("runtime_environment_sha256")
    if not isinstance(environment, dict):
        return {"valid": False, "sha256": declared}
    actual = runtime_environment_sha256(environment)
    return {"valid": declared == actual, "sha256": actual}


def _check(name: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _quality_view(report: dict[str, Any]) -> dict[str, float | None]:
    return {
        "fid": _finite_metric(report, "frechet_inception_distance"),
        "inception_score_mean": _finite_metric(report, "inception_score_mean"),
        "inception_score_std": _finite_metric(report, "inception_score_std"),
        "precision": _finite_metric(report, "precision"),
        "recall": _finite_metric(report, "recall"),
    }


def _formal_metrics_contract(
    report: dict[str, Any], *, expected_prefix_budget: int
) -> dict[str, Any]:
    sample_provenance = report.get("sample_provenance", {})
    sampling = sample_provenance.get("sampling", {})
    contract = (
        sampling_protocol_contract(
            sampling,
            stage="scaling",
            expected_num_train_timesteps=1000,
        )
        if isinstance(sampling, dict)
        else {"valid": False, "issues": ["sampling"]}
    )
    counts = report.get("counts", {})
    progress = sample_provenance.get("sampling_progress", {})
    schema_version = _integer(report.get("schema_version"))
    role = report.get("role")
    evaluator_runtime = _runtime_identity(report)
    sampling_runtime = _runtime_identity(sample_provenance)
    checkpoint_integrity_manifest = str(
        sample_provenance.get("checkpoint_integrity_manifest", "")
    )
    report_contract = (
        schema_version in {2, 3}
        and report.get("status") == "completed"
        and report.get("protocol") == "torch_fidelity_directory_metrics"
        and (schema_version == 2 or role == "generation_directory_metrics_report")
        and _integer(counts.get("real_image_count")) == 50_000
        and _integer(counts.get("generated_image_count")) == 10_000
        and sample_provenance.get("weights") == "ema"
        and _integer(sample_provenance.get("checkpoint_step")) == 50_000
        and _sha256(sample_provenance.get("checkpoint_sha256"))
        and checkpoint_integrity_manifest.endswith(".pt.integrity.json")
        and _sha256(sample_provenance.get("sample_set_sha256"))
        and _integer(sample_provenance.get("selected_prefix_budget"))
        == expected_prefix_budget
        and sampling.get("prefix_budgets") == [expected_prefix_budget]
        and sampling.get("image_shape") == [3, 256, 256]
        and _git_identity(report.get("git"))
        and _git_identity(sample_provenance.get("git"))
        and evaluator_runtime["valid"] is True
        and sampling_runtime["valid"] is True
        and isinstance(progress, dict)
        and progress.get("status") == "completed"
        and _integer(progress.get("completed_samples")) == 10_000
        and isinstance(progress.get("cumulative_elapsed_seconds"), (int, float))
        and math.isfinite(float(progress["cumulative_elapsed_seconds"]))
        and float(progress["cumulative_elapsed_seconds"]) > 0.0
        and contract.get("valid") is True
    )
    return {
        "valid": report_contract,
        "schema_version": schema_version,
        "role": role,
        "status": report.get("status"),
        "protocol": report.get("protocol"),
        "counts": counts,
        "weights": sample_provenance.get("weights"),
        "checkpoint_step": sample_provenance.get("checkpoint_step"),
        "checkpoint_sha256": sample_provenance.get("checkpoint_sha256"),
        "checkpoint_integrity_manifest": checkpoint_integrity_manifest,
        "sample_set_sha256": sample_provenance.get("sample_set_sha256"),
        "selected_prefix_budget": sample_provenance.get(
            "selected_prefix_budget"
        ),
        "expected_prefix_budget": expected_prefix_budget,
        "evaluator_git": report.get("git"),
        "sampling_git": sample_provenance.get("git"),
        "evaluator_runtime": evaluator_runtime,
        "sampling_runtime": sampling_runtime,
        "sampling_progress": progress,
        "sampling_contract": contract,
    }


def _matched_metrics_contract(
    cofitok: dict[str, Any], dense: dict[str, Any]
) -> dict[str, Any]:
    cofitok_provenance = cofitok.get("sample_provenance", {})
    dense_provenance = dense.get("sample_provenance", {})
    cofitok_contract = _formal_metrics_contract(
        cofitok, expected_prefix_budget=8
    )
    dense_contract = _formal_metrics_contract(dense, expected_prefix_budget=1)
    cofitok_real_set = cofitok.get("real_set")
    dense_real_set = dense.get("real_set")
    cofitok_git = cofitok.get("git")
    dense_git = dense.get("git")
    cofitok_sampling_git = cofitok_provenance.get("git")
    dense_sampling_git = dense_provenance.get("git")
    evaluator_runtime_match = (
        cofitok_contract["evaluator_runtime"]["valid"] is True
        and dense_contract["evaluator_runtime"]["valid"] is True
        and cofitok_contract["evaluator_runtime"]["sha256"]
        == dense_contract["evaluator_runtime"]["sha256"]
    )
    sampling_runtime_match = (
        cofitok_contract["sampling_runtime"]["valid"] is True
        and dense_contract["sampling_runtime"]["valid"] is True
        and cofitok_contract["sampling_runtime"]["sha256"]
        == dense_contract["sampling_runtime"]["sha256"]
    )
    evaluator_git_match = _git_identity(cofitok_git) and cofitok_git == dense_git
    sampling_git_match = (
        _git_identity(cofitok_sampling_git)
        and cofitok_sampling_git == dense_sampling_git
    )
    cofitok_sampling = cofitok_provenance.get("sampling", {})
    dense_sampling = dense_provenance.get("sampling", {})
    matched_sampling_protocol = (
        isinstance(cofitok_sampling, dict)
        and isinstance(dense_sampling, dict)
        and {
            key: value
            for key, value in cofitok_sampling.items()
            if key != "prefix_budgets"
        }
        == {
            key: value
            for key, value in dense_sampling.items()
            if key != "prefix_budgets"
        }
    )
    implementation = cofitok.get("implementation")
    real_set_path_matches = (
        isinstance(cofitok_real_set, dict)
        and isinstance(dense_real_set, dict)
        and cofitok_real_set.get("root")
        == cofitok.get("paths", {}).get("real_dir")
        and dense_real_set.get("root")
        == dense.get("paths", {}).get("real_dir")
    )
    checks = {
        "cofitok_formal_metrics": cofitok_contract["valid"],
        "dense_formal_metrics": dense_contract["valid"],
        "matched_real_set": (
            isinstance(cofitok_real_set, dict)
            and cofitok_real_set == dense_real_set
            and cofitok_real_set.get("digest_schema")
            == IMAGE_TREE_DIGEST_SCHEMA
            and _sha256(cofitok_real_set.get("sha256"))
            and _integer(cofitok_real_set.get("image_count")) == 50_000
            and real_set_path_matches
        ),
        "matched_evaluator": (
            isinstance(implementation, dict)
            and implementation == dense.get("implementation")
            and implementation.get("package") == "torch_fidelity"
            and bool(str(implementation.get("version", "")))
        ),
        "matched_evaluator_git": evaluator_git_match,
        "matched_evaluator_runtime": evaluator_runtime_match,
        "matched_sampling_git": sampling_git_match,
        "matched_sampling_runtime": sampling_runtime_match,
        "matched_sampling_protocol": matched_sampling_protocol,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "cofitok": cofitok_contract,
        "dense_identity": dense_contract,
        "real_set": cofitok_real_set,
        "implementation": implementation,
        "evaluator_git": cofitok_git,
        "evaluator_runtime_environment_sha256": cofitok.get(
            "runtime_environment_sha256"
        ),
        "sampling_git": cofitok_sampling_git,
        "sampling_runtime_environment_sha256": cofitok_provenance.get(
            "runtime_environment_sha256"
        ),
    }


def _base_gate_metric_binding(
    base_gate: dict[str, Any],
    cofitok_quality: dict[str, float | None],
    dense_quality: dict[str, float | None],
) -> dict[str, Any]:
    summary = base_gate.get("summary")
    expected = {
        "cofitok_fid": cofitok_quality["fid"],
        "dense_fid": dense_quality["fid"],
        "cofitok_inception_score": cofitok_quality["inception_score_mean"],
        "dense_inception_score": dense_quality["inception_score_mean"],
        "cofitok_precision": cofitok_quality["precision"],
        "dense_precision": dense_quality["precision"],
        "cofitok_recall": cofitok_quality["recall"],
        "dense_recall": dense_quality["recall"],
    }
    actual = (
        {name: summary.get(name) for name in expected}
        if isinstance(summary, dict)
        else None
    )
    return {"valid": actual == expected, "expected": expected, "actual": actual}


def build_stability_distribution_support_qualification(
    *,
    base_gate: dict[str, Any],
    cofitok_generation: dict[str, Any],
    dense_generation: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    builder_git: dict[str, Any],
    min_precision: float = STABILITY_SCALING_MIN_PRECISION,
    min_recall: float = STABILITY_SCALING_MIN_RECALL,
    max_precision_regression: float = (
        STABILITY_SCALING_MAX_PRECISION_REGRESSION
    ),
    max_recall_regression: float = STABILITY_SCALING_MAX_RECALL_REGRESSION,
) -> dict[str, Any]:
    thresholds = {
        "min_precision": min_precision,
        "min_recall": min_recall,
        "max_precision_regression": max_precision_regression,
        "max_recall_regression": max_recall_regression,
    }
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in thresholds.values()
    ):
        raise ValueError("distribution-support thresholds must be finite and in [0, 1]")
    if min_precision < STABILITY_SCALING_MIN_PRECISION:
        raise ValueError("min_precision is weaker than the stability-scaling boundary")
    if min_recall < STABILITY_SCALING_MIN_RECALL:
        raise ValueError("min_recall is weaker than the stability-scaling boundary")
    if max_precision_regression > STABILITY_SCALING_MAX_PRECISION_REGRESSION:
        raise ValueError(
            "max_precision_regression is weaker than the stability-scaling boundary"
        )
    if max_recall_regression > STABILITY_SCALING_MAX_RECALL_REGRESSION:
        raise ValueError(
            "max_recall_regression is weaker than the stability-scaling boundary"
        )
    if (
        base_gate.get("stage") != "scaling"
        or base_gate.get("source_profile") != "stability_scaling"
        or _integer(base_gate.get("schema_version")) not in {2, 3, 4}
    ):
        raise ValueError("base gate is not a supported stability-scaling gate")
    if not _git_identity(builder_git):
        raise ValueError("distribution-support builder Git identity is not clean")
    expected_sources = {"promotion_gate", "cofitok_generation", "dense_generation"}
    if set(sources) != expected_sources:
        raise ValueError("distribution-support source identity set is incomplete")
    for name, identity in sources.items():
        if (
            not isinstance(identity, dict)
            or not str(identity.get("path", ""))
            or _integer(identity.get("bytes"), default=0) < 1
            or not _sha256(identity.get("sha256"))
        ):
            raise ValueError(
                f"distribution-support source identity is invalid: {name}"
            )

    cofitok_quality = _quality_view(cofitok_generation)
    dense_quality = _quality_view(dense_generation)
    quality_values = (*cofitok_quality.values(), *dense_quality.values())
    metrics_in_range = (
        all(value is not None for value in quality_values)
        and cofitok_quality["fid"] >= 0.0
        and dense_quality["fid"] >= 0.0
        and cofitok_quality["inception_score_mean"] > 0.0
        and dense_quality["inception_score_mean"] > 0.0
        and cofitok_quality["inception_score_std"] >= 0.0
        and dense_quality["inception_score_std"] >= 0.0
        and all(
            0.0 <= value <= 1.0
            for value in (
                cofitok_quality["precision"],
                dense_quality["precision"],
                cofitok_quality["recall"],
                dense_quality["recall"],
            )
        )
    )
    matched_contract = _matched_metrics_contract(
        cofitok_generation, dense_generation
    )
    metric_binding = _base_gate_metric_binding(
        base_gate, cofitok_quality, dense_quality
    )
    checks = [
        _check(
            "base_gate_metric_binding",
            metric_binding["valid"],
            metric_binding,
        ),
        _check(
            "formal_matched_metrics",
            matched_contract["valid"],
            matched_contract,
        ),
        _check(
            "distribution_metric_ranges",
            metrics_in_range,
            {
                "cofitok": cofitok_quality,
                "dense_identity": dense_quality,
            },
        ),
        _check(
            "minimum_precision",
            cofitok_quality["precision"] is not None
            and cofitok_quality["precision"] >= min_precision,
            {
                "cofitok_precision": cofitok_quality["precision"],
                "min_precision": min_precision,
            },
        ),
        _check(
            "minimum_recall",
            cofitok_quality["recall"] is not None
            and cofitok_quality["recall"] >= min_recall,
            {
                "cofitok_recall": cofitok_quality["recall"],
                "min_recall": min_recall,
            },
        ),
        _check(
            "matched_precision_retention",
            cofitok_quality["precision"] is not None
            and dense_quality["precision"] is not None
            and cofitok_quality["precision"]
            >= dense_quality["precision"] - max_precision_regression,
            {
                "cofitok_precision": cofitok_quality["precision"],
                "dense_precision": dense_quality["precision"],
                "max_precision_regression": max_precision_regression,
            },
        ),
        _check(
            "matched_recall_retention",
            cofitok_quality["recall"] is not None
            and dense_quality["recall"] is not None
            and cofitok_quality["recall"]
            >= dense_quality["recall"] - max_recall_regression,
            {
                "cofitok_recall": cofitok_quality["recall"],
                "dense_recall": dense_quality["recall"],
                "max_recall_regression": max_recall_regression,
            },
        ),
    ]
    passed = all(check["passed"] for check in checks)
    base_gate_view = {
        "schema_version": _integer(base_gate.get("schema_version")),
        "stage": base_gate.get("stage"),
        "source_profile": base_gate.get("source_profile"),
        "status": base_gate.get("status"),
        "decision": base_gate.get("decision"),
    }
    base_gate_passed = (
        base_gate.get("status") == "pass"
        and base_gate.get("decision") == "promote_to_full_imagenet256"
    )
    return {
        "schema_version": STABILITY_DISTRIBUTION_SUPPORT_SCHEMA_VERSION,
        "role": STABILITY_DISTRIBUTION_SUPPORT_ROLE,
        "status": "pass" if passed else "fail",
        "decision": "distribution_support_qualified" if passed else "hold",
        "thresholds": thresholds,
        "base_gate": base_gate_view,
        "sources": sources,
        "builder_git": builder_git,
        "checks": checks,
        "summary": {
            "cofitok_fid": cofitok_quality["fid"],
            "dense_fid": dense_quality["fid"],
            "cofitok_inception_score": cofitok_quality[
                "inception_score_mean"
            ],
            "dense_inception_score": dense_quality["inception_score_mean"],
            "cofitok_precision": cofitok_quality["precision"],
            "dense_precision": dense_quality["precision"],
            "cofitok_recall": cofitok_quality["recall"],
            "dense_recall": dense_quality["recall"],
        },
        "decision_boundary": {
            "base_gate_passed": base_gate_passed,
            "distribution_support_passed": passed,
            "base_gate_and_distribution_support_passed": (
                base_gate_passed and passed
            ),
            "scaling_authorization_evaluated": False,
            "full_training_launch_allowed": False,
        },
        "claim_boundary": {
            "supplemental_non_authorizing": True,
            "replaces_generation_gate": False,
            "replaces_rollout_stability_qualification": False,
            "full_training_launch_allowed": False,
            "interpretation": (
                "Matched 10K EMA precision/recall non-collapse qualification only"
            ),
        },
    }
