from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.environment import runtime_environment_sha256
from cofitok.generation import sampling_protocol_contract
from cofitok.generation_class_fidelity import (
    validate_class_fidelity_qualification,
)
from cofitok.generation_cost import training_cost_summary
from cofitok.generation_gate import GENERATION_GATE_SCHEMA_VERSION
from cofitok.generation_gate_sources import (
    GATE_SOURCE_SUFFIXES,
    build_generation_gate_diagnostic_reports,
    build_generation_gate_source_reports,
)
from cofitok.generation_pair import generation_pair_contract
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ImageNet-256 10% generation promotion gate.")
    parser.add_argument("--cofitok-training", required=True)
    parser.add_argument("--dense-training", required=True)
    parser.add_argument("--cofitok-generation", required=True)
    parser.add_argument("--dense-generation", required=True)
    parser.add_argument("--cofitok-checkpoint-eval", required=True)
    parser.add_argument("--dense-checkpoint-eval", required=True)
    parser.add_argument(
        "--rollout-stability-qualification",
        help=(
            "Optional matched EMA rollout-stability qualification. When supplied, "
            "it becomes a blocking, source-bound gate check."
        ),
    )
    parser.add_argument(
        "--class-fidelity-qualification",
        help=(
            "Optional matched ImageNet class-fidelity qualification. When supplied, "
            "it becomes a blocking, source-bound gate check."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["scaling", "full"], default="scaling")
    parser.add_argument(
        "--source-profile",
        choices=sorted(GATE_SOURCE_SUFFIXES),
        help="Authoritative path profile; defaults to the scientific gate stage.",
    )
    parser.add_argument("--expected-training-revision")
    parser.add_argument(
        "--expected-training-branch",
        default="scale/generative-system",
    )
    parser.add_argument("--expected-evaluation-revision")
    parser.add_argument(
        "--expected-evaluation-branch",
        default="scale/generative-system",
    )
    parser.add_argument("--min-samples", type=int, default=10_000)
    parser.add_argument("--max-fid-regression", type=float, default=0.05)
    parser.add_argument("--max-absolute-fid", type=float, default=100.0)
    parser.add_argument("--max-endpoint-regression", type=float, default=0.05)
    parser.add_argument("--min-coarse-token-energy-ratio", type=float, default=0.05)
    parser.add_argument("--min-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.30)
    parser.add_argument("--max-precision-regression", type=float, default=0.05)
    parser.add_argument("--max-recall-regression", type=float, default=0.05)
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
        try:
            value = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            value = None
        values[name] = value if value is not None and math.isfinite(value) else None
    return values


def _gate(name: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _positive_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def _distribution_metrics_valid(metrics: dict[str, float | None]) -> bool:
    fid = metrics["frechet_inception_distance"]
    inception_mean = metrics["inception_score_mean"]
    inception_std = metrics["inception_score_std"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    return (
        fid is not None
        and fid >= 0.0
        and inception_mean is not None
        and inception_mean > 0.0
        and inception_std is not None
        and inception_std >= 0.0
        and precision is not None
        and 0.0 <= precision <= 1.0
        and recall is not None
        and 0.0 <= recall <= 1.0
    )


def _coarse_token_utilization(
    report: dict[str, Any],
    *,
    require_stride_partition: bool = False,
) -> dict[str, Any]:
    raw = report.get("metrics", {}).get("component_energy_ratio_per_sample_mean")
    model = report.get("config", {}).get("model", {})
    try:
        ratios = [float(value) for value in raw]
        token_count = int(model["token_count"])
    except (KeyError, TypeError, ValueError):
        ratios = []
        token_count = 0
    raw_strides = model.get("token_spatial_strides")
    strides: list[int] = []
    if (
        isinstance(raw_strides, list)
        and len(raw_strides) == token_count
        and all(type(value) is int for value in raw_strides)
    ):
        strides = list(raw_strides)
    stride_partition_valid = (
        token_count >= 3
        and len(strides) == token_count
        and all(value > 0 for value in strides)
        and 1 in strides
    )
    if stride_partition_valid:
        coarse_token_count = strides.index(1)
        stride_partition_valid = (
            coarse_token_count > 0
            and all(value > 1 for value in strides[:coarse_token_count])
            and all(value == 1 for value in strides[coarse_token_count:])
        )
    if stride_partition_valid:
        partition_schema = "token_spatial_stride_suffix_v1"
    elif require_stride_partition:
        partition_schema = "invalid_missing_token_spatial_stride_suffix"
        coarse_token_count = 0
        strides = []
    else:
        partition_schema = "legacy_last_two_tokens"
        coarse_token_count = max(token_count - 2, 0)
        strides = []
    valid = (
        token_count >= 3
        and partition_schema
        != "invalid_missing_token_spatial_stride_suffix"
        and len(ratios) == token_count
        and all(math.isfinite(value) and value >= 0.0 for value in ratios)
        and math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6)
    )
    coarse_ratio = sum(ratios[:coarse_token_count]) if valid else None
    return {
        "valid": valid,
        "source_metric": "component_energy_ratio_per_sample_mean",
        "partition_schema": partition_schema,
        "token_count": token_count,
        "coarse_token_count": coarse_token_count,
        "full_resolution_tail_token_count": token_count - coarse_token_count,
        "token_spatial_strides": strides,
        "component_energy_ratios": ratios,
        "coarse_token_energy_ratio": coarse_ratio,
    }


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


def _rollout_stability_evidence(
    report: dict[str, Any],
    *,
    stage: str,
    expected_step: int,
    expected_checkpoint_images: int,
    expected_cofitok_sha256: str,
    expected_dense_sha256: str,
    expected_evaluation_revision: str | None,
    expected_evaluation_branch: str,
    sampling_protocol: dict[str, Any],
) -> dict[str, Any]:
    protocol = report.get("protocol")
    protocol = protocol if isinstance(protocol, dict) else {}
    rollout = protocol.get("rollout")
    rollout = rollout if isinstance(rollout, dict) else {}
    identity = report.get("identity")
    identity = identity if isinstance(identity, dict) else {}
    qualification_gates = report.get("gates")
    qualification_gates = (
        qualification_gates if isinstance(qualification_gates, dict) else {}
    )
    failed_gates = sorted(
        name
        for name, gate in qualification_gates.items()
        if not isinstance(gate, dict) or gate.get("passed") is not True
    )
    pair_contract = report.get("pair_contract")
    pair_contract_valid = (
        isinstance(pair_contract, dict) and pair_contract.get("valid") is True
    )
    expected_sample_steps = 100 if stage == "scaling" else 250
    protocol_matches = (
        int(rollout.get("num_images", 0)) >= 64
        and int(rollout.get("sample_steps", -1)) == expected_sample_steps
        and int(rollout.get("sample_steps", -1))
        == int(sampling_protocol.get("sample_steps", -2))
        and rollout.get("guidance_scale") == sampling_protocol.get("guidance_scale")
        and rollout.get("guidance_rescale")
        == sampling_protocol.get("guidance_rescale")
        and rollout.get("cfg_batch_mode") == sampling_protocol.get("cfg_batch_mode")
        and rollout.get("clip_x0") is sampling_protocol.get("clip_x0") is True
        and rollout.get("precision") == sampling_protocol.get("precision") == "bf16"
    )
    checks = {
        "schema": int(report.get("schema_version", 0)) >= 2,
        "status": report.get("status") == "pass",
        "weights": protocol.get("weights") == "ema",
        "checkpoint_step": int(protocol.get("checkpoint_step", -1))
        == expected_step,
        "checkpoint_images": int(protocol.get("checkpoint_evaluated_images", -1))
        == expected_checkpoint_images,
        "checkpoint_identity": (
            identity.get("cofitok_checkpoint_sha256") == expected_cofitok_sha256
            and identity.get("dense_checkpoint_sha256") == expected_dense_sha256
        ),
        "evaluation_identity": (
            (
                expected_evaluation_revision is None
                or identity.get("evaluation_git_revision")
                == expected_evaluation_revision
            )
            and identity.get("evaluation_git_branch")
            == expected_evaluation_branch
        ),
        "qualification_gates": bool(qualification_gates) and not failed_gates,
        "pair_contract": pair_contract_valid,
        "rollout_protocol": protocol_matches,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "weights": protocol.get("weights"),
        "checkpoint_step": protocol.get("checkpoint_step"),
        "checkpoint_evaluated_images": protocol.get("checkpoint_evaluated_images"),
        "cofitok_checkpoint_sha256": identity.get("cofitok_checkpoint_sha256"),
        "dense_checkpoint_sha256": identity.get("dense_checkpoint_sha256"),
        "evaluation_git_revision": identity.get("evaluation_git_revision"),
        "evaluation_git_branch": identity.get("evaluation_git_branch"),
        "failed_gates": failed_gates,
        "pair_contract_valid": pair_contract_valid,
        "rollout_protocol": rollout,
    }


def _class_fidelity_evidence(
    report: dict[str, Any],
    *,
    stage: str,
    expected_evaluation_revision: str | None,
    expected_evaluation_branch: str,
    expected_sampling_protocol: dict[str, Any],
    expected_cofitok_checkpoint_sha256: str,
    expected_dense_checkpoint_sha256: str,
    expected_cofitok_sample_set_sha256: str,
    expected_dense_sample_set_sha256: str,
) -> dict[str, Any]:
    evidence = validate_class_fidelity_qualification(
        report,
        expected_stage=stage,
        expected_revision=expected_evaluation_revision,
        expected_branch=expected_evaluation_branch,
        require_pass=False,
    )
    contract = evidence["sampling_contract"]
    checks = {
        "qualification_status": evidence["valid"] is True,
        "sampling_protocol": contract["sampling"] == expected_sampling_protocol,
        "cofitok_checkpoint": contract["cofitok_checkpoint_sha256"]
        == expected_cofitok_checkpoint_sha256,
        "dense_checkpoint": contract["dense_checkpoint_sha256"]
        == expected_dense_checkpoint_sha256,
        "cofitok_sample_set": contract["cofitok_sample_set_sha256"]
        == expected_cofitok_sample_set_sha256,
        "dense_sample_set": contract["dense_sample_set_sha256"]
        == expected_dense_sample_set_sha256,
    }
    return {
        **evidence,
        "valid": all(checks.values()),
        "checks": checks,
        "qualification": report,
    }


def _runtime_environment_identity(payload: dict[str, Any]) -> dict[str, Any]:
    environment = payload.get("runtime_environment")
    declared = payload.get("runtime_environment_sha256")
    if not isinstance(environment, dict):
        return {"valid": False, "sha256": declared}
    actual = runtime_environment_sha256(environment)
    return {
        "valid": declared == actual,
        "sha256": actual,
        "device": environment.get("device"),
        "torch": environment.get("torch"),
    }


def _real_set_identity(report: dict[str, Any]) -> dict[str, Any]:
    real_set = report.get("real_set", {})
    sha256 = str(real_set.get("sha256", ""))
    root = real_set.get("root")
    image_count = int(real_set.get("image_count", -1))
    cache_name = str(report.get("parameters", {}).get("real_cache_name", ""))
    valid_sha = len(sha256) == 64 and all(
        character in "0123456789abcdef" for character in sha256
    )
    return {
        "valid": (
            real_set.get("digest_schema") == IMAGE_TREE_DIGEST_SCHEMA
            and valid_sha
            and root == report.get("paths", {}).get("real_dir")
            and image_count == int(report.get("counts", {}).get("real_image_count", -2))
            and cache_name.endswith(f"__cofitok_{sha256[:16]}")
        ),
        "digest_schema": real_set.get("digest_schema"),
        "sha256": sha256,
        "root": root,
        "image_count": image_count,
        "real_cache_name": cache_name,
    }


def build_report(
    *,
    cofitok_training: dict[str, Any],
    dense_training: dict[str, Any],
    cofitok_generation: dict[str, Any],
    dense_generation: dict[str, Any],
    cofitok_checkpoint: dict[str, Any],
    dense_checkpoint: dict[str, Any],
    rollout_stability_qualification: dict[str, Any] | None = None,
    class_fidelity_qualification: dict[str, Any] | None = None,
    min_samples: int,
    max_fid_regression: float,
    max_endpoint_regression: float,
    stage: str = "scaling",
    max_absolute_fid: float = 100.0,
    min_coarse_token_energy_ratio: float = 0.05,
    min_precision: float = 0.30,
    min_recall: float = 0.30,
    max_precision_regression: float = 0.05,
    max_recall_regression: float = 0.05,
    require_stride_partition: bool = False,
    require_scaling_distribution_support: bool = False,
    expected_training_revision: str | None = None,
    expected_training_branch: str = "scale/generative-system",
    expected_evaluation_revision: str | None = None,
    expected_evaluation_branch: str = "scale/generative-system",
) -> dict[str, Any]:
    if stage not in {"scaling", "full"}:
        raise ValueError("stage must be scaling or full")
    if require_scaling_distribution_support and stage != "scaling":
        raise ValueError(
            "scaling distribution-support enforcement requires the scaling stage"
        )
    if not math.isfinite(max_absolute_fid) or max_absolute_fid <= 0.0:
        raise ValueError("max_absolute_fid must be finite and positive")
    if (
        not math.isfinite(min_coarse_token_energy_ratio)
        or not 0.0 <= min_coarse_token_energy_ratio <= 1.0
    ):
        raise ValueError("min_coarse_token_energy_ratio must be finite and in [0, 1]")
    quality_thresholds = {
        "min_precision": min_precision,
        "min_recall": min_recall,
        "max_precision_regression": max_precision_regression,
        "max_recall_regression": max_recall_regression,
    }
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in quality_thresholds.values()
    ):
        raise ValueError("precision/recall thresholds must be finite and in [0, 1]")
    for name, revision in (
        ("expected_training_revision", expected_training_revision),
        ("expected_evaluation_revision", expected_evaluation_revision),
    ):
        if revision is not None and (
            len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError(f"{name} must be a lowercase 40-character Git revision")
    if not expected_training_branch or not expected_evaluation_branch:
        raise ValueError("expected Git branches must be non-empty")
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
    coarse_utilization = _coarse_token_utilization(
        cofitok_checkpoint,
        require_stride_partition=require_stride_partition,
    )
    cofitok_checkpoint_sha = str(cofitok_checkpoint["checkpoint_sha256"])
    dense_checkpoint_sha = str(dense_checkpoint["checkpoint_sha256"])
    checkpoint_evaluator_git_pair = (
        cofitok_checkpoint.get("git", {}),
        dense_checkpoint.get("git", {}),
    )
    generated_counts = (
        int(cofitok_generation["counts"]["generated_image_count"]),
        int(dense_generation["counts"]["generated_image_count"]),
    )
    evaluator_pair = (
        cofitok_generation["implementation"],
        dense_generation["implementation"],
    )
    evaluator_git_pair = (
        cofitok_generation.get("git", {}),
        dense_generation.get("git", {}),
    )
    cofitok_provenance = cofitok_generation["sample_provenance"]
    dense_provenance = dense_generation["sample_provenance"]
    cofitok_sampling = _sampling_protocol(cofitok_provenance)
    dense_sampling = _sampling_protocol(dense_provenance)
    cofitok_sampling_git = cofitok_provenance.get("git", {})
    dense_sampling_git = dense_provenance.get("git", {})
    cofitok_sampling_environment = _runtime_environment_identity(cofitok_provenance)
    dense_sampling_environment = _runtime_environment_identity(dense_provenance)
    cofitok_evaluator_environment = _runtime_environment_identity(cofitok_generation)
    dense_evaluator_environment = _runtime_environment_identity(dense_generation)
    cofitok_real_set = _real_set_identity(cofitok_generation)
    dense_real_set = _real_set_identity(dense_generation)
    cofitok_model_config = cofitok_training["config"]["model"]
    dense_model_config = dense_training["config"]["model"]
    cofitok_sampling_contract = sampling_protocol_contract(
        cofitok_sampling,
        stage=stage,
        expected_num_train_timesteps=int(
            cofitok_training["config"]["diffusion"]["num_train_timesteps"]
        ),
    )
    dense_sampling_contract = sampling_protocol_contract(
        dense_sampling,
        stage=stage,
        expected_num_train_timesteps=int(
            dense_training["config"]["diffusion"]["num_train_timesteps"]
        ),
    )
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
    cofitok_revision = str(cofitok_training.get("git", {}).get("revision", ""))
    dense_revision = str(dense_training.get("git", {}).get("revision", ""))
    cofitok_branch = str(cofitok_training.get("git", {}).get("branch", ""))
    dense_branch = str(dense_training.get("git", {}).get("branch", ""))
    required_evaluation_revision = expected_evaluation_revision
    if required_evaluation_revision is None and stage == "full":
        required_evaluation_revision = cofitok_revision

    def evaluation_git_matches(identity: dict[str, Any]) -> bool:
        revision = str(identity.get("revision", ""))
        return (
            len(revision) == 40
            and identity.get("branch") == expected_evaluation_branch
            and identity.get("tracked_dirty") is False
            and (
                required_evaluation_revision is None
                or revision == required_evaluation_revision
            )
        )

    cofitok_cost = training_cost_summary(cofitok_training)
    dense_cost = training_cost_summary(dense_training)
    pair_contract = generation_pair_contract(
        cofitok_training["config"], dense_training["config"]
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
            "matched_training_revision",
            len(cofitok_revision) == 40
            and cofitok_revision == dense_revision
            and cofitok_branch == dense_branch == expected_training_branch
            and (
                expected_training_revision is None
                or cofitok_revision == expected_training_revision
            ),
            {
                "cofitok_revision": cofitok_revision,
                "dense_revision": dense_revision,
                "cofitok_branch": cofitok_branch,
                "dense_branch": dense_branch,
                "expected_revision": expected_training_revision,
                "expected_branch": expected_training_branch,
            },
        ),
        _gate(
            "matched_training_protocol",
            pair_contract["valid"] and abs(parameter_gap) <= 0.02,
            {"pair_contract": pair_contract, "relative_parameter_gap": parameter_gap},
        ),
        _gate(
            "training_cost_accounting",
            cofitok_cost["valid"]
            and dense_cost["valid"]
            and cofitok_cost["expected_samples_seen"]
            == dense_cost["expected_samples_seen"],
            {"cofitok": cofitok_cost, "dense": dense_cost},
        ),
        _gate(
            "matched_generation_protocol",
            cofitok_generation.get("status") == dense_generation.get("status") == "completed"
            and cofitok_generation.get("protocol")
            == dense_generation.get("protocol")
            == "torch_fidelity_directory_metrics"
            and cofitok_generation.get("paths", {}).get("real_dir")
            == dense_generation.get("paths", {}).get("real_dir")
            and int(cofitok_generation["counts"]["real_image_count"])
            == int(dense_generation["counts"]["real_image_count"])
            and int(cofitok_generation["counts"]["real_image_count"]) >= min_samples
            and generated_counts == (min_samples, min_samples)
            and evaluator_pair[0] == evaluator_pair[1]
            and cofitok_generation["parameters"] == dense_generation["parameters"],
            {
                "generated_counts": list(generated_counts),
                "real_counts": [
                    cofitok_generation["counts"]["real_image_count"],
                    dense_generation["counts"]["real_image_count"],
                ],
                "real_dirs": [
                    cofitok_generation.get("paths", {}).get("real_dir"),
                    dense_generation.get("paths", {}).get("real_dir"),
                ],
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
            "matched_real_set_provenance",
            cofitok_real_set["valid"]
            and dense_real_set["valid"]
            and cofitok_real_set == dense_real_set,
            {
                "cofitok": cofitok_real_set,
                "dense_identity": dense_real_set,
            },
        ),
        _gate(
            "matched_evaluator_code_provenance",
            evaluator_git_pair[0] == evaluator_git_pair[1]
            and evaluation_git_matches(evaluator_git_pair[0]),
            {
                "stage": stage,
                "cofitok": evaluator_git_pair[0],
                "dense_identity": evaluator_git_pair[1],
                "expected_revision": required_evaluation_revision,
                "expected_branch": expected_evaluation_branch,
            },
        ),
        _gate(
            "matched_evaluator_runtime_environment",
            cofitok_evaluator_environment["valid"]
            and dense_evaluator_environment["valid"]
            and cofitok_evaluator_environment["sha256"]
            == dense_evaluator_environment["sha256"],
            {
                "cofitok": cofitok_evaluator_environment,
                "dense_identity": dense_evaluator_environment,
            },
        ),
        _gate(
            "distribution_metric_ranges",
            _distribution_metrics_valid(cofitok_quality)
            and _distribution_metrics_valid(dense_quality),
            {
                "cofitok": cofitok_quality,
                "dense": dense_quality,
                "valid_ranges": {
                    "fid": ">= 0",
                    "inception_score_mean": "> 0",
                    "inception_score_std": ">= 0",
                    "precision": "[0, 1]",
                    "recall": "[0, 1]",
                },
            },
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
            and str(cofitok_provenance.get("checkpoint_integrity_manifest", "")).endswith(
                ".pt.integrity.json"
            )
            and str(dense_provenance.get("checkpoint_integrity_manifest", "")).endswith(
                ".pt.integrity.json"
            )
            and len(str(cofitok_provenance.get("sample_set_sha256", ""))) == 64
            and len(str(dense_provenance.get("sample_set_sha256", ""))) == 64
            and cofitok_sampling.get("random_stream", {}).get("prefix_budgets_share_stream")
            is True
            and cofitok_sampling.get("random_stream", {}).get("batch_size_invariant") is True
            and cofitok_sampling.get("random_stream", {}).get("resume_index_invariant") is True
            and int(cofitok_sampling.get("start_index", -1)) == 0
            and int(dense_sampling.get("start_index", -1)) == 0
            and int(cofitok_sampling.get("num_samples", -1)) == min_samples
            and int(dense_sampling.get("num_samples", -1)) == min_samples
            and cofitok_provenance.get("sampling_progress", {}).get("status") == "completed"
            and dense_provenance.get("sampling_progress", {}).get("status") == "completed"
            and int(
                cofitok_provenance.get("sampling_progress", {}).get(
                    "completed_samples", -1
                )
            )
            == min_samples
            and int(
                dense_provenance.get("sampling_progress", {}).get(
                    "completed_samples", -1
                )
            )
            == min_samples
            and _positive_finite(
                cofitok_provenance.get("sampling_progress", {}).get(
                    "cumulative_elapsed_seconds"
                )
            )
            and _positive_finite(
                dense_provenance.get("sampling_progress", {}).get(
                    "cumulative_elapsed_seconds"
                )
            )
            and cofitok_sampling.get("class_schedule")
            == dense_sampling.get("class_schedule")
            == "balanced_modulo"
            and cofitok_sampling.get("image_shape") == cofitok_expected_shape
            and dense_sampling.get("image_shape") == dense_expected_shape,
            {
                "protocols_match": cofitok_sampling == dense_sampling,
                "cofitok_checkpoint_step": cofitok_provenance["checkpoint_step"],
                "dense_checkpoint_step": dense_provenance["checkpoint_step"],
                "cofitok_checkpoint_sha256": cofitok_provenance["checkpoint_sha256"],
                "dense_checkpoint_sha256": dense_provenance["checkpoint_sha256"],
                "cofitok_checkpoint_integrity_manifest": cofitok_provenance.get(
                    "checkpoint_integrity_manifest"
                ),
                "dense_checkpoint_integrity_manifest": dense_provenance.get(
                    "checkpoint_integrity_manifest"
                ),
                "cofitok_prefix_budget": cofitok_provenance["selected_prefix_budget"],
                "dense_prefix_budget": dense_provenance["selected_prefix_budget"],
                "cofitok_image_shape": cofitok_sampling.get("image_shape"),
                "dense_image_shape": dense_sampling.get("image_shape"),
                "cofitok_sample_set_sha256": cofitok_provenance.get("sample_set_sha256"),
                "dense_sample_set_sha256": dense_provenance.get("sample_set_sha256"),
                "cofitok_sampling_progress": cofitok_provenance.get("sampling_progress"),
                "dense_sampling_progress": dense_provenance.get("sampling_progress"),
            },
        ),
        _gate(
            "formal_sampling_protocol",
            cofitok_sampling_contract["valid"] is True
            and dense_sampling_contract["valid"] is True,
            {
                "stage": stage,
                "cofitok": cofitok_sampling_contract,
                "dense_identity": dense_sampling_contract,
            },
        ),
        _gate(
            "matched_sampling_code_provenance",
            cofitok_sampling_git == dense_sampling_git
            and evaluation_git_matches(cofitok_sampling_git),
            {
                "stage": stage,
                "cofitok": cofitok_sampling_git,
                "dense_identity": dense_sampling_git,
                "expected_revision": required_evaluation_revision,
                "expected_branch": expected_evaluation_branch,
            },
        ),
        _gate(
            "matched_sampling_runtime_environment",
            cofitok_sampling_environment["valid"]
            and dense_sampling_environment["valid"]
            and cofitok_sampling_environment["sha256"]
            == dense_sampling_environment["sha256"],
            {
                "cofitok": cofitok_sampling_environment,
                "dense_identity": dense_sampling_environment,
            },
        ),
        _gate(
            "checkpoint_evaluation_provenance",
            int(cofitok_checkpoint["checkpoint_step"])
            == int(cofitok_provenance["checkpoint_step"])
            and int(dense_checkpoint["checkpoint_step"])
            == int(dense_provenance["checkpoint_step"])
            and cofitok_checkpoint_sha == cofitok_provenance["checkpoint_sha256"]
            and dense_checkpoint_sha == dense_provenance["checkpoint_sha256"]
            and cofitok_checkpoint.get("checkpoint_integrity_manifest")
            == cofitok_provenance.get("checkpoint_integrity_manifest")
            and dense_checkpoint.get("checkpoint_integrity_manifest")
            == dense_provenance.get("checkpoint_integrity_manifest"),
            {
                "cofitok_hash_matches": cofitok_checkpoint_sha
                == cofitok_provenance["checkpoint_sha256"],
                "dense_hash_matches": dense_checkpoint_sha
                == dense_provenance["checkpoint_sha256"],
                "cofitok_integrity_manifest_matches": cofitok_checkpoint.get(
                    "checkpoint_integrity_manifest"
                )
                == cofitok_provenance.get("checkpoint_integrity_manifest"),
                "dense_integrity_manifest_matches": dense_checkpoint.get(
                    "checkpoint_integrity_manifest"
                )
                == dense_provenance.get("checkpoint_integrity_manifest"),
                "cofitok_step": cofitok_checkpoint["checkpoint_step"],
                "dense_step": dense_checkpoint["checkpoint_step"],
            },
        ),
        _gate(
            "matched_checkpoint_evaluator_code_provenance",
            checkpoint_evaluator_git_pair[0] == checkpoint_evaluator_git_pair[1]
            and evaluation_git_matches(checkpoint_evaluator_git_pair[0]),
            {
                "stage": stage,
                "cofitok": checkpoint_evaluator_git_pair[0],
                "dense_identity": checkpoint_evaluator_git_pair[1],
                "expected_revision": required_evaluation_revision,
                "expected_branch": expected_evaluation_branch,
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
            "full_precision_recall_quality",
            stage != "full"
            or (
                cofitok_quality["precision"] is not None
                and dense_quality["precision"] is not None
                and cofitok_quality["recall"] is not None
                and dense_quality["recall"] is not None
                and cofitok_quality["precision"] >= min_precision
                and cofitok_quality["recall"] >= min_recall
                and cofitok_quality["precision"]
                >= dense_quality["precision"] - max_precision_regression
                and cofitok_quality["recall"]
                >= dense_quality["recall"] - max_recall_regression
            ),
            {
                "enforced": stage == "full",
                "cofitok_precision": cofitok_quality["precision"],
                "dense_precision": dense_quality["precision"],
                "cofitok_recall": cofitok_quality["recall"],
                "dense_recall": dense_quality["recall"],
                **quality_thresholds,
            },
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
            "coarse_token_utilization",
            coarse_utilization["valid"]
            and float(coarse_utilization["coarse_token_energy_ratio"])
            >= min_coarse_token_energy_ratio,
            {
                **coarse_utilization,
                "min_coarse_token_energy_ratio": min_coarse_token_energy_ratio,
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
    if require_scaling_distribution_support:
        gates.append(
            _gate(
                "scaling_precision_recall_quality",
                cofitok_quality["precision"] is not None
                and dense_quality["precision"] is not None
                and cofitok_quality["recall"] is not None
                and dense_quality["recall"] is not None
                and cofitok_quality["precision"] >= min_precision
                and cofitok_quality["recall"] >= min_recall
                and cofitok_quality["precision"]
                >= dense_quality["precision"] - max_precision_regression
                and cofitok_quality["recall"]
                >= dense_quality["recall"] - max_recall_regression,
                {
                    "enforced": True,
                    "cofitok_precision": cofitok_quality["precision"],
                    "dense_precision": dense_quality["precision"],
                    "cofitok_recall": cofitok_quality["recall"],
                    "dense_recall": dense_quality["recall"],
                    **quality_thresholds,
                },
            )
        )
    if rollout_stability_qualification is not None:
        rollout_stability = _rollout_stability_evidence(
            rollout_stability_qualification,
            stage=stage,
            expected_step=int(cofitok_training["target_steps"]),
            expected_checkpoint_images=int(
                cofitok_checkpoint.get("metrics", {}).get("evaluated_images", -1)
            ),
            expected_cofitok_sha256=cofitok_checkpoint_sha,
            expected_dense_sha256=dense_checkpoint_sha,
            expected_evaluation_revision=required_evaluation_revision,
            expected_evaluation_branch=expected_evaluation_branch,
            sampling_protocol=cofitok_sampling,
        )
        gates.append(
            _gate(
                "rollout_stability_diagnostic",
                rollout_stability["valid"],
                rollout_stability,
            )
        )
    if class_fidelity_qualification is not None:
        class_fidelity = _class_fidelity_evidence(
            class_fidelity_qualification,
            stage=stage,
            expected_evaluation_revision=required_evaluation_revision,
            expected_evaluation_branch=expected_evaluation_branch,
            expected_sampling_protocol=cofitok_sampling,
            expected_cofitok_checkpoint_sha256=cofitok_checkpoint_sha,
            expected_dense_checkpoint_sha256=dense_checkpoint_sha,
            expected_cofitok_sample_set_sha256=str(
                cofitok_provenance["sample_set_sha256"]
            ),
            expected_dense_sample_set_sha256=str(
                dense_provenance["sample_set_sha256"]
            ),
        )
        gates.append(
            _gate(
                "class_conditional_fidelity",
                class_fidelity["valid"],
                class_fidelity,
            )
        )
    passed = all(gate["passed"] for gate in gates)
    pass_decision = (
        "promote_to_full_imagenet256"
        if stage == "scaling"
        else "large_scale_generation_ready"
    )
    return {
        "schema_version": GENERATION_GATE_SCHEMA_VERSION,
        "stage": stage,
        "status": "pass" if passed else "fail",
        "decision": pass_decision if passed else "hold",
        "provenance_contract": {
            "training_revision": expected_training_revision,
            "training_branch": expected_training_branch,
            "evaluation_revision": required_evaluation_revision,
            "evaluation_branch": expected_evaluation_branch,
        },
        "thresholds": {
            "min_samples": min_samples,
            "max_fid_regression": max_fid_regression,
            "max_absolute_fid": max_absolute_fid,
            "max_endpoint_regression": max_endpoint_regression,
            "min_coarse_token_energy_ratio": min_coarse_token_energy_ratio,
            **quality_thresholds,
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
            "coarse_token_energy_ratio": coarse_utilization[
                "coarse_token_energy_ratio"
            ],
            "cofitok_training_cost": cofitok_cost,
            "dense_training_cost": dense_cost,
        },
    }


def main() -> None:
    args = parse_args()
    source_profile = args.source_profile or args.stage
    class_fidelity_qualification = (
        _read(args.class_fidelity_qualification)
        if args.class_fidelity_qualification
        else None
    )
    stability_profiles = {"stability_scaling", "stability_full", "capacity_full"}
    if source_profile in stability_profiles:
        if not args.rollout_stability_qualification:
            raise ValueError(
                "schema-v5 stability gates require rollout-stability qualification"
            )
        if class_fidelity_qualification is None:
            raise ValueError(
                "schema-v5 stability gates require class-fidelity qualification"
            )
    report = build_report(
        cofitok_training=_read(args.cofitok_training),
        dense_training=_read(args.dense_training),
        cofitok_generation=_read(args.cofitok_generation),
        dense_generation=_read(args.dense_generation),
        cofitok_checkpoint=_read(args.cofitok_checkpoint_eval),
        dense_checkpoint=_read(args.dense_checkpoint_eval),
        rollout_stability_qualification=(
            _read(args.rollout_stability_qualification)
            if args.rollout_stability_qualification
            else None
        ),
        class_fidelity_qualification=class_fidelity_qualification,
        min_samples=args.min_samples,
        max_fid_regression=args.max_fid_regression,
        max_endpoint_regression=args.max_endpoint_regression,
        stage=args.stage,
        max_absolute_fid=args.max_absolute_fid,
        min_coarse_token_energy_ratio=args.min_coarse_token_energy_ratio,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_precision_regression=args.max_precision_regression,
        max_recall_regression=args.max_recall_regression,
        require_stride_partition=args.source_profile == "stability_scaling",
        require_scaling_distribution_support=(
            source_profile == "stability_scaling"
        ),
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
        expected_evaluation_revision=args.expected_evaluation_revision,
        expected_evaluation_branch=args.expected_evaluation_branch,
    )
    report["source_reports"] = build_generation_gate_source_reports(
        stage=args.stage,
        profile=args.source_profile,
        paths={
            "cofitok_training": args.cofitok_training,
            "dense_training": args.dense_training,
            "cofitok_generation": args.cofitok_generation,
            "dense_generation": args.dense_generation,
            "cofitok_checkpoint_eval": args.cofitok_checkpoint_eval,
            "dense_checkpoint_eval": args.dense_checkpoint_eval,
        },
    )
    report["source_profile"] = source_profile
    if args.rollout_stability_qualification:
        if report["source_profile"] not in stability_profiles:
            raise ValueError(
                "rollout-stability diagnostics require a stability source profile"
            )
        diagnostic_paths = {
            "rollout_stability_qualification": (
                args.rollout_stability_qualification
            )
        }
        if class_fidelity_qualification is not None:
            diagnostic_paths.update(
                {
                    "cofitok_class_fidelity": class_fidelity_qualification[
                        "sources"
                    ]["cofitok"]["path"],
                    "dense_class_fidelity": class_fidelity_qualification[
                        "sources"
                    ]["dense_identity"]["path"],
                    "class_fidelity_qualification": (
                        args.class_fidelity_qualification
                    ),
                }
            )
        report["diagnostic_reports"] = build_generation_gate_diagnostic_reports(
            profile=report["source_profile"],
            paths=diagnostic_paths,
            schema_version=GENERATION_GATE_SCHEMA_VERSION,
        )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")
    if report["status"] != "pass" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
