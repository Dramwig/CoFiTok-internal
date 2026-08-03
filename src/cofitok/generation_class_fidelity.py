from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256


CLASS_FIDELITY_REPORT_SCHEMA_VERSION = 1
CLASS_FIDELITY_REPORT_ROLE = "generation_class_fidelity_report"
CLASS_FIDELITY_QUALIFICATION_SCHEMA_VERSION = 1
CLASS_FIDELITY_QUALIFICATION_ROLE = "generation_class_fidelity_qualification"
CLASS_FIDELITY_CLASSIFIER_NAME = "torchvision_resnet50_imagenet1k_v2"
CLASS_FIDELITY_CLASSIFIER_BYTES = 102540417
CLASS_FIDELITY_CLASSIFIER_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
CLASS_FIDELITY_CATEGORIES_SHA256 = (
    "62fff941ecff3f19de9128c6ca9c2097807c6b7bafc5552d7589a219431221ed"
)
CLASS_FIDELITY_PREPROCESSING = {
    "resize_size": [232],
    "crop_size": [224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "interpolation": "bilinear",
    "antialias": True,
}
CLASS_FIDELITY_SAMPLING_PROTOCOL_SCHEMA = "cofitok_ddim_sampling_v1"
CLASS_FIDELITY_INFERENCE_API = {
    "name": "cofitok.generation.GenerationSession",
    "version": 1,
}
CLASS_FIDELITY_STAGE_REQUIREMENTS = {
    "scaling": {
        "min_samples": 10_000,
        "sample_steps": 100,
        "min_top1": 0.01,
        "min_top5": 0.05,
        "min_predicted_class_fraction": 0.25,
        "min_normalized_predicted_entropy": 0.50,
        "max_top1_regression": 0.05,
        "max_top5_regression": 0.05,
    },
    "full": {
        "min_samples": 50_000,
        "sample_steps": 250,
        "min_top1": 0.10,
        "min_top5": 0.25,
        "min_predicted_class_fraction": 0.50,
        "min_normalized_predicted_entropy": 0.70,
        "max_top1_regression": 0.05,
        "max_top5_regression": 0.05,
    },
}


def _formal_sampling_protocol_is_valid(
    sampling: dict[str, Any],
    *,
    stage: str,
) -> bool:
    requirements = CLASS_FIDELITY_STAGE_REQUIREMENTS[stage]
    expected = {
        "protocol_schema": CLASS_FIDELITY_SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": CLASS_FIDELITY_INFERENCE_API,
        "sampler": "ddim",
        "num_samples": int(requirements["min_samples"]),
        "start_index": 0,
        "sample_steps": int(requirements["sample_steps"]),
        "num_train_timesteps": 1000,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "class_schedule": "balanced_modulo",
    }
    if any(sampling.get(name) != value for name, value in expected.items()):
        return False
    return sampling.get("actual_timesteps") == select_sampling_timesteps(
        1000,
        int(requirements["sample_steps"]),
    )


@dataclass
class ClassFidelityAccumulator:
    """Streaming class-conditional fidelity statistics for ImageNet samples."""

    num_classes: int
    sample_count: int = 0
    top1_correct: int = 0
    top5_correct: int = 0
    target_probability_sum: float = 0.0
    target_log_probability_sum: float = 0.0
    requested_histogram: torch.Tensor = field(init=False)
    predicted_histogram: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes < 5:
            raise ValueError("class-fidelity evaluation requires at least five classes")
        self.requested_histogram = torch.zeros(self.num_classes, dtype=torch.int64)
        self.predicted_histogram = torch.zeros(self.num_classes, dtype=torch.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError("classifier logits have the wrong class dimension")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("classifier targets do not match the batch")
        if logits.shape[0] < 1 or not torch.isfinite(logits).all():
            raise ValueError("classifier logits must be nonempty and finite")
        targets = targets.to(device=logits.device, dtype=torch.long)
        if torch.any(targets < 0) or torch.any(targets >= self.num_classes):
            raise ValueError("classifier targets are outside the class range")

        log_probabilities = torch.log_softmax(logits.float(), dim=1)
        probabilities = log_probabilities.exp()
        top5 = torch.topk(logits, k=5, dim=1).indices
        top1 = top5[:, 0]
        row_indices = torch.arange(logits.shape[0], device=logits.device)
        target_log_probabilities = log_probabilities[row_indices, targets]

        self.sample_count += int(logits.shape[0])
        self.top1_correct += int((top1 == targets).sum().item())
        self.top5_correct += int((top5 == targets[:, None]).any(dim=1).sum().item())
        self.target_probability_sum += float(
            probabilities[row_indices, targets].sum().item()
        )
        self.target_log_probability_sum += float(target_log_probabilities.sum().item())
        self.requested_histogram += torch.bincount(
            targets.detach().cpu(), minlength=self.num_classes
        )
        self.predicted_histogram += torch.bincount(
            top1.detach().cpu(), minlength=self.num_classes
        )

    def finalize(self) -> dict[str, Any]:
        if self.sample_count < 1:
            raise ValueError("class-fidelity evaluation has no samples")
        if int(self.requested_histogram.sum().item()) != self.sample_count:
            raise ValueError("requested-class accounting differs from sample count")
        if int(self.predicted_histogram.sum().item()) != self.sample_count:
            raise ValueError("predicted-class accounting differs from sample count")

        predicted = self.predicted_histogram.to(torch.float64)
        nonzero = predicted[predicted > 0]
        probabilities = nonzero / float(self.sample_count)
        entropy = float(-(probabilities * probabilities.log()).sum().item())
        normalized_entropy = entropy / math.log(self.num_classes)
        result = {
            "sample_count": self.sample_count,
            "num_classes": self.num_classes,
            "top1_correct": self.top1_correct,
            "top5_correct": self.top5_correct,
            "top1_accuracy": self.top1_correct / self.sample_count,
            "top5_accuracy": self.top5_correct / self.sample_count,
            "mean_target_probability": self.target_probability_sum / self.sample_count,
            "target_negative_log_likelihood": (
                -self.target_log_probability_sum / self.sample_count
            ),
            "requested_class_count": int(
                (self.requested_histogram > 0).sum().item()
            ),
            "requested_count_min": int(self.requested_histogram.min().item()),
            "requested_count_max": int(self.requested_histogram.max().item()),
            "predicted_class_count": int(
                (self.predicted_histogram > 0).sum().item()
            ),
            "predicted_class_fraction": float(
                (self.predicted_histogram > 0).sum().item() / self.num_classes
            ),
            "predicted_class_entropy": entropy,
            "normalized_predicted_class_entropy": normalized_entropy,
        }
        for name, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"class-fidelity metric {name} is not finite")
        return result


def validate_class_fidelity_metrics(metrics: dict[str, Any]) -> None:
    try:
        sample_count = int(metrics["sample_count"])
        num_classes = int(metrics["num_classes"])
        top1_correct = int(metrics["top1_correct"])
        top5_correct = int(metrics["top5_correct"])
        requested_class_count = int(metrics["requested_class_count"])
        predicted_class_count = int(metrics["predicted_class_count"])
        requested_min = int(metrics["requested_count_min"])
        requested_max = int(metrics["requested_count_max"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("class-fidelity integer metrics are malformed") from error
    if (
        sample_count < 1
        or num_classes < 5
        or not 0 <= top1_correct <= top5_correct <= sample_count
        or not 1 <= requested_class_count <= num_classes
        or not 1 <= predicted_class_count <= num_classes
        or requested_min < 0
        or requested_max < requested_min
    ):
        raise ValueError("class-fidelity integer metrics are outside their domains")
    if (
        requested_class_count != num_classes
        or requested_max - requested_min > 1
        or requested_min * num_classes > sample_count
        or requested_max * num_classes < sample_count
    ):
        raise ValueError("class-fidelity requested classes are not balanced")

    bounded = (
        "top1_accuracy",
        "top5_accuracy",
        "mean_target_probability",
        "predicted_class_fraction",
        "normalized_predicted_class_entropy",
    )
    for name in bounded:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"class-fidelity metric {name} is malformed") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"class-fidelity metric {name} is outside [0, 1]")
    try:
        nll = float(metrics["target_negative_log_likelihood"])
        entropy = float(metrics["predicted_class_entropy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("class-fidelity loss/entropy metrics are malformed") from error
    if not math.isfinite(nll) or nll < 0.0 or not math.isfinite(entropy) or entropy < 0.0:
        raise ValueError("class-fidelity loss/entropy metrics are outside their domains")
    if entropy > math.log(num_classes) + 1e-12:
        raise ValueError("class-fidelity entropy exceeds the class support")
    expected_values = {
        "top1_accuracy": top1_correct / sample_count,
        "top5_accuracy": top5_correct / sample_count,
        "predicted_class_fraction": predicted_class_count / num_classes,
        "normalized_predicted_class_entropy": entropy / math.log(num_classes),
    }
    for name, expected in expected_values.items():
        if not math.isclose(
            float(metrics[name]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"class-fidelity metric {name} is internally inconsistent")


def _validate_classifier_identity(classifier: dict[str, Any]) -> None:
    if (
        classifier.get("name") != CLASS_FIDELITY_CLASSIFIER_NAME
        or classifier.get("weights_enum")
        != "ResNet50_Weights.IMAGENET1K_V2"
        or int(classifier.get("weights_bytes", -1))
        != CLASS_FIDELITY_CLASSIFIER_BYTES
        or classifier.get("weights_sha256")
        != CLASS_FIDELITY_CLASSIFIER_SHA256
        or int(classifier.get("num_classes", -1)) != 1000
        or classifier.get("categories_sha256")
        != CLASS_FIDELITY_CATEGORIES_SHA256
        or classifier.get("preprocessing") != CLASS_FIDELITY_PREPROCESSING
    ):
        raise ValueError("class-fidelity classifier contract differs")


def validate_class_fidelity_report(report: dict[str, Any]) -> None:
    if (
        report.get("schema_version") != CLASS_FIDELITY_REPORT_SCHEMA_VERSION
        or report.get("role") != CLASS_FIDELITY_REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("protocol") != "torchvision_imagenet_class_fidelity"
    ):
        raise ValueError("class-fidelity report contract differs")
    sample = report.get("sample_provenance")
    classifier = report.get("classifier")
    parameters = report.get("parameters")
    if not isinstance(sample, dict) or not isinstance(classifier, dict):
        raise ValueError("class-fidelity source provenance is missing")
    if not isinstance(parameters, dict):
        raise ValueError("class-fidelity parameters are missing")
    sampling = sample.get("sampling")
    git = report.get("git")
    environment = report.get("runtime_environment")
    if not isinstance(sampling, dict) or not isinstance(git, dict):
        raise ValueError("class-fidelity sampling or Git provenance is missing")
    if not isinstance(environment, dict):
        raise ValueError("class-fidelity runtime environment is missing")
    selected_budget = sample.get("selected_prefix_budget")
    if (
        sample.get("weights") != "ema"
        or sampling.get("class_schedule") != "balanced_modulo"
        or sampling.get("prefix_budgets") != [selected_budget]
        or selected_budget not in {1, 8}
        or len(str(sample.get("sample_set_sha256", ""))) != 64
        or len(str(sample.get("checkpoint_sha256", ""))) != 64
        or git.get("tracked_dirty") is not False
        or sample.get("git") != git
        or report.get("runtime_environment_sha256")
        != runtime_environment_sha256(environment)
        or int(parameters.get("num_classes", -1)) != 1000
        or parameters.get("target_from_filename")
        != "int(zero_based_png_stem) mod 1000"
    ):
        raise ValueError("class-fidelity source contract differs")
    _validate_classifier_identity(classifier)
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("class-fidelity metrics are missing")
    validate_class_fidelity_metrics(metrics)
    if int(metrics["sample_count"]) != int(sample["sampling"]["num_samples"]):
        raise ValueError("class-fidelity count differs from sampling provenance")
    if int(metrics["sample_count"]) != int(parameters.get("sample_count", -1)):
        raise ValueError("class-fidelity count differs from evaluator parameters")


def _finite_fraction(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"class-fidelity {name} is malformed") from error
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"class-fidelity {name} is not a finite fraction")
    return numeric


def _finite_number(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"class-fidelity {name} is malformed") from error
    if not math.isfinite(numeric):
        raise ValueError(f"class-fidelity {name} is not finite")
    return numeric


def validate_class_fidelity_qualification(
    report: dict[str, Any],
    *,
    expected_stage: str,
    expected_revision: str | None = None,
    expected_branch: str | None = None,
    require_pass: bool = True,
) -> dict[str, Any]:
    requirements = CLASS_FIDELITY_STAGE_REQUIREMENTS.get(expected_stage)
    if requirements is None:
        raise ValueError("unsupported class-fidelity qualification stage")
    status = report.get("status")
    if (
        report.get("schema_version")
        != CLASS_FIDELITY_QUALIFICATION_SCHEMA_VERSION
        or report.get("role") != CLASS_FIDELITY_QUALIFICATION_ROLE
        or status not in {"pass", "hold"}
        or report.get("stage") != expected_stage
    ):
        raise ValueError("class-fidelity qualification contract differs")
    git = report.get("git")
    if not isinstance(git, dict) or git.get("tracked_dirty") is not False:
        raise ValueError("class-fidelity qualification Git provenance is invalid")
    if expected_revision is not None and git.get("revision") != expected_revision:
        raise ValueError("class-fidelity qualification revision differs")
    if expected_branch is not None and git.get("branch") != expected_branch:
        raise ValueError("class-fidelity qualification branch differs")
    classifier = report.get("classifier")
    if not isinstance(classifier, dict):
        raise ValueError("class-fidelity qualification classifier is missing")
    _validate_classifier_identity(classifier)

    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("class-fidelity qualification thresholds are missing")
    threshold_values: dict[str, float] = {}
    for name in (
        "min_top1",
        "min_top5",
        "min_predicted_class_fraction",
        "min_normalized_predicted_entropy",
        "max_top1_regression",
        "max_top5_regression",
    ):
        value = _finite_fraction(thresholds.get(name), name=f"threshold {name}")
        required = float(requirements[name])
        if (name.startswith("min_") and value < required) or (
            name.startswith("max_") and value > required
        ):
            raise ValueError(f"class-fidelity threshold {name} is weaker than required")
        threshold_values[name] = value

    sampling_contract = report.get("sampling_contract")
    if not isinstance(sampling_contract, dict):
        raise ValueError("class-fidelity sampling contract is missing")
    sampling = sampling_contract.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("class-fidelity sampling protocol is missing")
    sample_count = int(sampling_contract.get("sample_count_per_method", -1))
    random_stream = sampling.get("random_stream")
    formal_sampling = _formal_sampling_protocol_is_valid(
        sampling,
        stage=expected_stage,
    )
    identity_fields = (
        "cofitok_checkpoint_sha256",
        "dense_checkpoint_sha256",
        "cofitok_sample_set_sha256",
        "dense_sample_set_sha256",
    )
    if (
        formal_sampling is not True
        or sample_count < int(requirements["min_samples"])
        or int(sampling.get("num_samples", -1)) != sample_count
        or sampling_contract.get("cofitok_prefix_budget") != 8
        or sampling_contract.get("dense_prefix_budget") != 1
        or sampling_contract.get("weights") != "ema"
        or sampling.get("sampler") != "ddim"
        or int(sampling.get("sample_steps", -1))
        != int(requirements["sample_steps"])
        or int(sampling.get("num_train_timesteps", -1)) != 1000
        or sampling.get("image_shape") != [3, 256, 256]
        or sampling.get("class_schedule") != "balanced_modulo"
        or sampling.get("guidance_scale") != 1.5
        or sampling.get("guidance_rescale") != 0.0
        or sampling.get("cfg_batch_mode") != "batched"
        or sampling.get("eta") != 0.0
        or sampling.get("clip_x0") is not True
        or sampling.get("precision") != "bf16"
        or not isinstance(random_stream, dict)
        or random_stream.get("prefix_budgets_share_stream") is not True
        or random_stream.get("batch_size_invariant") is not True
        or random_stream.get("resume_index_invariant") is not True
        or any(
            len(str(sampling_contract.get(name, ""))) != 64
            for name in identity_fields
        )
    ):
        raise ValueError("class-fidelity sampling protocol is not formal")

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("class-fidelity qualification metrics are missing")
    cofitok = metrics.get("cofitok")
    dense = metrics.get("dense_identity")
    if not isinstance(cofitok, dict) or not isinstance(dense, dict):
        raise ValueError("class-fidelity paired metrics are missing")
    validate_class_fidelity_metrics(cofitok)
    validate_class_fidelity_metrics(dense)
    if (
        int(cofitok["sample_count"]) != sample_count
        or int(dense["sample_count"]) != sample_count
    ):
        raise ValueError("class-fidelity paired metric counts differ")
    observed_delta = report.get("metrics", {}).get("cofitok_minus_dense")
    delta_fields = (
        "top1_accuracy",
        "top5_accuracy",
        "mean_target_probability",
        "predicted_class_fraction",
        "normalized_predicted_class_entropy",
    )
    if not isinstance(observed_delta, dict) or set(observed_delta) != set(delta_fields):
        raise ValueError("class-fidelity paired metric delta is incomplete")
    for name in delta_fields:
        expected_delta = float(cofitok[name]) - float(dense[name])
        if not math.isclose(
            _finite_number(
                observed_delta.get(name),
                name=f"paired metric delta {name}",
            ),
            expected_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"class-fidelity paired metric delta differs: {name}")

    expected_checks: dict[str, tuple[float, float, str, bool]] = {}
    for method, values in (("cofitok", cofitok), ("dense_identity", dense)):
        for metric_name, threshold_name in (
            ("top1_accuracy", "min_top1"),
            ("top5_accuracy", "min_top5"),
            ("predicted_class_fraction", "min_predicted_class_fraction"),
            (
                "normalized_predicted_class_entropy",
                "min_normalized_predicted_entropy",
            ),
        ):
            observed = float(values[metric_name])
            threshold = threshold_values[threshold_name]
            expected_checks[f"{method}_{metric_name}"] = (
                observed,
                threshold,
                ">=",
                observed >= threshold,
            )
    for metric_name, threshold_name in (
        ("top1_accuracy", "max_top1_regression"),
        ("top5_accuracy", "max_top5_regression"),
    ):
        observed = float(dense[metric_name]) - float(cofitok[metric_name])
        threshold = threshold_values[threshold_name]
        expected_checks[f"cofitok_{metric_name}_regression_vs_dense"] = (
            observed,
            threshold,
            "<=",
            observed <= threshold,
        )
    rows = report.get("checks")
    if not isinstance(rows, list) or len(rows) != len(expected_checks):
        raise ValueError("class-fidelity qualification checks are incomplete")
    indexed = {
        str(row.get("name", "")): row for row in rows if isinstance(row, dict)
    }
    if set(indexed) != set(expected_checks):
        raise ValueError("class-fidelity qualification check names differ")
    for name, (observed, threshold, comparison, passed) in expected_checks.items():
        row = indexed[name]
        if (
            row.get("status") != ("pass" if passed else "hold")
            or row.get("comparison") != comparison
            or not math.isclose(
                _finite_number(row.get("observed"), name=f"check {name} observed"),
                observed,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _finite_fraction(row.get("threshold"), name=f"check {name} threshold"),
                threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"class-fidelity qualification check differs: {name}")
    computed_status = (
        "pass" if all(values[3] for values in expected_checks.values()) else "hold"
    )
    if status != computed_status:
        raise ValueError("class-fidelity qualification status differs from its checks")
    if require_pass and computed_status != "pass":
        raise ValueError("class-fidelity qualification did not pass its stage")

    sources = report.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"cofitok", "dense_identity"}:
        raise ValueError("class-fidelity qualification sources are incomplete")
    for name, identity in sources.items():
        if (
            not isinstance(identity, dict)
            or not str(identity.get("path", ""))
            or int(identity.get("bytes", 0)) < 1
            or len(str(identity.get("sha256", ""))) != 64
        ):
            raise ValueError(f"class-fidelity qualification source is invalid: {name}")
    boundary = report.get("claim_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("class_conditional_quality_evaluated") is not True
        or boundary.get("unconditional_distribution_quality_evaluated") is not False
        or boundary.get("standalone_generation_quality_claim_allowed") is not False
        or boundary.get("full_training_launch_allowed") is not False
        or boundary.get("release_authorization_allowed") is not False
    ):
        raise ValueError("class-fidelity qualification claim boundary differs")
    return {
        "valid": computed_status == "pass",
        "stage": expected_stage,
        "status": computed_status,
        "git": git,
        "classifier": classifier,
        "sampling_contract": sampling_contract,
        "thresholds": threshold_values,
        "metrics": {"cofitok": cofitok, "dense_identity": dense},
        "sources": sources,
        "claim_boundary": boundary,
    }
