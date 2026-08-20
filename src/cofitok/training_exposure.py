from __future__ import annotations

import math
from typing import Any, Mapping

from cofitok.data.provenance import validate_dataset_provenance


TRAINING_EXPOSURE_SCHEMA_VERSION = 1


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a positive integer")
    if value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def training_exposure_summary(
    report: Mapping[str, Any],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    config = report.get("config")
    final_metrics = report.get("final_metrics")
    provenance = report.get("dataset_provenance")
    if not isinstance(config, Mapping) or not isinstance(final_metrics, Mapping):
        raise ValueError("training report lacks config or final metrics")
    if not isinstance(provenance, Mapping):
        raise ValueError("training report lacks dataset provenance")

    data = config.get("data")
    optimization = config.get("optimization")
    if not isinstance(data, Mapping) or not isinstance(optimization, Mapping):
        raise ValueError("training report lacks data or optimization config")
    dataset = str(data.get("dataset", ""))
    if not dataset:
        raise ValueError("training report dataset is missing")
    validated_provenance = validate_dataset_provenance(
        provenance,
        expected_dataset=dataset,
    )

    target_steps = _positive_int(report.get("target_steps"), label="target steps")
    completed_steps = _nonnegative_int(
        report.get("completed_steps"), label="completed steps"
    )
    if completed_steps > target_steps:
        raise ValueError("completed steps are outside the training horizon")
    final_step = _nonnegative_int(final_metrics.get("step"), label="final metric step")
    if final_step != completed_steps:
        raise ValueError("final metric step differs from completed steps")

    micro_batch = _positive_int(data.get("batch_size"), label="micro batch size")
    accumulation = _positive_int(
        optimization.get("gradient_accumulation_steps"),
        label="gradient accumulation steps",
    )
    effective_batch = micro_batch * accumulation
    samples_seen = _nonnegative_int(
        final_metrics.get("samples_seen"), label="samples seen"
    )
    expected_samples_seen = completed_steps * effective_batch
    if samples_seen != expected_samples_seen:
        raise ValueError("samples_seen differs from completed_steps * effective_batch")

    training_complete = report.get("training_complete")
    if type(training_complete) is not bool:
        raise ValueError("training_complete must be boolean")
    expected_complete = completed_steps == target_steps
    if training_complete is not expected_complete:
        raise ValueError("training_complete differs from completed training horizon")
    if require_complete and not training_complete:
        raise ValueError("complete training exposure was required")

    splits = validated_provenance.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("validated dataset provenance lacks split counts")
    train_images = _positive_int(splits.get("train"), label="train image count")
    target_samples_seen = target_steps * effective_batch
    completed_equivalent_epochs = samples_seen / train_images
    target_equivalent_epochs = target_samples_seen / train_images
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (completed_equivalent_epochs, target_equivalent_epochs)
    ):
        raise ValueError("training exposure is not finite")

    git = report.get("git")
    git_identity = dict(git) if isinstance(git, Mapping) else None
    return {
        "schema_version": TRAINING_EXPOSURE_SCHEMA_VERSION,
        "status": "complete" if training_complete else "partial",
        "dataset": dataset,
        "dataset_identity_sha256": validated_provenance["identity_sha256"],
        "train_image_count": train_images,
        "target_steps": target_steps,
        "completed_steps": completed_steps,
        "training_complete": training_complete,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "samples_seen": samples_seen,
        "expected_samples_seen_at_completed_step": expected_samples_seen,
        "target_samples_seen": target_samples_seen,
        "completed_fraction": completed_steps / target_steps,
        "completed_equivalent_epochs": completed_equivalent_epochs,
        "target_equivalent_epochs": target_equivalent_epochs,
        "git": git_identity,
    }


def compare_training_exposures(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one training exposure is required")
    normalized = {str(label): dict(row) for label, row in rows.items()}
    if any(not label for label in normalized) or len(normalized) != len(rows):
        raise ValueError("training exposure labels must be unique and non-empty")

    datasets = {str(row["dataset"]) for row in normalized.values()}
    dataset_identities = {
        str(row["dataset_identity_sha256"]) for row in normalized.values()
    }
    effective_batches = {
        int(row["effective_batch_size"]) for row in normalized.values()
    }
    completed_steps = {int(row["completed_steps"]) for row in normalized.values()}
    samples_seen = {int(row["samples_seen"]) for row in normalized.values()}
    equivalent_epochs = [
        float(row["completed_equivalent_epochs"]) for row in normalized.values()
    ]
    same_equivalent_epochs = max(equivalent_epochs) - min(equivalent_epochs) <= 1e-12
    same_dataset = len(datasets) == 1
    same_dataset_identity = len(dataset_identities) == 1
    same_effective_batch = len(effective_batches) == 1
    same_completed_steps = len(completed_steps) == 1
    same_images_seen = len(samples_seen) == 1
    return {
        "same_dataset": same_dataset,
        "same_dataset_identity": same_dataset_identity,
        "same_effective_batch_size": same_effective_batch,
        "same_completed_steps": same_completed_steps,
        "same_images_seen": same_images_seen,
        "same_equivalent_epochs": same_equivalent_epochs,
        "same_dataset_normalized_exposure": (
            same_dataset_identity and same_equivalent_epochs
        ),
        "step_budget_directly_comparable": (
            same_dataset_identity and same_effective_batch and same_completed_steps
        ),
        "image_budget_directly_comparable": (
            same_dataset_identity and same_images_seen
        ),
        "dataset_normalized_budget_directly_comparable": (
            same_dataset_identity and same_equivalent_epochs
        ),
        "quality_metric_comparison_allowed": False,
        "quality_metric_comparison_reason": (
            "Training exposure alone never establishes matched sampling, evaluator, "
            "checkpoint, random-stream, or generated-sample provenance."
        ),
    }
