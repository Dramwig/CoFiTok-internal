"""Physical validation for one frozen arm of capacity confirmation.

Each report replays the passing screen arm, reuses its exact step-10K
checkpoint without training, and physically validates a new 10K EMA DDIM-100
sample tree plus distribution and class-fidelity reports.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.environment import runtime_environment_sha256
from cofitok.generation.capacity_confirmation import (
    GUIDANCE_RESCALE,
    GUIDANCE_SCALE,
    SAMPLE_BATCH_SIZE,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    SAMPLE_STEPS,
)
from cofitok.generation.capacity_confirmation_execution import (
    LAUNCH_RECEIPT_ROLE,
    LAUNCH_RECEIPT_SCHEMA,
    validate_capacity_confirmation_launch_receipt_contract,
)
from cofitok.generation.capacity_screen import ARM_NAMES, ARM_SPECS, STOP_STEP
from cofitok.generation.capacity_screen_arm import (
    validate_capacity_screen_arm_validation,
)
from cofitok.generation_class_fidelity import validate_class_fidelity_report
from cofitok.image_integrity import sample_set_sha256
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import file_sha256


ARM_VALIDATION_SCHEMA = (
    "cofitok_generation_capacity_confirmation_arm_validation_v1"
)
ARM_VALIDATION_ROLE = "physical_capacity_confirmation_arm_validation"
ARM_VALIDATION_BOUNDARY = {
    "arm_evidence_complete": True,
    "frozen_checkpoint_training_performed": False,
    "confirmation_decision_allowed": False,
    "large_capacity_readiness_preparation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{name} is outside its finite domain")
    return result


def _identity(path: str | Path, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    return file_identity(source)


def _expected_report_git(execution: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(execution, "capacity confirmation execution checkout")
    result = {
        "revision": row.get("revision"),
        "branch": row.get("branch"),
        "tracked_dirty": False,
    }
    if (
        not isinstance(result["revision"], str)
        or len(result["revision"]) != 40
        or not isinstance(result["branch"], str)
        or not result["branch"]
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError("capacity confirmation execution checkout is malformed")
    return result


def _validate_report_git(
    value: Any,
    expected: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    row = _object(value, name)
    if row != dict(expected):
        raise ValueError(f"{name} differs from confirmation checkout")
    return copy.deepcopy(row)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sampling_evidence(
    report_path: Path,
    *,
    arm: str,
    expected_checkpoint: Mapping[str, Any],
    expected_git: Mapping[str, Any],
    expected_runtime_environment_sha256: str,
    expected_output_root: Path,
) -> dict[str, Any]:
    try:
        from scripts.evaluate_generation_metrics import (
            find_images,
            validate_sampling_provenance,
        )
    except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
        from evaluate_generation_metrics import (  # type: ignore[no-redef]
            find_images,
            validate_sampling_provenance,
        )

    spec = ARM_SPECS[arm]
    report = read_json_object(report_path, name=f"{arm} confirmation sampling")
    sampling = _object(report.get("sampling"), f"{arm} confirmation protocol")
    expected_sampling = {
        "sampler": "ddim",
        "num_samples": SAMPLE_COUNT,
        "start_index": 0,
        "batch_size": SAMPLE_BATCH_SIZE,
        "sample_steps": SAMPLE_STEPS,
        "prefix_budgets": [spec["prefix_budget"]],
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": SAMPLE_SEED,
        "precision": "bf16",
        "class_schedule": "balanced_modulo",
        "image_shape": [3, 256, 256],
    }
    for key, expected in expected_sampling.items():
        if sampling.get(key) != expected:
            raise ValueError(f"{arm} confirmation sampling differs at {key}")
    random_stream = _object(
        sampling.get("random_stream"), f"{arm} confirmation random stream"
    )
    if (
        random_stream.get("scope") != "per_global_sample_index"
        or random_stream.get("prefix_budgets_share_stream") is not True
        or random_stream.get("batch_size_invariant") is not True
        or random_stream.get("resume_index_invariant") is not True
    ):
        raise ValueError(f"{arm} confirmation random-stream contract differs")
    if (
        report.get("status") != "completed"
        or report.get("weights") != "ema"
        or int(report.get("checkpoint_step", -1)) != STOP_STEP
        or report.get("checkpoint_sha256") != expected_checkpoint["sha256"]
        or Path(str(report.get("checkpoint", ""))).resolve()
        != Path(str(expected_checkpoint["path"])).resolve()
        or Path(str(report.get("checkpoint_integrity_manifest", ""))).resolve()
        != Path(str(expected_checkpoint["integrity_manifest"]["path"])).resolve()
    ):
        raise ValueError(f"{arm} confirmation sampling checkpoint differs")
    git = _validate_report_git(
        report.get("git"), expected_git, f"{arm} confirmation sampling Git"
    )
    environment = _object(
        report.get("runtime_environment"), f"{arm} confirmation environment"
    )
    environment_sha = runtime_environment_sha256(environment)
    if (
        report.get("runtime_environment_sha256") != environment_sha
        or environment_sha != expected_runtime_environment_sha256
    ):
        raise ValueError(f"{arm} confirmation runtime environment differs")
    output_dirs = _object(report.get("output_dirs"), f"{arm} confirmation outputs")
    generated_dir = reject_symlink_chain(
        output_dirs.get(str(spec["prefix_budget"]), ""),
        name=f"{arm} confirmation sample directory",
    )
    if not _within(generated_dir, expected_output_root):
        raise ValueError(f"{arm} confirmation samples are outside authorized output")
    images = find_images(generated_dir)
    provenance = validate_sampling_provenance(report_path, generated_dir, images)
    if (
        len(images) != SAMPLE_COUNT
        or provenance["selected_prefix_budget"] != spec["prefix_budget"]
        or provenance["checkpoint_step"] != STOP_STEP
        or provenance["checkpoint_sha256"] != expected_checkpoint["sha256"]
        or provenance["weights"] != "ema"
        or provenance["git"] != git
        or provenance["sample_set_sha256"] != sample_set_sha256(images)
    ):
        raise ValueError(f"{arm} confirmation physical sample provenance differs")
    return {
        "report": _identity(report_path, f"{arm} confirmation sampling report"),
        "manifest": provenance["manifest_identity"],
        "progress": provenance["sampling_progress"]["identity"],
        "generated_dir": generated_dir.resolve().as_posix(),
        "sample_count": len(images),
        "sample_set_sha256": provenance["sample_set_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "runtime_environment_sha256": environment_sha,
        "git": git,
        "sampling": copy.deepcopy(sampling),
        "provenance": provenance,
    }


def _distribution_evidence(
    report_path: Path,
    *,
    arm: str,
    sampling: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json_object(report_path, name=f"{arm} confirmation metrics")
    if (
        int(report.get("schema_version", -1)) != 3
        or report.get("role") != "generation_directory_metrics_report"
        or report.get("status") != "completed"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
        or report.get("sample_provenance") != sampling["provenance"]
    ):
        raise ValueError(f"{arm} confirmation distribution report differs")
    git = _validate_report_git(
        report.get("git"), expected_git, f"{arm} confirmation metrics Git"
    )
    environment = _object(
        report.get("runtime_environment"), f"{arm} metrics environment"
    )
    environment_sha = runtime_environment_sha256(environment)
    if report.get("runtime_environment_sha256") != environment_sha:
        raise ValueError(f"{arm} metrics runtime environment SHA256 differs")
    counts = _object(report.get("counts"), f"{arm} confirmation counts")
    parameters = _object(report.get("parameters"), f"{arm} metric parameters")
    real_set = _object(report.get("real_set"), f"{arm} confirmation real set")
    if (
        int(counts.get("generated_image_count", -1)) != SAMPLE_COUNT
        or int(counts.get("real_image_count", -1)) != 50_000
        or parameters.get("precision_recall_enabled") is not True
        or int(parameters.get("batch_size", -1)) != 64
        or int(parameters.get("prc_batch_size", -1)) != SAMPLE_COUNT
        or int(parameters.get("seed", -1)) != SAMPLE_SEED
        or int(real_set.get("image_count", -1)) != 50_000
        or real_set.get("digest_schema") != "cofitok_image_tree_sha256_v1"
    ):
        raise ValueError(f"{arm} confirmation metric selection differs")
    values = _object(report.get("metrics"), f"{arm} confirmation metric values")
    metrics = {
        "fid": _finite(
            values.get("frechet_inception_distance"), f"{arm} FID", minimum=0.0
        ),
        "inception_score_mean": _finite(
            values.get("inception_score_mean"),
            f"{arm} inception score",
            minimum=0.0,
        ),
        "inception_score_std": _finite(
            values.get("inception_score_std"),
            f"{arm} inception score std",
            minimum=0.0,
        ),
        "precision": _finite(
            values.get("precision"), f"{arm} precision", minimum=0.0, maximum=1.0
        ),
        "recall": _finite(
            values.get("recall"), f"{arm} recall", minimum=0.0, maximum=1.0
        ),
    }
    if metrics["inception_score_mean"] <= 0.0:
        raise ValueError(f"{arm} inception score must be positive")
    return {
        "report": _identity(report_path, f"{arm} confirmation metrics report"),
        "git": git,
        "metrics": metrics,
        "counts": copy.deepcopy(counts),
        "real_set": copy.deepcopy(real_set),
        "runtime_environment_sha256": environment_sha,
    }


def _class_fidelity_evidence(
    report_path: Path,
    *,
    arm: str,
    sampling: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json_object(report_path, name=f"{arm} confirmation class fidelity")
    validate_class_fidelity_report(report)
    if report.get("sample_provenance") != sampling["provenance"]:
        raise ValueError(f"{arm} confirmation class fidelity uses another sample set")
    git = _validate_report_git(
        report.get("git"), expected_git, f"{arm} confirmation class Git"
    )
    metrics = _object(report.get("metrics"), f"{arm} confirmation class metrics")
    if (
        int(metrics.get("sample_count", -1)) != SAMPLE_COUNT
        or int(metrics.get("num_classes", -1)) != 1_000
        or int(metrics.get("requested_class_count", -1)) != 1_000
        or int(metrics.get("requested_count_min", -1)) != 10
        or int(metrics.get("requested_count_max", -1)) != 10
    ):
        raise ValueError(f"{arm} confirmation class count contract differs")
    selected = {
        name: _finite(
            metrics.get(name), f"{arm} class {name}", minimum=0.0, maximum=1.0
        )
        for name in (
            "top1_accuracy",
            "top5_accuracy",
            "mean_target_probability",
            "predicted_class_fraction",
            "normalized_predicted_class_entropy",
        )
    }
    selected["target_negative_log_likelihood"] = _finite(
        metrics.get("target_negative_log_likelihood"),
        f"{arm} target negative log likelihood",
        minimum=0.0,
    )
    return {
        "report": _identity(report_path, f"{arm} confirmation class report"),
        "git": git,
        "metrics": selected,
        "classifier": copy.deepcopy(report["classifier"]),
        "runtime_environment_sha256": report["runtime_environment_sha256"],
    }


def build_capacity_confirmation_arm_validation(
    *,
    arm: str,
    launch_receipt_path: str | Path,
    expected_launch_receipt_sha256: str,
    screen_arm_validation_path: str | Path,
    expected_screen_arm_validation_sha256: str,
    sampling_report_path: str | Path,
    metrics_report_path: str | Path,
    class_fidelity_report_path: str | Path,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError("capacity confirmation arm name differs")
    launch_path = reject_symlink_chain(
        launch_receipt_path, name="capacity confirmation launch receipt"
    ).resolve()
    if file_sha256(launch_path) != expected_launch_receipt_sha256:
        raise ValueError("capacity confirmation launch receipt SHA256 differs")
    launch = read_json_object(launch_path, name="capacity confirmation launch receipt")
    if (
        launch.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or launch.get("role") != LAUNCH_RECEIPT_ROLE
    ):
        raise ValueError("capacity confirmation launch receipt schema differs")
    execution = _object(launch.get("execution_checkout"), "confirmation execution")
    validated_launch = validate_capacity_confirmation_launch_receipt_contract(
        launch, expected_execution_checkout=execution
    )
    expected_git = _expected_report_git(execution)

    screen_path = reject_symlink_chain(
        screen_arm_validation_path, name=f"{arm} screen arm validation"
    ).resolve()
    if file_sha256(screen_path) != expected_screen_arm_validation_sha256:
        raise ValueError(f"{arm} screen arm validation SHA256 differs")
    screen_validation = validate_capacity_screen_arm_validation(
        read_json_object(screen_path, name=f"{arm} screen arm validation")
    )
    if screen_validation.get("arm") != arm:
        raise ValueError("capacity confirmation screen arm name differs")
    screen_identity = _identity(screen_path, f"{arm} screen arm validation")
    launch_sources = _object(
        validated_launch.get("source_evidence"), "confirmation launch sources"
    )
    prepared_arms = _object(
        validated_launch.get("frozen_arms"), "confirmation frozen arms"
    )
    if (
        _object(
            launch_sources.get("capacity_screen_arm_validations"),
            "confirmation screen arm identities",
        ).get(arm)
        != screen_identity
        or _object(prepared_arms.get(arm), f"{arm} frozen arm").get(
            "screen_arm_validation"
        )
        != screen_identity
    ):
        raise ValueError(f"{arm} confirmation binds another screen validation")
    checkpoint = _object(
        _object(screen_validation.get("training"), f"{arm} screen training").get(
            "checkpoint"
        ),
        f"{arm} frozen checkpoint",
    )
    if checkpoint != _object(prepared_arms[arm], f"{arm} frozen arm").get(
        "checkpoint"
    ):
        raise ValueError(f"{arm} frozen checkpoint differs from confirmation launch")
    output_root = Path(
        str(
            _object(
                validated_launch.get("output_dirs"), "confirmation output roots"
            )[arm]
        )
    )
    sampling = _sampling_evidence(
        Path(sampling_report_path),
        arm=arm,
        expected_checkpoint=checkpoint,
        expected_git=expected_git,
        expected_runtime_environment_sha256=str(
            validated_launch["runtime_environment_sha256"]
        ),
        expected_output_root=output_root,
    )
    distribution = _distribution_evidence(
        Path(metrics_report_path),
        arm=arm,
        sampling=sampling,
        expected_git=expected_git,
    )
    class_fidelity = _class_fidelity_evidence(
        Path(class_fidelity_report_path),
        arm=arm,
        sampling=sampling,
        expected_git=expected_git,
    )
    sources = {
        "launch_receipt": _identity(launch_path, "confirmation launch receipt"),
        "screen_arm_validation": screen_identity,
        "sampling_report": sampling["report"],
        "metrics_report": distribution["report"],
        "class_fidelity_report": class_fidelity["report"],
    }
    return {
        "schema_version": ARM_VALIDATION_SCHEMA,
        "role": ARM_VALIDATION_ROLE,
        "status": "pass",
        "arm": arm,
        "capacity": ARM_SPECS[arm]["capacity"],
        "method": ARM_SPECS[arm]["method"],
        "execution_git": copy.deepcopy(execution),
        "sources": sources,
        "frozen_training": {
            "training_performed": False,
            "checkpoint": copy.deepcopy(checkpoint),
            "screen_training": copy.deepcopy(screen_validation["training"]),
        },
        "sampling": sampling,
        "distribution": distribution,
        "class_fidelity": class_fidelity,
        "screen_checkpoint_evaluation": copy.deepcopy(
            screen_validation["checkpoint_evaluation"]
        ),
        "screen_rollout": copy.deepcopy(screen_validation["rollout"]),
        "authorization_boundary": copy.deepcopy(ARM_VALIDATION_BOUNDARY),
    }


def replay_capacity_confirmation_arm_validation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "capacity confirmation arm validation")
    sources = _object(row.get("sources"), "capacity confirmation arm sources")
    required = {
        "launch_receipt",
        "screen_arm_validation",
        "sampling_report",
        "metrics_report",
        "class_fidelity_report",
    }
    if set(sources) != required:
        raise ValueError("capacity confirmation arm source set differs")
    for name in required:
        identity = _object(sources[name], f"capacity confirmation source {name}")
        if set(identity) != {"path", "bytes", "sha256"}:
            raise ValueError(f"capacity confirmation source identity differs: {name}")
    return build_capacity_confirmation_arm_validation(
        arm=str(row.get("arm", "")),
        launch_receipt_path=sources["launch_receipt"]["path"],
        expected_launch_receipt_sha256=sources["launch_receipt"]["sha256"],
        screen_arm_validation_path=sources["screen_arm_validation"]["path"],
        expected_screen_arm_validation_sha256=sources["screen_arm_validation"][
            "sha256"
        ],
        sampling_report_path=sources["sampling_report"]["path"],
        metrics_report_path=sources["metrics_report"]["path"],
        class_fidelity_report_path=sources["class_fidelity_report"]["path"],
    )


def validate_capacity_confirmation_arm_validation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "capacity confirmation arm validation")
    if (
        row.get("schema_version") != ARM_VALIDATION_SCHEMA
        or row.get("role") != ARM_VALIDATION_ROLE
        or row.get("status") != "pass"
        or row.get("arm") not in ARM_NAMES
        or row.get("authorization_boundary") != ARM_VALIDATION_BOUNDARY
        or _object(
            row.get("frozen_training"), "capacity confirmation frozen training"
        ).get("training_performed")
        is not False
    ):
        raise ValueError("capacity confirmation arm validation contract differs")
    expected = replay_capacity_confirmation_arm_validation(row)
    if row != expected:
        raise ValueError("capacity confirmation arm validation is not exact replay")
    return copy.deepcopy(row)


__all__ = [
    "ARM_VALIDATION_BOUNDARY",
    "ARM_VALIDATION_ROLE",
    "ARM_VALIDATION_SCHEMA",
    "build_capacity_confirmation_arm_validation",
    "replay_capacity_confirmation_arm_validation",
    "validate_capacity_confirmation_arm_validation",
]
