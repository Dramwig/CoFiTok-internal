"""Physical validation for one arm of the four-arm capacity screen.

The validator deliberately re-reads every physical source: the intentional
step-10K training stop and checkpoint sidecar, the immutable 1K sampling
manifest/progress/image tree, distribution metrics, class fidelity, checkpoint
mechanism evaluation, and matched rollout diagnostic.  Its report is therefore
replayable from content-addressed source paths rather than a summary assembled
by the controller.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.environment import runtime_environment_sha256
from cofitok.generation.capacity_qualification_training import (
    validate_capacity_qualification_partial_training,
)
from cofitok.generation.capacity_screen import (
    ARM_NAMES,
    ARM_SPECS,
    GUIDANCE_RESCALE,
    GUIDANCE_SCALE,
    MECHANISM_IMAGES,
    MECHANISM_RANDOM_ORDERS,
    MECHANISM_TIMESTEP,
    ROLLOUT_BATCH_SIZE,
    ROLLOUT_IMAGES,
    ROLLOUT_SEED,
    SAMPLE_BATCH_SIZE,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    SAMPLE_STEPS,
    STOP_STEP,
)
from cofitok.generation.capacity_screen_execution import (
    LAUNCH_RECEIPT_ROLE,
    LAUNCH_RECEIPT_SCHEMA,
    validate_capacity_screen_launch_receipt_contract,
)
from cofitok.generation.stability_qualification import (
    DEFAULT_HIGH_FREQUENCY_TIMESTEPS,
)
from cofitok.generation_class_fidelity import validate_class_fidelity_report
from cofitok.image_integrity import sample_set_sha256
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import file_sha256


ARM_VALIDATION_SCHEMA = "cofitok_generation_capacity_screen_arm_validation_v1"
ARM_VALIDATION_ROLE = "physical_capacity_screen_arm_validation"
ARM_VALIDATION_BOUNDARY = {
    "arm_evidence_complete": True,
    "screen_decision_allowed": False,
    "capacity_confirmation_allowed": False,
    "additional_training_allowed": False,
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
    row = _object(execution, "capacity screen execution checkout")
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
        raise ValueError("capacity screen execution checkout is malformed")
    return result


def _validate_report_git(
    value: Any,
    expected: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    row = _object(value, name)
    if row != dict(expected):
        raise ValueError(f"{name} differs from the launch checkout")
    return copy.deepcopy(row)


def _sampling_evidence(
    report_path: Path,
    *,
    arm: str,
    expected_checkpoint: Mapping[str, Any],
    expected_git: Mapping[str, Any],
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
    report = read_json_object(report_path, name=f"{arm} sampling report")
    sampling = _object(report.get("sampling"), f"{arm} sampling protocol")
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
            raise ValueError(f"{arm} sampling protocol differs at {key}")
    random_stream = _object(
        sampling.get("random_stream"), f"{arm} sampling random stream"
    )
    if (
        random_stream.get("scope") != "per_global_sample_index"
        or random_stream.get("prefix_budgets_share_stream") is not True
        or random_stream.get("batch_size_invariant") is not True
        or random_stream.get("resume_index_invariant") is not True
    ):
        raise ValueError(f"{arm} sampling random-stream contract differs")
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
        raise ValueError(f"{arm} sampling checkpoint contract differs")
    git = _validate_report_git(report.get("git"), expected_git, f"{arm} sampling Git")
    environment = _object(
        report.get("runtime_environment"), f"{arm} sampling runtime environment"
    )
    environment_sha = runtime_environment_sha256(environment)
    if report.get("runtime_environment_sha256") != environment_sha:
        raise ValueError(f"{arm} sampling runtime environment SHA256 differs")
    output_dirs = _object(report.get("output_dirs"), f"{arm} sampling outputs")
    generated_dir = reject_symlink_chain(
        output_dirs.get(str(spec["prefix_budget"]), ""),
        name=f"{arm} generated sample directory",
    )
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
        raise ValueError(f"{arm} physical sample provenance differs")
    return {
        "report": _identity(report_path, f"{arm} sampling report"),
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
    report = read_json_object(report_path, name=f"{arm} distribution metrics")
    if (
        int(report.get("schema_version", -1)) != 3
        or report.get("role") != "generation_directory_metrics_report"
        or report.get("status") != "completed"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
        or report.get("sample_provenance") != sampling["provenance"]
    ):
        raise ValueError(f"{arm} distribution report contract differs")
    git = _validate_report_git(
        report.get("git"), expected_git, f"{arm} distribution evaluator Git"
    )
    environment = _object(
        report.get("runtime_environment"), f"{arm} metrics runtime environment"
    )
    if report.get("runtime_environment_sha256") != runtime_environment_sha256(
        environment
    ):
        raise ValueError(f"{arm} metrics runtime environment SHA256 differs")
    counts = _object(report.get("counts"), f"{arm} distribution counts")
    parameters = _object(report.get("parameters"), f"{arm} distribution parameters")
    real_set = _object(report.get("real_set"), f"{arm} real-set provenance")
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
        raise ValueError(f"{arm} distribution evaluation selection differs")
    values = _object(report.get("metrics"), f"{arm} distribution metrics")
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
        "report": _identity(report_path, f"{arm} distribution metrics"),
        "git": git,
        "metrics": metrics,
        "counts": copy.deepcopy(counts),
        "real_set": copy.deepcopy(real_set),
        "runtime_environment_sha256": report["runtime_environment_sha256"],
    }


def _class_fidelity_evidence(
    report_path: Path,
    *,
    arm: str,
    sampling: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json_object(report_path, name=f"{arm} class-fidelity report")
    validate_class_fidelity_report(report)
    if report.get("sample_provenance") != sampling["provenance"]:
        raise ValueError(f"{arm} class fidelity uses another sample set")
    git = _validate_report_git(
        report.get("git"), expected_git, f"{arm} class-fidelity evaluator Git"
    )
    metrics = _object(report.get("metrics"), f"{arm} class-fidelity metrics")
    if (
        int(metrics.get("sample_count", -1)) != SAMPLE_COUNT
        or int(metrics.get("num_classes", -1)) != 1_000
        or int(metrics.get("requested_class_count", -1)) != 1_000
        or int(metrics.get("requested_count_min", -1)) != 1
        or int(metrics.get("requested_count_max", -1)) != 1
    ):
        raise ValueError(f"{arm} class-fidelity count contract differs")
    selected = {
        name: _finite(metrics.get(name), f"{arm} class {name}", minimum=0.0, maximum=1.0)
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
        "report": _identity(report_path, f"{arm} class-fidelity report"),
        "git": git,
        "metrics": selected,
        "classifier": copy.deepcopy(report["classifier"]),
        "runtime_environment_sha256": report["runtime_environment_sha256"],
    }


def _coarse_utilization(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model = _object(config.get("model"), "CoFiTok capacity model")
    ratios = [
        _finite(value, "CoFiTok component-energy ratio", minimum=0.0)
        for value in metrics.get("component_energy_ratio_per_sample_mean", [])
    ]
    strides = [int(value) for value in model.get("token_spatial_strides", [])]
    token_count = int(model.get("token_count", -1))
    if (
        token_count != 8
        or len(ratios) != token_count
        or len(strides) != token_count
        or 1 not in strides
        or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("CoFiTok capacity utilization evidence is malformed")
    coarse_count = strides.index(1)
    if (
        coarse_count < 1
        or any(value <= 1 for value in strides[:coarse_count])
        or any(value != 1 for value in strides[coarse_count:])
    ):
        raise ValueError("CoFiTok capacity stride partition differs")
    return {
        "component_energy_ratios": ratios,
        "coarse_token_count": coarse_count,
        "coarse_token_energy_ratio": sum(ratios[:coarse_count]),
        "tail_two_energy_ratio": sum(ratios[-2:]),
        "max_single_token_energy_ratio": max(ratios),
        "token_spatial_strides": strides,
    }


def _checkpoint_evidence(
    report_path: Path,
    *,
    arm: str,
    expected_checkpoint: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json_object(report_path, name=f"{arm} checkpoint evaluation")
    spec = ARM_SPECS[arm]
    expected_request = {
        "num_images": MECHANISM_IMAGES,
        "timestep": MECHANISM_TIMESTEP,
        "random_orders": MECHANISM_RANDOM_ORDERS if spec["method"] == "cofitok" else 0,
        "seed": SAMPLE_SEED,
        "weights": "ema",
        "precision": "bf16",
    }
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("role") != "generation_checkpoint_evaluation_report"
        or report.get("status") != "completed"
        or report.get("request") != expected_request
        or report.get("weights") != "ema"
        or report.get("precision") != "bf16"
        or int(report.get("checkpoint_step", -1)) != STOP_STEP
        or report.get("checkpoint_sha256") != expected_checkpoint["sha256"]
        or Path(str(report.get("checkpoint", ""))).resolve()
        != Path(str(expected_checkpoint["path"])).resolve()
        or Path(str(report.get("checkpoint_integrity_manifest", ""))).resolve()
        != Path(str(expected_checkpoint["integrity_manifest"]["path"])).resolve()
        or report.get("config") != dict(expected_config)
    ):
        raise ValueError(f"{arm} checkpoint-evaluation contract differs")
    git = _validate_report_git(
        report.get("git"), expected_git, f"{arm} checkpoint evaluator Git"
    )
    manifest_path = report_path.parent / "checkpoint_evaluation_manifest.json"
    manifest = read_json_object(manifest_path, name=f"{arm} checkpoint manifest")
    checkpoint_manifest = _object(
        manifest.get("checkpoint"), f"{arm} checkpoint manifest identity"
    )
    if (
        int(manifest.get("schema_version", -1)) != 1
        or manifest.get("role") != "generation_checkpoint_evaluation_manifest"
        or manifest.get("git") != git
        or manifest.get("request") != expected_request
        or Path(str(manifest.get("output_dir", ""))).resolve()
        != report_path.parent.resolve()
        or Path(str(manifest.get("report", ""))).resolve() != report_path.resolve()
        or Path(str(checkpoint_manifest.get("path", ""))).resolve()
        != Path(str(expected_checkpoint["path"])).resolve()
        or checkpoint_manifest.get("bytes") != expected_checkpoint["bytes"]
        or checkpoint_manifest.get("sha256") != expected_checkpoint["sha256"]
        or int(checkpoint_manifest.get("step", -1)) != STOP_STEP
        or checkpoint_manifest.get("integrity_manifest")
        != expected_checkpoint["integrity_manifest"]
        or report.get("manifest") != file_identity(manifest_path)
    ):
        raise ValueError(f"{arm} checkpoint-evaluation manifest differs")
    metrics = _object(report.get("metrics"), f"{arm} checkpoint metrics")
    order_count = int(metrics.get("order_count", -1))
    expected_order_count = 2 + MECHANISM_RANDOM_ORDERS if spec["method"] == "cofitok" else 1
    ordered = _object(
        _object(metrics.get("orders"), f"{arm} order metrics").get("ordered"),
        f"{arm} ordered metrics",
    )
    endpoint = _finite(
        ordered.get("endpoint_clean_mse"), f"{arm} endpoint clean MSE", minimum=0.0
    )
    if (
        int(metrics.get("evaluated_images", -1)) != MECHANISM_IMAGES
        or int(metrics.get("timestep", -1)) != MECHANISM_TIMESTEP
        or order_count != expected_order_count
        or endpoint <= 0.0
    ):
        raise ValueError(f"{arm} checkpoint metrics selection differs")
    summary: dict[str, Any] = {
        "evaluated_images": MECHANISM_IMAGES,
        "timestep": MECHANISM_TIMESTEP,
        "order_count": order_count,
        "ordered_endpoint_clean_mse": endpoint,
    }
    if spec["method"] == "cofitok":
        utilization = _coarse_utilization(metrics, expected_config)
        summary.update(
            {
                "ordered_rank_by_path_auc": int(
                    metrics.get("ordered_rank_by_path_auc", -1)
                ),
                "zero_token_max_abs": _finite(
                    metrics.get("zero_token_max_abs"),
                    f"{arm} zero-token maximum",
                    minimum=0.0,
                ),
                "shuffled_to_ordered_endpoint_ratio": _finite(
                    metrics.get("shuffled_to_ordered_endpoint_ratio"),
                    f"{arm} shuffle mismatch",
                    minimum=0.0,
                ),
                "utilization": utilization,
            }
        )
    return {
        "report": _identity(report_path, f"{arm} checkpoint evaluation"),
        "manifest": _identity(manifest_path, f"{arm} checkpoint manifest"),
        "git": git,
        "request": expected_request,
        "summary": summary,
    }


def _rollout_evidence(
    report_path: Path,
    *,
    arm: str,
    expected_checkpoint: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json_object(report_path, name=f"{arm} rollout report")
    expected_protocol = {
        "num_images": ROLLOUT_IMAGES,
        "batch_size": ROLLOUT_BATCH_SIZE,
        "teacher_timesteps": [999, 900, 750, 500, 250, 100, 10],
        "sample_steps": SAMPLE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "teacher_guidance_scale": 1.0,
        "guidance_rescale": GUIDANCE_RESCALE,
        "cfg_batch_mode": "batched",
        "clip_x0": True,
        "precision": "bf16",
        "seed": ROLLOUT_SEED,
    }
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "completed"
        or report.get("weights") != "ema"
        or int(report.get("checkpoint_step", -1)) != STOP_STEP
        or report.get("checkpoint_sha256") != expected_checkpoint["sha256"]
        or Path(str(report.get("checkpoint", ""))).resolve()
        != Path(str(expected_checkpoint["path"])).resolve()
        or Path(str(report.get("checkpoint_integrity_manifest", ""))).resolve()
        != Path(str(expected_checkpoint["integrity_manifest"]["path"])).resolve()
        or report.get("config") != dict(expected_config)
        or report.get("protocol") != expected_protocol
    ):
        raise ValueError(f"{arm} rollout contract differs")
    git = _validate_report_git(report.get("git"), expected_git, f"{arm} rollout Git")
    reconstruction = _object(
        report.get("reconstruction_rollout"), f"{arm} reconstruction rollout"
    )
    free = _object(report.get("free_sampling_rollout"), f"{arm} free rollout")
    reconstruction_summary = _object(
        reconstruction.get("summary"), f"{arm} reconstruction summary"
    )
    final_mse = _finite(
        reconstruction_summary.get("final_clipped_x0_mse"),
        f"{arm} final reconstruction MSE",
        minimum=0.0,
    )
    amplification = _finite(
        reconstruction_summary.get("final_to_best_x0_mse_amplification"),
        f"{arm} reconstruction amplification",
        minimum=0.0,
    )
    steps = free.get("steps")
    if not isinstance(steps, list) or len(steps) != SAMPLE_STEPS:
        raise ValueError(f"{arm} free rollout step count differs")
    by_timestep = {
        int(_object(step, f"{arm} free rollout step").get("timestep", -1)): _object(
            step, f"{arm} free rollout step"
        )
        for step in steps
    }
    high_frequency: dict[str, float] = {}
    for timestep in DEFAULT_HIGH_FREQUENCY_TIMESTEPS:
        if timestep not in by_timestep:
            raise ValueError(f"{arm} free rollout lacks timestep {timestep}")
        high_frequency[str(timestep)] = _finite(
            by_timestep[timestep].get("predicted_x0_high_frequency_ratio"),
            f"{arm} high-frequency ratio at {timestep}",
            minimum=0.0,
        )
    return {
        "report": _identity(report_path, f"{arm} rollout report"),
        "git": git,
        "protocol": expected_protocol,
        "summary": {
            "final_reconstruction_x0_mse": final_mse,
            "final_to_best_x0_mse_amplification": amplification,
            "predicted_x0_high_frequency_ratio": high_frequency,
        },
    }


def build_capacity_screen_arm_validation(
    *,
    arm: str,
    launch_receipt_path: str | Path,
    expected_launch_receipt_sha256: str,
    config_path: str | Path,
    training_report_path: str | Path,
    sampling_report_path: str | Path,
    metrics_report_path: str | Path,
    class_fidelity_report_path: str | Path,
    checkpoint_evaluation_report_path: str | Path,
    rollout_report_path: str | Path,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError("capacity screen arm name differs")
    spec = ARM_SPECS[arm]
    launch_path = reject_symlink_chain(
        launch_receipt_path, name="capacity screen launch receipt"
    ).resolve()
    if file_sha256(launch_path) != expected_launch_receipt_sha256:
        raise ValueError("capacity screen launch receipt SHA256 differs")
    launch = read_json_object(launch_path, name="capacity screen launch receipt")
    if (
        launch.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or launch.get("role") != LAUNCH_RECEIPT_ROLE
    ):
        raise ValueError("capacity screen launch receipt schema differs")
    execution = _object(launch.get("execution_checkout"), "capacity execution Git")
    validated_launch = validate_capacity_screen_launch_receipt_contract(
        launch,
        expected_execution_checkout=execution,
    )
    expected_git = _expected_report_git(execution)
    config_file = reject_symlink_chain(config_path, name=f"{arm} config").resolve()
    config_identity = _identity(config_file, f"{arm} config")
    launch_configs = _object(
        _object(validated_launch.get("source_evidence"), "capacity launch sources").get(
            "configs"
        ),
        "capacity launch configs",
    )
    if config_identity != launch_configs.get(arm):
        raise ValueError(f"{arm} config differs from the launch receipt")
    run_dir = Path(str(_object(validated_launch.get("run_dirs"), "capacity run dirs")[arm]))
    training_report = reject_symlink_chain(
        training_report_path, name=f"{arm} training report"
    ).resolve()
    if training_report.parent.resolve() != run_dir.resolve():
        raise ValueError(f"{arm} training report is outside its authorized run")
    runtime = _object(
        validated_launch.get("runtime_selection"), "capacity runtime selection"
    )
    training = validate_capacity_qualification_partial_training(
        report_path=training_report,
        config_path=config_file,
        expected_revision=str(execution["revision"]),
        expected_branch=str(execution["branch"]),
        expected_parameter_count=int(spec["parameter_count"]),
        expected_micro_batch_size=int(runtime["micro_batch_size"]),
        expected_gradient_accumulation_steps=int(
            runtime["gradient_accumulation_steps"]
        ),
        expected_stage=str(spec["recipe_stage"]),
        expected_base_channels=int(spec["base_channels"]),
        allow_exact_resume=True,
    )
    if (
        training["runtime_environment_sha256"]
        != runtime["runtime_environment_sha256"]
    ):
        raise ValueError(f"{arm} training runtime differs from launch selection")
    raw_training = read_json_object(training_report, name=f"{arm} training report")
    config = _object(raw_training.get("config"), f"{arm} resolved training config")
    final_metrics = _object(
        raw_training.get("final_metrics"), f"{arm} final training metrics"
    )
    validation_epsilon = _finite(
        final_metrics.get("validation_epsilon_mse"),
        f"{arm} final validation epsilon MSE",
        minimum=0.0,
    )
    checkpoint = _object(training.get("checkpoint"), f"{arm} checkpoint")
    sampling = _sampling_evidence(
        Path(sampling_report_path),
        arm=arm,
        expected_checkpoint=checkpoint,
        expected_git=expected_git,
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
    checkpoint_evaluation = _checkpoint_evidence(
        Path(checkpoint_evaluation_report_path),
        arm=arm,
        expected_checkpoint=checkpoint,
        expected_config=config,
        expected_git=expected_git,
    )
    rollout = _rollout_evidence(
        Path(rollout_report_path),
        arm=arm,
        expected_checkpoint=checkpoint,
        expected_config=config,
        expected_git=expected_git,
    )
    sources = {
        "launch_receipt": _identity(launch_path, "capacity launch receipt"),
        "config": config_identity,
        "training_report": _identity(training_report, f"{arm} training report"),
        "sampling_report": sampling["report"],
        "metrics_report": distribution["report"],
        "class_fidelity_report": class_fidelity["report"],
        "checkpoint_evaluation_report": checkpoint_evaluation["report"],
        "rollout_report": rollout["report"],
    }
    return {
        "schema_version": ARM_VALIDATION_SCHEMA,
        "role": ARM_VALIDATION_ROLE,
        "status": "pass",
        "arm": arm,
        "capacity": spec["capacity"],
        "method": spec["method"],
        "execution_git": copy.deepcopy(execution),
        "sources": sources,
        "training": {
            "validation": training,
            "validation_epsilon_mse": validation_epsilon,
            "checkpoint": checkpoint,
        },
        "sampling": sampling,
        "distribution": distribution,
        "class_fidelity": class_fidelity,
        "checkpoint_evaluation": checkpoint_evaluation,
        "rollout": rollout,
        "authorization_boundary": copy.deepcopy(ARM_VALIDATION_BOUNDARY),
    }


def replay_capacity_screen_arm_validation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "capacity screen arm validation")
    sources = _object(row.get("sources"), "capacity screen arm sources")
    required = {
        "launch_receipt",
        "config",
        "training_report",
        "sampling_report",
        "metrics_report",
        "class_fidelity_report",
        "checkpoint_evaluation_report",
        "rollout_report",
    }
    if set(sources) != required:
        raise ValueError("capacity screen arm source set differs")
    for name in required:
        source = _object(sources[name], f"capacity arm source {name}")
        if set(source) != {"path", "bytes", "sha256"}:
            raise ValueError(f"capacity arm source identity differs: {name}")
    return build_capacity_screen_arm_validation(
        arm=str(row.get("arm", "")),
        launch_receipt_path=sources["launch_receipt"]["path"],
        expected_launch_receipt_sha256=sources["launch_receipt"]["sha256"],
        config_path=sources["config"]["path"],
        training_report_path=sources["training_report"]["path"],
        sampling_report_path=sources["sampling_report"]["path"],
        metrics_report_path=sources["metrics_report"]["path"],
        class_fidelity_report_path=sources["class_fidelity_report"]["path"],
        checkpoint_evaluation_report_path=sources[
            "checkpoint_evaluation_report"
        ]["path"],
        rollout_report_path=sources["rollout_report"]["path"],
    )


def validate_capacity_screen_arm_validation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "capacity screen arm validation")
    if (
        row.get("schema_version") != ARM_VALIDATION_SCHEMA
        or row.get("role") != ARM_VALIDATION_ROLE
        or row.get("status") != "pass"
        or row.get("arm") not in ARM_NAMES
        or row.get("authorization_boundary") != ARM_VALIDATION_BOUNDARY
    ):
        raise ValueError("capacity screen arm validation contract differs")
    expected = replay_capacity_screen_arm_validation(row)
    if row != expected:
        raise ValueError("capacity screen arm validation is not an exact replay")
    return copy.deepcopy(row)


__all__ = [
    "ARM_VALIDATION_BOUNDARY",
    "ARM_VALIDATION_ROLE",
    "ARM_VALIDATION_SCHEMA",
    "build_capacity_screen_arm_validation",
    "replay_capacity_screen_arm_validation",
    "validate_capacity_screen_arm_validation",
]
