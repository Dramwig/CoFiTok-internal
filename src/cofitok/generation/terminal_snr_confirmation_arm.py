"""Physical replay for one terminal-SNR frozen-confirmation arm."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from cofitok.environment import runtime_environment_sha256
from cofitok.generation import capacity_confirmation_arm as shared_arm
from cofitok.generation.terminal_snr_confirmation import SAMPLE_SEED
from cofitok.generation.terminal_snr_confirmation_execution import (
    LAUNCH_RECEIPT_ROLE,
    LAUNCH_RECEIPT_SCHEMA,
    validate_terminal_snr_confirmation_launch_receipt_contract,
)
from cofitok.generation.terminal_snr_screen import ARM_NAMES, ARM_SPECS, STOP_STEP
from cofitok.generation.terminal_snr_screen_arm import (
    validate_terminal_snr_screen_arm_validation,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256


ARM_VALIDATION_SCHEMA = (
    "cofitok_generation_terminal_snr_confirmation_arm_validation_v1"
)
ARM_VALIDATION_ROLE = "physical_terminal_snr_confirmation_arm_validation"
ARM_VALIDATION_BOUNDARY = {
    "arm_evidence_complete": True,
    "frozen_checkpoint_training_performed": False,
    "confirmation_decision_allowed": False,
    "large_capacity_readiness_preparation_allowed": False,
    "large_capacity_readiness_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


@contextmanager
def _terminal_confirmation_helper_contract() -> Iterator[None]:
    """Bind shared physical helpers to terminal-SNR arms and the new seed."""

    saved_specs = shared_arm.ARM_SPECS
    saved_names = shared_arm.ARM_NAMES
    saved_seed = shared_arm.SAMPLE_SEED
    try:
        shared_arm.ARM_SPECS = ARM_SPECS
        shared_arm.ARM_NAMES = ARM_NAMES
        shared_arm.SAMPLE_SEED = SAMPLE_SEED
        yield
    finally:
        shared_arm.ARM_SPECS = saved_specs
        shared_arm.ARM_NAMES = saved_names
        shared_arm.SAMPLE_SEED = saved_seed


def _require_generation_runtime(
    evidence: Mapping[str, Any],
    *,
    arm: str,
    expected_runtime_environment_sha256: str,
) -> None:
    if evidence.get("runtime_environment_sha256") != expected_runtime_environment_sha256:
        raise ValueError(f"{arm} confirmation sampling runtime differs from launch")


def _require_matching_evaluator_runtime(
    distribution: Mapping[str, Any],
    class_fidelity: Mapping[str, Any],
    *,
    arm: str,
) -> None:
    distribution_runtime = distribution.get("runtime_environment_sha256")
    class_runtime = class_fidelity.get("runtime_environment_sha256")
    if (
        not isinstance(distribution_runtime, str)
        or len(distribution_runtime) != 64
        or distribution_runtime != distribution_runtime.lower()
        or any(
            character not in "0123456789abcdef"
            for character in distribution_runtime
        )
        or class_runtime != distribution_runtime
    ):
        raise ValueError(f"{arm} confirmation evaluator runtime identities differ")


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def validate_terminal_snr_confirmation_sampling_preflight(
    report_path: str | Path,
    *,
    arm: str,
    expected_checkpoint: Mapping[str, Any],
    expected_execution_git: Mapping[str, Any],
    expected_screen_git: Mapping[str, Any],
    expected_runtime_environment_sha256: str,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError("terminal-SNR confirmation preflight arm differs")
    spec = ARM_SPECS[arm]
    expected_output_dtype = (
        "torch.float32" if spec["method"] == "cofitok" else "torch.bfloat16"
    )
    path = reject_symlink_chain(
        report_path, name=f"{arm} confirmation sampling preflight"
    ).resolve()
    report = shared_arm.read_json_object(
        path, name=f"{arm} confirmation sampling preflight"
    )
    execution_git = shared_arm._expected_report_git(expected_execution_git)
    screen_git = shared_arm._expected_report_git(expected_screen_git)
    request = _object(report.get("request"), f"{arm} confirmation preflight request")
    result = _object(report.get("result"), f"{arm} confirmation preflight result")
    environment = _object(
        report.get("runtime_environment"), f"{arm} confirmation preflight runtime"
    )
    environment_sha = runtime_environment_sha256(environment)
    memory = _object(
        result.get("cuda_memory_after_forward"),
        f"{arm} confirmation preflight CUDA memory",
    )
    checkpoint_path = Path(str(expected_checkpoint.get("path", ""))).resolve()
    integrity_path = Path(
        str(_object(expected_checkpoint.get("integrity_manifest"), f"{arm} integrity")["path"])
    ).resolve()
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "passed"
        or report.get("git") != execution_git
        # Raw training checkpoints intentionally do not expose inference-artifact
        # source provenance. Their originating screen checkout is instead bound
        # by the physically replayed screen-arm validation and frozen checkpoint.
        or report.get("source_git") is not None
        or report.get("runtime_environment_sha256") != environment_sha
        or environment_sha != expected_runtime_environment_sha256
        or Path(str(report.get("checkpoint", ""))).resolve() != checkpoint_path
        or report.get("checkpoint_sha256") != expected_checkpoint.get("sha256")
        or Path(str(report.get("checkpoint_integrity_manifest", ""))).resolve()
        != integrity_path
        or int(report.get("checkpoint_step", -1)) != STOP_STEP
        or report.get("weights") != "ema"
        or report.get("requested_weights") != "ema"
        or report.get("artifact_type") != "training_checkpoint"
        or report.get("release_authorization_required") is not False
        or report.get("completion_authorization_required") is not False
        or report.get("release_authorization") is not None
        or report.get("completion_authorization") is not None
        or report.get("source_checkpoint_sha256") is not None
        or report.get("source_runtime_environment_sha256") is not None
        or not str(report.get("device", "")).startswith("cuda")
        or int(request.get("batch_size", -1)) != 4
        or int(request.get("effective_model_batch_size", -1)) != 8
        or int(request.get("forward_passes", -1)) != 1
        or request.get("image_shape") != [3, 256, 256]
        or int(request.get("prefix_budget", -1)) != int(spec["prefix_budget"])
        or int(request.get("token_count", -1)) != int(spec["prefix_budget"])
        or request.get("precision") != "bf16"
        or float(request.get("guidance_scale", -1.0)) != 1.5
        or float(request.get("guidance_rescale", -1.0)) != 0.0
        or request.get("cfg_batch_mode") != "batched"
        or request.get("class_conditional") is not True
        or int(request.get("timestep", -1)) != 999
        or int(request.get("warmup_forwards", -1)) != 0
        or int(request.get("measured_forwards", -1)) != 1
        or result.get("output_shape") != [4, 3, 256, 256]
        or result.get("output_dtype") != expected_output_dtype
        or result.get("output_finite") is not True
        or int(memory.get("peak_allocated_bytes", 0)) < 1
        or int(result.get("device_total_memory_bytes", 0)) < 1
    ):
        raise ValueError(f"{arm} confirmation sampling preflight differs")
    return {
        "report": shared_arm._identity(path, f"{arm} confirmation preflight"),
        "status": "passed",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "checkpoint_step": int(report["checkpoint_step"]),
        "runtime_environment_sha256": environment_sha,
        "execution_git": execution_git,
        "checkpoint_screen_git": screen_git,
        "source_git": None,
        "request": copy.deepcopy(request),
        "output_finite": True,
        "elapsed_seconds": _finite_positive(
            result.get("elapsed_seconds"), f"{arm} preflight elapsed seconds"
        ),
        "peak_allocated_bytes": int(memory["peak_allocated_bytes"]),
        "device_total_memory_bytes": int(result["device_total_memory_bytes"]),
    }


def build_terminal_snr_confirmation_arm_validation(
    *,
    arm: str,
    launch_receipt_path: str | Path,
    expected_launch_receipt_sha256: str,
    screen_arm_validation_path: str | Path,
    expected_screen_arm_validation_sha256: str,
    sampling_preflight_report_path: str | Path,
    sampling_report_path: str | Path,
    metrics_report_path: str | Path,
    class_fidelity_report_path: str | Path,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError("terminal-SNR confirmation arm name differs")
    spec = ARM_SPECS[arm]
    launch_path = reject_symlink_chain(
        launch_receipt_path, name="terminal-SNR confirmation launch receipt"
    ).resolve()
    if file_sha256(launch_path) != expected_launch_receipt_sha256:
        raise ValueError("terminal-SNR confirmation launch receipt SHA256 differs")
    launch = shared_arm.read_json_object(
        launch_path, name="terminal-SNR confirmation launch receipt"
    )
    if (
        launch.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or launch.get("role") != LAUNCH_RECEIPT_ROLE
    ):
        raise ValueError("terminal-SNR confirmation launch receipt schema differs")
    execution = _object(
        launch.get("execution_checkout"), "terminal-SNR confirmation execution"
    )
    validated_launch = validate_terminal_snr_confirmation_launch_receipt_contract(
        launch, expected_execution_checkout=execution
    )
    expected_git = shared_arm._expected_report_git(execution)

    screen_path = reject_symlink_chain(
        screen_arm_validation_path, name=f"{arm} terminal-SNR screen validation"
    ).resolve()
    if file_sha256(screen_path) != expected_screen_arm_validation_sha256:
        raise ValueError(f"{arm} terminal-SNR screen validation SHA256 differs")
    screen_validation = validate_terminal_snr_screen_arm_validation(
        shared_arm.read_json_object(
            screen_path, name=f"{arm} terminal-SNR screen validation"
        )
    )
    if screen_validation.get("arm") != arm:
        raise ValueError("terminal-SNR confirmation screen arm name differs")
    screen_identity = shared_arm._identity(
        screen_path, f"{arm} terminal-SNR screen validation"
    )
    launch_sources = _object(
        validated_launch.get("source_evidence"), "terminal-SNR confirmation sources"
    )
    prepared_arms = _object(
        validated_launch.get("frozen_arms"), "terminal-SNR confirmation frozen arms"
    )
    frozen = _object(prepared_arms.get(arm), f"{arm} frozen arm")
    screen_ids = _object(
        launch_sources.get("terminal_snr_screen_arm_validations"),
        "terminal-SNR confirmation screen identities",
    )
    if (
        screen_ids.get(arm) != screen_identity
        or frozen.get("screen_arm_validation") != screen_identity
        or screen_validation.get("execution_git")
        != validated_launch.get("screen_execution_checkout")
        or frozen.get("condition") != spec["condition"]
        or frozen.get("method") != spec["method"]
        or frozen.get("endpoint_fraction") != spec["endpoint_fraction"]
    ):
        raise ValueError(f"{arm} confirmation binds another screen validation")
    checkpoint = _object(
        _object(screen_validation.get("training"), f"{arm} screen training").get(
            "checkpoint"
        ),
        f"{arm} frozen checkpoint",
    )
    if checkpoint != frozen.get("checkpoint"):
        raise ValueError(f"{arm} frozen checkpoint differs from confirmation launch")
    preflight = validate_terminal_snr_confirmation_sampling_preflight(
        sampling_preflight_report_path,
        arm=arm,
        expected_checkpoint=checkpoint,
        expected_execution_git=execution,
        expected_screen_git=screen_validation["execution_git"],
        expected_runtime_environment_sha256=str(
            validated_launch["runtime_environment_sha256"]
        ),
    )
    output_root = Path(
        str(
            _object(
                validated_launch.get("output_dirs"), "terminal-SNR confirmation outputs"
            )[arm]
        )
    )
    with _terminal_confirmation_helper_contract():
        sampling = shared_arm._sampling_evidence(
            Path(sampling_report_path),
            arm=arm,
            expected_checkpoint=checkpoint,
            expected_git=expected_git,
            expected_runtime_environment_sha256=str(
                validated_launch["runtime_environment_sha256"]
            ),
            expected_output_root=output_root,
        )
        distribution = shared_arm._distribution_evidence(
            Path(metrics_report_path),
            arm=arm,
            sampling=sampling,
            expected_git=expected_git,
        )
        class_fidelity = shared_arm._class_fidelity_evidence(
            Path(class_fidelity_report_path),
            arm=arm,
            sampling=sampling,
            expected_git=expected_git,
        )
    _require_generation_runtime(
        sampling,
        arm=arm,
        expected_runtime_environment_sha256=str(
            validated_launch["runtime_environment_sha256"]
        ),
    )
    _require_matching_evaluator_runtime(distribution, class_fidelity, arm=arm)
    screen_sampling = _object(
        screen_validation.get("sampling"), f"{arm} screen sampling"
    )
    screen_sample_sha = screen_sampling.get("sample_set_sha256")
    if (
        not isinstance(screen_sample_sha, str)
        or len(screen_sample_sha) != 64
        or screen_sample_sha != screen_sample_sha.lower()
        or any(character not in "0123456789abcdef" for character in screen_sample_sha)
        or sampling.get("sample_set_sha256") == screen_sample_sha
    ):
        raise ValueError(f"{arm} confirmation sample stream is not distinct")
    sources = {
        "launch_receipt": shared_arm._identity(
            launch_path, "terminal-SNR confirmation launch receipt"
        ),
        "screen_arm_validation": screen_identity,
        "sampling_preflight": preflight["report"],
        "sampling_report": sampling["report"],
        "metrics_report": distribution["report"],
        "class_fidelity_report": class_fidelity["report"],
    }
    report = {
        "schema_version": ARM_VALIDATION_SCHEMA,
        "role": ARM_VALIDATION_ROLE,
        "status": "pass",
        "arm": arm,
        "condition": spec["condition"],
        "method": spec["method"],
        "endpoint_fraction": spec["endpoint_fraction"],
        "execution_git": copy.deepcopy(execution),
        "screen_execution_git": copy.deepcopy(screen_validation["execution_git"]),
        "sources": sources,
        "frozen_training": {
            "training_performed": False,
            "checkpoint": copy.deepcopy(checkpoint),
            "screen_training": copy.deepcopy(screen_validation["training"]),
        },
        "sampling": sampling,
        "sampling_preflight": preflight,
        "screen_sample_set_sha256": screen_sample_sha,
        "confirmation_sample_set_distinct": True,
        "distribution": distribution,
        "class_fidelity": class_fidelity,
        "screen_checkpoint_evaluation": copy.deepcopy(
            screen_validation["checkpoint_evaluation"]
        ),
        "screen_rollout": copy.deepcopy(screen_validation["rollout"]),
        "authorization_boundary": copy.deepcopy(ARM_VALIDATION_BOUNDARY),
    }
    return report


def replay_terminal_snr_confirmation_arm_validation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation arm validation")
    sources = _object(row.get("sources"), "terminal-SNR confirmation arm sources")
    required = {
        "launch_receipt",
        "screen_arm_validation",
        "sampling_preflight",
        "sampling_report",
        "metrics_report",
        "class_fidelity_report",
    }
    if set(sources) != required:
        raise ValueError("terminal-SNR confirmation arm source set differs")
    for name in required:
        identity = _object(sources[name], f"terminal-SNR confirmation source {name}")
        if set(identity) != {"path", "bytes", "sha256"}:
            raise ValueError(
                f"terminal-SNR confirmation source identity differs: {name}"
            )
    return build_terminal_snr_confirmation_arm_validation(
        arm=str(row.get("arm", "")),
        launch_receipt_path=sources["launch_receipt"]["path"],
        expected_launch_receipt_sha256=sources["launch_receipt"]["sha256"],
        screen_arm_validation_path=sources["screen_arm_validation"]["path"],
        expected_screen_arm_validation_sha256=sources["screen_arm_validation"][
            "sha256"
        ],
        sampling_preflight_report_path=sources["sampling_preflight"]["path"],
        sampling_report_path=sources["sampling_report"]["path"],
        metrics_report_path=sources["metrics_report"]["path"],
        class_fidelity_report_path=sources["class_fidelity_report"]["path"],
    )


def validate_terminal_snr_confirmation_arm_validation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation arm validation")
    if (
        row.get("schema_version") != ARM_VALIDATION_SCHEMA
        or row.get("role") != ARM_VALIDATION_ROLE
        or row.get("status") != "pass"
        or row.get("arm") not in ARM_NAMES
        or row.get("authorization_boundary") != ARM_VALIDATION_BOUNDARY
        or row.get("confirmation_sample_set_distinct") is not True
        or _object(
            row.get("frozen_training"), "terminal-SNR confirmation frozen training"
        ).get("training_performed")
        is not False
    ):
        raise ValueError("terminal-SNR confirmation arm validation contract differs")
    expected = replay_terminal_snr_confirmation_arm_validation(row)
    if row != expected:
        raise ValueError(
            "terminal-SNR confirmation arm validation is not exact replay"
        )
    return copy.deepcopy(row)


__all__ = [
    "ARM_VALIDATION_BOUNDARY",
    "ARM_VALIDATION_ROLE",
    "ARM_VALIDATION_SCHEMA",
    "build_terminal_snr_confirmation_arm_validation",
    "replay_terminal_snr_confirmation_arm_validation",
    "validate_terminal_snr_confirmation_sampling_preflight",
    "validate_terminal_snr_confirmation_arm_validation",
]
