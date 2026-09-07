"""Physical, replayable validation for one terminal-SNR screen arm."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from cofitok.generation import capacity_screen_arm as shared_arm
from cofitok.generation.capacity_qualification_training import (
    validate_capacity_qualification_partial_training,
)
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_screen import (
    ARM_NAMES,
    ARM_SPECS,
    SAMPLE_SEED,
)
from cofitok.generation.terminal_snr_screen_execution import (
    LAUNCH_RECEIPT_ROLE,
    LAUNCH_RECEIPT_SCHEMA,
    validate_terminal_snr_screen_launch_receipt_physical,
)
from cofitok.inference_replay import file_identity, reject_symlink_chain
from cofitok.reporting import file_sha256


ARM_VALIDATION_SCHEMA = "cofitok_generation_terminal_snr_screen_arm_validation_v1"
ARM_VALIDATION_ROLE = "physical_terminal_snr_screen_arm_validation"
ARM_VALIDATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "frozen_confirmation_allowed": False,
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


def _finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    return shared_arm._finite(
        value, name, minimum=minimum, maximum=maximum
    )


def _identity(path: str | Path, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    return file_identity(source)


@contextmanager
def _terminal_helper_contract() -> Iterator[None]:
    """Bind proven physical helpers to this screen's arm set and seed.

    The capacity and endpoint screens deliberately share every evaluation
    mechanism.  Only arm labels/configs and the predeclared seed differ.  The
    binding is scoped and restored so importing this module cannot weaken the
    original capacity-screen validators.
    """

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


def _terminal_clip_fraction(report_path: Path, arm: str) -> dict[str, Any]:
    report = read_object(report_path, name=f"{arm} rollout report")
    free = _object(report.get("free_sampling_rollout"), f"{arm} free rollout")
    steps = free.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{arm} free rollout has no terminal step")
    first = _object(steps[0], f"{arm} terminal free-rollout step")
    fraction = _finite(
        first.get("raw_x0_clip_fraction"),
        f"{arm} terminal raw-x0 clip fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if int(first.get("step_index", -1)) != 0 or int(
        first.get("timestep", -1)
    ) != 999:
        raise ValueError(f"{arm} terminal clip source is not rollout step 0 at t=999")
    return {
        "source": "free_sampling_rollout.steps[0].raw_x0_clip_fraction",
        "step_index": 0,
        "timestep": 999,
        "raw_x0_clip_fraction": fraction,
    }


def _require_launch_runtime(
    evidence: Mapping[str, Any],
    *,
    arm: str,
    evidence_name: str,
    expected_runtime_environment_sha256: str,
) -> None:
    if (
        evidence.get("runtime_environment_sha256")
        != expected_runtime_environment_sha256
    ):
        raise ValueError(
            f"{arm} {evidence_name} runtime differs from launch selection"
        )


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
        raise ValueError(f"{arm} evaluator runtime identities differ")


def build_terminal_snr_screen_arm_validation(
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
        raise ValueError("terminal-SNR screen arm name differs")
    spec = ARM_SPECS[arm]
    launch_path = reject_symlink_chain(
        launch_receipt_path, name="terminal-SNR launch receipt"
    ).resolve()
    if file_sha256(launch_path) != expected_launch_receipt_sha256:
        raise ValueError("terminal-SNR launch receipt SHA256 differs")
    launch = read_object(launch_path, name="terminal-SNR launch receipt")
    if (
        launch.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or launch.get("role") != LAUNCH_RECEIPT_ROLE
    ):
        raise ValueError("terminal-SNR launch receipt schema differs")
    execution = _object(launch.get("execution_checkout"), "terminal-SNR execution")
    validated_launch = validate_terminal_snr_screen_launch_receipt_physical(
        launch, expected_execution_checkout=execution
    )
    expected_git = shared_arm._expected_report_git(execution)
    config_file = reject_symlink_chain(config_path, name=f"{arm} config").resolve()
    config_id = _identity(config_file, f"{arm} config")
    launch_configs = _object(
        _object(validated_launch.get("source_evidence"), "terminal-SNR sources").get(
            "configs"
        ),
        "terminal-SNR launch configs",
    )
    if config_id != launch_configs.get(arm):
        raise ValueError(f"{arm} config differs from launch receipt")
    run_dir = Path(
        str(_object(validated_launch.get("run_dirs"), "terminal-SNR runs")[arm])
    )
    training_report = reject_symlink_chain(
        training_report_path, name=f"{arm} training report"
    ).resolve()
    if training_report.parent.resolve() != run_dir.resolve():
        raise ValueError(f"{arm} training report is outside its authorized run")
    runtime = _object(validated_launch.get("runtime_selection"), "terminal-SNR runtime")
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
    if training["runtime_environment_sha256"] != runtime[
        "runtime_environment_sha256"
    ]:
        raise ValueError(f"{arm} training runtime differs from launch selection")
    raw_training = read_object(training_report, name=f"{arm} training report")
    config = _object(raw_training.get("config"), f"{arm} resolved training config")
    final_metrics = _object(raw_training.get("final_metrics"), f"{arm} final metrics")
    validation_epsilon = _finite(
        final_metrics.get("validation_epsilon_mse"),
        f"{arm} final validation epsilon MSE",
        minimum=0.0,
    )
    checkpoint = _object(training.get("checkpoint"), f"{arm} checkpoint")
    with _terminal_helper_contract():
        sampling = shared_arm._sampling_evidence(
            Path(sampling_report_path),
            arm=arm,
            expected_checkpoint=checkpoint,
            expected_git=expected_git,
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
        checkpoint_evaluation = shared_arm._checkpoint_evidence(
            Path(checkpoint_evaluation_report_path),
            arm=arm,
            expected_checkpoint=checkpoint,
            expected_config=config,
            expected_git=expected_git,
        )
        rollout = shared_arm._rollout_evidence(
            Path(rollout_report_path),
            arm=arm,
            expected_checkpoint=checkpoint,
            expected_config=config,
            expected_git=expected_git,
        )
    expected_runtime = str(runtime["runtime_environment_sha256"])
    _require_launch_runtime(
        sampling,
        arm=arm,
        evidence_name="sampling",
        expected_runtime_environment_sha256=expected_runtime,
    )
    _require_matching_evaluator_runtime(
        distribution,
        class_fidelity,
        arm=arm,
    )
    rollout["terminal_raw_x0_clipping"] = _terminal_clip_fraction(
        Path(rollout_report_path), arm
    )
    sources = {
        "launch_receipt": _identity(launch_path, "terminal-SNR launch receipt"),
        "config": config_id,
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
        "condition": spec["condition"],
        "method": spec["method"],
        "endpoint_fraction": spec["endpoint_fraction"],
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


def replay_terminal_snr_screen_arm_validation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR arm validation")
    sources = _object(row.get("sources"), "terminal-SNR arm sources")
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
        raise ValueError("terminal-SNR arm source set differs")
    return build_terminal_snr_screen_arm_validation(
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


def validate_terminal_snr_screen_arm_validation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR arm validation")
    if (
        row.get("schema_version") != ARM_VALIDATION_SCHEMA
        or row.get("role") != ARM_VALIDATION_ROLE
        or row.get("status") != "pass"
        or row.get("arm") not in ARM_NAMES
        or row.get("authorization_boundary") != ARM_VALIDATION_BOUNDARY
    ):
        raise ValueError("terminal-SNR arm validation contract differs")
    expected = replay_terminal_snr_screen_arm_validation(row)
    if row != expected:
        raise ValueError("terminal-SNR arm validation is not an exact replay")
    return copy.deepcopy(row)


__all__ = [
    "ARM_VALIDATION_BOUNDARY",
    "ARM_VALIDATION_ROLE",
    "ARM_VALIDATION_SCHEMA",
    "build_terminal_snr_screen_arm_validation",
    "replay_terminal_snr_screen_arm_validation",
    "validate_terminal_snr_screen_arm_validation",
]
