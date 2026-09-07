"""Non-authorizing preparation for the terminal-SNR frozen confirmation.

The directional screen evaluates 1K samples from four freshly trained step-10K
checkpoints.  A passing screen may freeze those exact checkpoints for a new,
disjoint 10K-sample evaluation.  This module binds that selection, but it does
not authorize GPU use, sampling, training, full-scale training, or release.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation.terminal_snr_screen import (
    ARM_NAMES,
    ARM_SPECS,
    SAMPLE_SEED as SCREEN_SAMPLE_SEED,
    SCREEN_THRESHOLDS,
    STOP_STEP,
)
from cofitok.generation.terminal_snr_screen_arm import (
    validate_terminal_snr_screen_arm_validation,
)
from cofitok.generation.terminal_snr_screen_execution import (
    validate_terminal_snr_screen_launch_receipt_physical,
)
from cofitok.generation.terminal_snr_screen_result import (
    replay_terminal_snr_screen_result,
    validate_terminal_snr_screen_result_contract,
    validate_terminal_snr_screen_validation_receipt,
)


PREPARATION_SCHEMA = (
    "cofitok_generation_terminal_snr_confirmation_preparation_v1"
)
PREPARATION_ROLE = (
    "source_bound_frozen_four_arm_terminal_snr_confirmation_preparation"
)
CONFIRMATION_DIRNAME = "frozen_confirmation_10000"
SAMPLE_COUNT = 10_000
SAMPLE_STEPS = 100
SAMPLE_BATCH_SIZE = 4
SAMPLE_SEED = 2028
GUIDANCE_SCALE = 1.5
GUIDANCE_RESCALE = 0.0

PREPARATION_BOUNDARY = {
    "terminal_snr_confirmation_prepared": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "checkpoint_mutation_allowed": False,
    "large_capacity_readiness_preparation_allowed": False,
    "large_capacity_readiness_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
EVALUATION_CONTRACT = {
    "arms": list(ARM_NAMES),
    "checkpoint_step": STOP_STEP,
    "frozen_checkpoint_training_allowed": False,
    "weights": "ema",
    "sampler": "ddim",
    "sample_steps": SAMPLE_STEPS,
    "samples_per_arm": SAMPLE_COUNT,
    "sampling_batch_size": SAMPLE_BATCH_SIZE,
    "per_arm_real_forward_preflight_required": True,
    "preflight_warmup_forwards": 0,
    "preflight_measured_forwards": 1,
    "preflight_timestep": 999,
    "guidance_scale": GUIDANCE_SCALE,
    "guidance_rescale": GUIDANCE_RESCALE,
    "cfg_batch_mode": "batched",
    "eta": 0.0,
    "clip_x0": True,
    "precision": "bf16",
    "seed": SAMPLE_SEED,
    "screen_seed": SCREEN_SAMPLE_SEED,
    "sampling_seed_disjoint_from_screen": SAMPLE_SEED != SCREEN_SAMPLE_SEED,
    "class_schedule": "balanced_modulo",
    "requested_samples_per_class": 10,
    "fixed_random_stream_across_arms": True,
    "fid_is_precision_recall_required": True,
    "class_fidelity_required": True,
    "screen_mechanism_and_rollout_physical_replay_required": True,
    "primary_estimand": "within_method_endpoint0975_minus_control",
}
DECISION_CONTRACT = {
    "all_screen_thresholds_reapplied_to_10k_metrics": True,
    "screen_mechanism_and_rollout_must_physically_replay": True,
    "confirmation_may_only_prepare_large_capacity_readiness": True,
    "large_capacity_readiness_requires_separate_authorization": True,
    "confirmation_cannot_authorize_full_300k": True,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    if (
        not isinstance(row.get("path"), str)
        or not row["path"]
        or not isinstance(row.get("bytes"), int)
        or isinstance(row.get("bytes"), bool)
        or row["bytes"] < 1
        or not _hex(row.get("sha256"), 64)
    ):
        raise ValueError(f"{name} identity is malformed")
    return {key: row[key] for key in ("path", "bytes", "sha256")}


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    if (
        not _hex(row.get("revision"), 40)
        or not _hex(row.get("tree"), 40)
        or not isinstance(row.get("branch"), str)
        or not row["branch"]
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} must identify one exact clean checkout")
    return {
        "revision": row["revision"],
        "tree": row["tree"],
        "branch": row["branch"],
        "tracked_dirty": False,
    }


def _absolute(value: Any, name: str) -> str:
    text = str(value or "")
    if not text or not (Path(text).is_absolute() or PurePosixPath(text).is_absolute()):
        raise ValueError(f"{name} must be absolute")
    return text


def _expected_confirmation_root(screen_root: str) -> str:
    return str(PurePosixPath(screen_root) / CONFIRMATION_DIRNAME)


def _checkpoint_summary(report: Mapping[str, Any], arm: str) -> dict[str, Any]:
    training = _object(report.get("training"), f"{arm} screen training")
    validation = _object(
        training.get("validation"), f"{arm} screen training validation"
    )
    checkpoint = _object(training.get("checkpoint"), f"{arm} frozen checkpoint")
    integrity = _identity(
        checkpoint.get("integrity_manifest"), f"{arm} checkpoint integrity"
    )
    if (
        int(validation.get("completed_steps", -1)) != STOP_STEP
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint["path"]
        or not isinstance(checkpoint.get("bytes"), int)
        or isinstance(checkpoint.get("bytes"), bool)
        or checkpoint["bytes"] < 1
        or not _hex(checkpoint.get("sha256"), 64)
        or checkpoint.get("integrity_manifest") != integrity
    ):
        raise ValueError(f"{arm} frozen checkpoint differs")
    return copy.deepcopy(checkpoint)


def build_terminal_snr_confirmation_preparation(
    *,
    screen_result: Mapping[str, Any],
    screen_result_identity: Mapping[str, Any],
    screen_result_validation: Mapping[str, Any],
    screen_result_validation_identity: Mapping[str, Any],
    screen_launch_receipt: Mapping[str, Any],
    screen_launch_receipt_identity: Mapping[str, Any],
    screen_arm_validations: Mapping[str, Mapping[str, Any]],
    screen_arm_validation_identities: Mapping[str, Mapping[str, Any]],
    preparation_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    result = validate_terminal_snr_screen_result_contract(screen_result)
    if replay_terminal_snr_screen_result(result) != result:
        raise ValueError("terminal-SNR screen result differs from physical replay")
    result_id = _identity(screen_result_identity, "terminal-SNR screen result")
    result_validation_id = _identity(
        screen_result_validation_identity,
        "terminal-SNR screen result validation",
    )
    result_validation = validate_terminal_snr_screen_validation_receipt(
        screen_result_validation,
        result=result,
        result_identity=result_id,
        validator_git=_git(
            screen_result_validation.get("validator_git"),
            "terminal-SNR screen result validator",
        ),
    )
    if (
        result_validation.get("result") != result_id
        or result_validation.get("screen_pass") is not True
        or result_validation.get("failed_checks") != []
    ):
        raise ValueError("terminal-SNR screen result validation differs")
    launch_id = _identity(
        screen_launch_receipt_identity, "terminal-SNR screen launch receipt"
    )
    next_stage = _object(result.get("next_stage"), "terminal-SNR screen next stage")
    if (
        result.get("scientific_status") != "pass"
        or result.get("screen_pass") is not True
        or result.get("failed_checks") != []
        or next_stage.get("route") != "frozen_10k_confirmation_preparation"
        or next_stage.get("frozen_confirmation_preparation_allowed") is not True
        or next_stage.get("frozen_confirmation_launch_allowed") is not False
    ):
        raise ValueError(
            "terminal-SNR confirmation requires a passing terminal-SNR screen"
        )
    execution = _git(
        screen_launch_receipt.get("execution_checkout"),
        "terminal-SNR screen execution checkout",
    )
    launch = validate_terminal_snr_screen_launch_receipt_physical(
        screen_launch_receipt,
        expected_execution_checkout=execution,
    )
    if result.get("result_git") != execution:
        raise ValueError("terminal-SNR screen result Git differs from launch")
    result_sources = _object(
        result.get("source_evidence"), "terminal-SNR screen result sources"
    )
    if result_sources.get("launch_receipt") != launch_id:
        raise ValueError("terminal-SNR confirmation launch source differs from screen")
    if set(screen_arm_validations) != set(ARM_NAMES) or set(
        screen_arm_validation_identities
    ) != set(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation screen arm set differs")
    expected_arm_ids = _object(
        result_sources.get("arm_validations"), "terminal-SNR screen arm identities"
    )

    frozen_arms: dict[str, dict[str, Any]] = {}
    normalized_arm_ids: dict[str, dict[str, Any]] = {}
    checkpoint_shas: set[str] = set()
    for arm in ARM_NAMES:
        arm_id = _identity(
            screen_arm_validation_identities[arm], f"{arm} screen validation"
        )
        if expected_arm_ids.get(arm) != arm_id:
            raise ValueError(f"{arm} validation differs from terminal-SNR result")
        validation = validate_terminal_snr_screen_arm_validation(
            screen_arm_validations[arm]
        )
        if validation.get("arm") != arm or validation.get("execution_git") != execution:
            raise ValueError(f"{arm} screen validation uses another checkout")
        spec = ARM_SPECS[arm]
        if (
            validation.get("condition") != spec["condition"]
            or validation.get("method") != spec["method"]
            or validation.get("endpoint_fraction") != spec["endpoint_fraction"]
        ):
            raise ValueError(f"{arm} terminal-SNR selection differs")
        sources = _object(validation.get("sources"), f"{arm} screen sources")
        checkpoint = _checkpoint_summary(validation, arm)
        checkpoint_shas.add(str(checkpoint["sha256"]))
        frozen_arms[arm] = {
            "condition": spec["condition"],
            "method": spec["method"],
            "endpoint_fraction": spec["endpoint_fraction"],
            "parameter_count": spec["parameter_count"],
            "prefix_budget": spec["prefix_budget"],
            "checkpoint_step": STOP_STEP,
            "screen_arm_validation": arm_id,
            "config": _identity(sources.get("config"), f"{arm} config"),
            "training_report": _identity(
                sources.get("training_report"), f"{arm} training report"
            ),
            "checkpoint": checkpoint,
            "checkpoint_evaluation": copy.deepcopy(
                validation["checkpoint_evaluation"]
            ),
            "rollout": copy.deepcopy(validation["rollout"]),
        }
        normalized_arm_ids[arm] = arm_id
    if len(checkpoint_shas) != len(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation frozen checkpoints are duplicated")

    screen_root = _absolute(launch.get("output_root"), "terminal-SNR screen root")
    root = _absolute(output_root, "terminal-SNR confirmation output root")
    if root != _expected_confirmation_root(screen_root):
        raise ValueError("terminal-SNR confirmation output root differs")
    evaluation_contract = copy.deepcopy(EVALUATION_CONTRACT)
    report = {
        "schema_version": PREPARATION_SCHEMA,
        "role": PREPARATION_ROLE,
        "status": "prepared",
        "scientific_status": "terminal_snr_confirmation_prepared",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation_git": _git(preparation_git, "terminal-SNR confirmation preparation"),
        "source_evidence": {
            "terminal_snr_screen_result": result_id,
            "terminal_snr_screen_result_validation": result_validation_id,
            "terminal_snr_screen_launch_receipt": launch_id,
            "terminal_snr_screen_arm_validations": normalized_arm_ids,
        },
        "screen_execution_checkout": execution,
        "selection": {
            "dataset": "imagenet_256",
            "screen_output_root": screen_root,
            "output_root": root,
            "fresh_training_allowed": False,
            "frozen_training_step": STOP_STEP,
            "frozen_arms": list(ARM_NAMES),
            "single_scientific_config_field": "diffusion.cosine_endpoint_fraction",
        },
        "frozen_arms": frozen_arms,
        "evaluation_contract": evaluation_contract,
        "thresholds": copy.deepcopy(SCREEN_THRESHOLDS),
        "decision_contract": copy.deepcopy(DECISION_CONTRACT),
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }
    return validate_terminal_snr_confirmation_preparation_contract(
        report, expected_output_root=root
    )


def validate_terminal_snr_confirmation_preparation_contract(
    report: Mapping[str, Any], *, expected_output_root: str | None = None
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation preparation")
    selection = _object(row.get("selection"), "terminal-SNR confirmation selection")
    evaluation = _object(
        row.get("evaluation_contract"), "terminal-SNR confirmation evaluation"
    )
    decision = _object(
        row.get("decision_contract"), "terminal-SNR confirmation decision"
    )
    root = _absolute(selection.get("output_root"), "terminal-SNR confirmation root")
    screen_root = _absolute(
        selection.get("screen_output_root"), "terminal-SNR screen root"
    )
    if root != _expected_confirmation_root(screen_root):
        raise ValueError("terminal-SNR confirmation output root differs")
    if expected_output_root is not None and root != expected_output_root:
        raise ValueError("terminal-SNR confirmation expected output root differs")
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "scientific_status",
            "terminal_status",
            "generation_advantage_proven",
            "preparation_git",
            "source_evidence",
            "screen_execution_checkout",
            "selection",
            "frozen_arms",
            "evaluation_contract",
            "thresholds",
            "decision_contract",
            "authorization_boundary",
        }
        or set(selection)
        != {
            "dataset",
            "screen_output_root",
            "output_root",
            "fresh_training_allowed",
            "frozen_training_step",
            "frozen_arms",
            "single_scientific_config_field",
        }
        or evaluation != EVALUATION_CONTRACT
        or decision != DECISION_CONTRACT
        or row.get("schema_version") != PREPARATION_SCHEMA
        or row.get("role") != PREPARATION_ROLE
        or row.get("status") != "prepared"
        or row.get("scientific_status") != "terminal_snr_confirmation_prepared"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("authorization_boundary") != PREPARATION_BOUNDARY
        or row.get("thresholds") != SCREEN_THRESHOLDS
        or selection.get("dataset") != "imagenet_256"
        or selection.get("fresh_training_allowed") is not False
        or int(selection.get("frozen_training_step", -1)) != STOP_STEP
        or selection.get("frozen_arms") != list(ARM_NAMES)
        or selection.get("single_scientific_config_field")
        != "diffusion.cosine_endpoint_fraction"
        or evaluation.get("arms") != list(ARM_NAMES)
        or int(evaluation.get("checkpoint_step", -1)) != STOP_STEP
        or evaluation.get("frozen_checkpoint_training_allowed") is not False
        or int(evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or int(evaluation.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(evaluation.get("sampling_batch_size", -1)) != SAMPLE_BATCH_SIZE
        or int(evaluation.get("seed", -1)) != SAMPLE_SEED
        or int(evaluation.get("screen_seed", -1)) != SCREEN_SAMPLE_SEED
        or evaluation.get("sampling_seed_disjoint_from_screen") is not True
        or evaluation.get("weights") != "ema"
        or evaluation.get("fixed_random_stream_across_arms") is not True
        or evaluation.get("fid_is_precision_recall_required") is not True
        or evaluation.get("class_fidelity_required") is not True
        or evaluation.get("screen_mechanism_and_rollout_physical_replay_required")
        is not True
        or evaluation.get("primary_estimand")
        != "within_method_endpoint0975_minus_control"
        or decision.get("all_screen_thresholds_reapplied_to_10k_metrics") is not True
        or decision.get("screen_mechanism_and_rollout_must_physically_replay")
        is not True
        or decision.get("confirmation_may_only_prepare_large_capacity_readiness")
        is not True
        or decision.get("large_capacity_readiness_requires_separate_authorization")
        is not True
        or decision.get("confirmation_cannot_authorize_full_300k") is not True
    ):
        raise ValueError("terminal-SNR confirmation preparation contract differs")
    _git(row.get("preparation_git"), "terminal-SNR confirmation preparation")
    _git(row.get("screen_execution_checkout"), "terminal-SNR screen execution")
    sources = _object(
        row.get("source_evidence"), "terminal-SNR confirmation preparation sources"
    )
    if set(sources) != {
        "terminal_snr_screen_result",
        "terminal_snr_screen_result_validation",
        "terminal_snr_screen_launch_receipt",
        "terminal_snr_screen_arm_validations",
    }:
        raise ValueError("terminal-SNR confirmation source set differs")
    _identity(sources["terminal_snr_screen_result"], "terminal-SNR screen result")
    _identity(
        sources["terminal_snr_screen_result_validation"],
        "terminal-SNR screen result validation",
    )
    _identity(
        sources["terminal_snr_screen_launch_receipt"],
        "terminal-SNR screen launch receipt",
    )
    arm_sources = _object(
        sources["terminal_snr_screen_arm_validations"],
        "terminal-SNR screen arm validations",
    )
    frozen = _object(row.get("frozen_arms"), "terminal-SNR frozen arms")
    if set(arm_sources) != set(ARM_NAMES) or set(frozen) != set(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation frozen arm set differs")
    checkpoint_shas: set[str] = set()
    for arm in ARM_NAMES:
        source_id = _identity(arm_sources[arm], f"{arm} screen validation")
        item = _object(frozen[arm], f"{arm} frozen arm")
        spec = ARM_SPECS[arm]
        if (
            set(item)
            != {
                "condition",
                "method",
                "endpoint_fraction",
                "parameter_count",
                "prefix_budget",
                "checkpoint_step",
                "screen_arm_validation",
                "config",
                "training_report",
                "checkpoint",
                "checkpoint_evaluation",
                "rollout",
            }
            or item.get("condition") != spec["condition"]
            or item.get("method") != spec["method"]
            or item.get("endpoint_fraction") != spec["endpoint_fraction"]
            or int(item.get("parameter_count", -1)) != spec["parameter_count"]
            or int(item.get("prefix_budget", -1)) != spec["prefix_budget"]
            or int(item.get("checkpoint_step", -1)) != STOP_STEP
            or item.get("screen_arm_validation") != source_id
        ):
            raise ValueError(f"{arm} frozen selection differs")
        _identity(item.get("config"), f"{arm} frozen config")
        _identity(item.get("training_report"), f"{arm} frozen training report")
        _object(item.get("checkpoint_evaluation"), f"{arm} checkpoint evaluation")
        _object(item.get("rollout"), f"{arm} rollout")
        checkpoint = _object(item.get("checkpoint"), f"{arm} frozen checkpoint")
        _identity(checkpoint.get("integrity_manifest"), f"{arm} checkpoint integrity")
        if (
            not isinstance(checkpoint.get("path"), str)
            or not checkpoint["path"]
            or not isinstance(checkpoint.get("bytes"), int)
            or checkpoint["bytes"] < 1
            or not _hex(checkpoint.get("sha256"), 64)
        ):
            raise ValueError(f"{arm} frozen checkpoint differs")
        checkpoint_shas.add(checkpoint["sha256"])
    if len(checkpoint_shas) != len(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation frozen checkpoints are duplicated")
    return copy.deepcopy(row)


__all__ = [
    "CONFIRMATION_DIRNAME",
    "DECISION_CONTRACT",
    "EVALUATION_CONTRACT",
    "GUIDANCE_RESCALE",
    "GUIDANCE_SCALE",
    "PREPARATION_BOUNDARY",
    "PREPARATION_ROLE",
    "PREPARATION_SCHEMA",
    "SAMPLE_BATCH_SIZE",
    "SAMPLE_COUNT",
    "SAMPLE_SEED",
    "SAMPLE_STEPS",
    "build_terminal_snr_confirmation_preparation",
    "validate_terminal_snr_confirmation_preparation_contract",
]
