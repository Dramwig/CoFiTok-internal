"""Source-bound preparation for the four-arm 10K capacity confirmation.

The 1K capacity screen is directional evidence only.  This module freezes the
four exact step-10K checkpoints selected by a passing screen and prepares a
larger 10K-sample confirmation over those unchanged checkpoints.  Preparation
is CPU-only and cannot authorize sampling, training, full 300K, or release.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation.capacity_screen import ARM_NAMES, ARM_SPECS, STOP_STEP
from cofitok.generation.capacity_screen_arm import (
    validate_capacity_screen_arm_validation,
)
from cofitok.generation.capacity_screen_execution import (
    validate_capacity_screen_launch_receipt_contract,
)
from cofitok.generation.capacity_screen_result import (
    validate_capacity_screen_result_contract,
)


PREPARATION_SCHEMA = "cofitok_generation_capacity_confirmation_preparation_v1"
PREPARATION_ROLE = "source_bound_four_arm_capacity_confirmation_preparation"
CONFIRMATION_DIRNAME = "confirmation_10000"
SAMPLE_COUNT = 10_000
SAMPLE_STEPS = 100
SAMPLE_BATCH_SIZE = 4
SAMPLE_SEED = 0
GUIDANCE_SCALE = 1.5
GUIDANCE_RESCALE = 0.0

PREPARATION_BOUNDARY = {
    "capacity_confirmation_prepared": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "large_capacity_readiness_preparation_allowed": False,
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
    checkpoint = _object(training.get("checkpoint"), f"{arm} frozen checkpoint")
    integrity = _object(
        checkpoint.get("integrity_manifest"), f"{arm} checkpoint integrity"
    )
    if (
        int(checkpoint.get("step", -1)) != STOP_STEP
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint["path"]
        or not isinstance(checkpoint.get("bytes"), int)
        or checkpoint["bytes"] < 1
        or not _hex(checkpoint.get("sha256"), 64)
        or set(integrity) != {"path", "bytes", "sha256"}
    ):
        raise ValueError(f"{arm} frozen checkpoint differs")
    _identity(integrity, f"{arm} checkpoint integrity")
    return copy.deepcopy(checkpoint)


def build_capacity_confirmation_preparation(
    *,
    screen_result: Mapping[str, Any],
    screen_result_identity: Mapping[str, Any],
    screen_launch_receipt: Mapping[str, Any],
    screen_launch_receipt_identity: Mapping[str, Any],
    screen_arm_validations: Mapping[str, Mapping[str, Any]],
    screen_arm_validation_identities: Mapping[str, Mapping[str, Any]],
    preparation_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    result = validate_capacity_screen_result_contract(screen_result)
    if (
        result.get("scientific_status") != "screen_pass"
        or result.get("failed_checks") != []
        or _object(result.get("next_stage"), "capacity screen next stage").get(
            "capacity_confirmation_preparation_allowed"
        )
        is not True
    ):
        raise ValueError("capacity confirmation requires a passing capacity screen")
    result_id = _identity(screen_result_identity, "capacity screen result")
    launch_id = _identity(
        screen_launch_receipt_identity, "capacity screen launch receipt"
    )
    execution = _git(
        screen_launch_receipt.get("execution_checkout"),
        "capacity screen execution checkout",
    )
    launch = validate_capacity_screen_launch_receipt_contract(
        screen_launch_receipt,
        expected_execution_checkout=execution,
    )
    result_sources = _object(
        result.get("source_evidence"), "capacity screen result sources"
    )
    if result_sources.get("launch_receipt") != launch_id:
        raise ValueError("capacity confirmation launch source differs from screen")
    if set(screen_arm_validations) != set(ARM_NAMES) or set(
        screen_arm_validation_identities
    ) != set(ARM_NAMES):
        raise ValueError("capacity confirmation screen arm set differs")
    expected_arm_ids = _object(
        result_sources.get("arm_validations"), "capacity screen arm identities"
    )
    frozen_arms: dict[str, dict[str, Any]] = {}
    normalized_arm_ids: dict[str, dict[str, Any]] = {}
    checkpoint_shas: set[str] = set()
    for arm in ARM_NAMES:
        arm_id = _identity(
            screen_arm_validation_identities[arm], f"{arm} screen validation"
        )
        if expected_arm_ids.get(arm) != arm_id:
            raise ValueError(f"{arm} validation differs from capacity screen result")
        validation = validate_capacity_screen_arm_validation(
            screen_arm_validations[arm]
        )
        if validation.get("execution_git") != execution:
            raise ValueError(f"{arm} screen validation uses another checkout")
        sources = _object(validation.get("sources"), f"{arm} screen sources")
        checkpoint = _checkpoint_summary(validation, arm)
        checkpoint_shas.add(str(checkpoint["sha256"]))
        frozen_arms[arm] = {
            "capacity": ARM_SPECS[arm]["capacity"],
            "method": ARM_SPECS[arm]["method"],
            "parameter_count": ARM_SPECS[arm]["parameter_count"],
            "prefix_budget": ARM_SPECS[arm]["prefix_budget"],
            "screen_arm_validation": arm_id,
            "config": _identity(sources.get("config"), f"{arm} config"),
            "training_report": _identity(
                sources.get("training_report"), f"{arm} training report"
            ),
            "checkpoint": checkpoint,
            "checkpoint_evaluation": copy.deepcopy(
                validation["checkpoint_evaluation"]["summary"]
            ),
            "rollout": copy.deepcopy(validation["rollout"]["summary"]),
        }
        normalized_arm_ids[arm] = arm_id
    if len(checkpoint_shas) != len(ARM_NAMES):
        raise ValueError("capacity confirmation frozen checkpoints are duplicated")

    screen_root = _absolute(launch.get("output_root"), "capacity screen root")
    root = _absolute(output_root, "capacity confirmation output root")
    if root != _expected_confirmation_root(screen_root):
        raise ValueError("capacity confirmation output root differs")
    evaluation_contract = {
        "arms": list(ARM_NAMES),
        "checkpoint_step": STOP_STEP,
        "frozen_checkpoint_training_allowed": False,
        "weights": "ema",
        "sampler": "ddim",
        "sample_steps": SAMPLE_STEPS,
        "samples_per_arm": SAMPLE_COUNT,
        "sampling_batch_size": SAMPLE_BATCH_SIZE,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "precision": "bf16",
        "seed": SAMPLE_SEED,
        "class_schedule": "balanced_modulo",
        "requested_samples_per_class": 10,
        "fixed_random_stream_across_arms": True,
        "fid_is_precision_recall_required": True,
        "class_fidelity_required": True,
        "screen_mechanism_and_rollout_replay_required": True,
    }
    report = {
        "schema_version": PREPARATION_SCHEMA,
        "role": PREPARATION_ROLE,
        "status": "prepared",
        "scientific_status": "capacity_confirmation_prepared",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation_git": _git(preparation_git, "capacity confirmation preparation"),
        "source_evidence": {
            "capacity_screen_result": result_id,
            "capacity_screen_launch_receipt": launch_id,
            "capacity_screen_arm_validations": normalized_arm_ids,
        },
        "screen_execution_checkout": execution,
        "selection": {
            "dataset": "imagenet_256",
            "screen_output_root": screen_root,
            "output_root": root,
            "fresh_training_allowed": False,
            "frozen_training_step": STOP_STEP,
            "frozen_arms": list(ARM_NAMES),
        },
        "frozen_arms": frozen_arms,
        "evaluation_contract": evaluation_contract,
        "decision_contract": {
            "absolute_support_and_class_gates_required": True,
            "confirmation_may_only_prepare_large_capacity_readiness": True,
            "large_capacity_readiness_requires_separate_authorization": True,
            "confirmation_cannot_authorize_full_300k": True,
        },
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }
    validate_capacity_confirmation_preparation_contract(
        report, expected_output_root=root
    )
    return report


def validate_capacity_confirmation_preparation_contract(
    report: Mapping[str, Any],
    *,
    expected_output_root: str | None = None,
) -> dict[str, Any]:
    row = _object(report, "capacity confirmation preparation")
    selection = _object(row.get("selection"), "capacity confirmation selection")
    evaluation = _object(
        row.get("evaluation_contract"), "capacity confirmation evaluation"
    )
    decision = _object(
        row.get("decision_contract"), "capacity confirmation decision"
    )
    root = _absolute(selection.get("output_root"), "capacity confirmation root")
    screen_root = _absolute(selection.get("screen_output_root"), "capacity screen root")
    if root != _expected_confirmation_root(screen_root):
        raise ValueError("capacity confirmation output root differs")
    if expected_output_root is not None and root != expected_output_root:
        raise ValueError("capacity confirmation expected output root differs")
    if (
        row.get("schema_version") != PREPARATION_SCHEMA
        or row.get("role") != PREPARATION_ROLE
        or row.get("status") != "prepared"
        or row.get("scientific_status") != "capacity_confirmation_prepared"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("authorization_boundary") != PREPARATION_BOUNDARY
        or selection.get("dataset") != "imagenet_256"
        or selection.get("fresh_training_allowed") is not False
        or int(selection.get("frozen_training_step", -1)) != STOP_STEP
        or selection.get("frozen_arms") != list(ARM_NAMES)
        or evaluation.get("arms") != list(ARM_NAMES)
        or int(evaluation.get("checkpoint_step", -1)) != STOP_STEP
        or evaluation.get("frozen_checkpoint_training_allowed") is not False
        or int(evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or int(evaluation.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(evaluation.get("sampling_batch_size", -1)) != SAMPLE_BATCH_SIZE
        or evaluation.get("weights") != "ema"
        or evaluation.get("fixed_random_stream_across_arms") is not True
        or evaluation.get("fid_is_precision_recall_required") is not True
        or evaluation.get("class_fidelity_required") is not True
        or evaluation.get("screen_mechanism_and_rollout_replay_required") is not True
        or decision.get("absolute_support_and_class_gates_required") is not True
        or decision.get("confirmation_may_only_prepare_large_capacity_readiness")
        is not True
        or decision.get("large_capacity_readiness_requires_separate_authorization")
        is not True
        or decision.get("confirmation_cannot_authorize_full_300k") is not True
        or set(_object(row.get("frozen_arms"), "capacity frozen arms"))
        != set(ARM_NAMES)
    ):
        raise ValueError("capacity confirmation preparation contract differs")
    _git(row.get("preparation_git"), "capacity confirmation preparation")
    _git(row.get("screen_execution_checkout"), "capacity screen execution")
    sources = _object(
        row.get("source_evidence"), "capacity confirmation preparation sources"
    )
    _identity(sources.get("capacity_screen_result"), "capacity screen result")
    _identity(
        sources.get("capacity_screen_launch_receipt"),
        "capacity screen launch receipt",
    )
    arm_sources = _object(
        sources.get("capacity_screen_arm_validations"),
        "capacity screen arm validations",
    )
    if set(arm_sources) != set(ARM_NAMES):
        raise ValueError("capacity confirmation preparation arm source set differs")
    for arm in ARM_NAMES:
        _identity(arm_sources[arm], f"{arm} screen validation")
    return copy.deepcopy(row)


__all__ = [
    "CONFIRMATION_DIRNAME",
    "GUIDANCE_RESCALE",
    "GUIDANCE_SCALE",
    "PREPARATION_BOUNDARY",
    "PREPARATION_ROLE",
    "PREPARATION_SCHEMA",
    "SAMPLE_BATCH_SIZE",
    "SAMPLE_COUNT",
    "SAMPLE_SEED",
    "SAMPLE_STEPS",
    "build_capacity_confirmation_preparation",
    "validate_capacity_confirmation_preparation_contract",
]
