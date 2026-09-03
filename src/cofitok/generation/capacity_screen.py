"""Source-bound preparation for the fresh four-arm capacity screen.

The capacity screen is a bounded causal diagnostic.  It trains fresh base-128
and base-256 CoFiTok/dense pairs to the same intentional step-10K stop, then
evaluates all four arms under one matched 1K/DDIM-100 protocol.  A successful
screen may justify a separately authorized 10K-sample confirmation; it never
authorizes full 300K training directly.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation.exposure_capacity_decision import (
    DECISION_ROLE,
    DECISION_SCHEMA,
    validate_decision_contract,
)


PREPARATION_SCHEMA = "cofitok_generation_capacity_screen_preparation_v1"
PREPARATION_ROLE = "source_bound_fresh_four_arm_capacity_screen_preparation"
DECISION_VALIDATION_SCHEMA = (
    "cofitok_generation_exposure_capacity_decision_validation_v1"
)
DECISION_VALIDATION_ROLE = (
    "content_addressed_exposure_capacity_scientific_decision_validation"
)

DATASET = "imagenet_256"
CONFIGURED_STEPS = 100_000
STOP_STEP = 10_000
EFFECTIVE_BATCH = 64
TRAINING_SEED = 2027
SAMPLE_COUNT = 1_000
SAMPLE_STEPS = 100
SAMPLE_SEED = 0
SAMPLE_BATCH_SIZE = 4
GUIDANCE_SCALE = 1.5
GUIDANCE_RESCALE = 0.0
MECHANISM_IMAGES = 256
MECHANISM_TIMESTEP = 500
MECHANISM_RANDOM_ORDERS = 4
ROLLOUT_IMAGES = 64
ROLLOUT_BATCH_SIZE = 2
ROLLOUT_SEED = 2029

ARM_SPECS: dict[str, dict[str, Any]] = {
    "base128_cofitok": {
        "capacity": "base128",
        "method": "cofitok",
        "base_channels": 128,
        "parameter_count": 62_834_083,
        "recipe_stage": "stability_capacity_reference",
        "prefix_budget": 8,
        "mechanism_required": True,
        "rollout_required": True,
    },
    "base128_dense_identity": {
        "capacity": "base128",
        "method": "dense_identity",
        "base_channels": 128,
        "parameter_count": 62_824_707,
        "recipe_stage": "stability_capacity_reference",
        "prefix_budget": 1,
        "mechanism_required": True,
        "rollout_required": True,
    },
    "base256_cofitok": {
        "capacity": "base256",
        "method": "cofitok",
        "base_channels": 256,
        "parameter_count": 250_153_763,
        "recipe_stage": "stability_capacity_qualification",
        "prefix_budget": 8,
        "mechanism_required": True,
        "rollout_required": True,
    },
    "base256_dense_identity": {
        "capacity": "base256",
        "method": "dense_identity",
        "base_channels": 256,
        "parameter_count": 250_135_043,
        "recipe_stage": "stability_capacity_qualification",
        "prefix_budget": 1,
        "mechanism_required": True,
        "rollout_required": True,
    },
}
ARM_NAMES = tuple(ARM_SPECS)
CAPACITY_NAMES = ("base128", "base256")
METHOD_NAMES = ("cofitok", "dense_identity")

ALLOWED_WITHIN_METHOD_CAPACITY_DIFFERENCES = {
    "name",
    "model.base_channels",
}

PREPARATION_BOUNDARY = {
    "capacity_screen_prepared": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "capacity_confirmation_allowed": False,
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


def _hex(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length or value != value.lower():
        return False
    return all(character in "0123456789abcdef" for character in value)


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
        or not _hex(row.get("sha256"), length=64)
    ):
        raise ValueError(f"{name} identity is malformed")
    return {
        "path": row["path"],
        "bytes": row["bytes"],
        "sha256": row["sha256"],
    }


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    if (
        not _hex(row.get("revision"), length=40)
        or not _hex(row.get("tree"), length=40)
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


def _absolute(path: str, name: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} is missing")
    if not (Path(path).is_absolute() or PurePosixPath(path).is_absolute()):
        raise ValueError(f"{name} must be absolute")
    return path


def _differences(left: Any, right: Any, path: str = "") -> set[str]:
    if type(left) is not type(right):
        return {path}
    if isinstance(left, Mapping):
        result: set[str] = set()
        for key in set(left) | set(right):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                result.add(child)
            else:
                result.update(_differences(left[key], right[key], child))
        return result
    if isinstance(left, list):
        return set() if left == right else {path}
    return set() if left == right else {path}


def _validate_pair_report(
    report: Mapping[str, Any],
    *,
    capacity: str,
) -> dict[str, Any]:
    if capacity not in CAPACITY_NAMES:
        raise ValueError("capacity pair name differs")
    expected_stage = ARM_SPECS[f"{capacity}_cofitok"]["recipe_stage"]
    expected_counts = {
        method: ARM_SPECS[f"{capacity}_{method}"]["parameter_count"]
        for method in METHOD_NAMES
    }
    recipe = _object(report.get("training_recipe"), f"{capacity} recipe")
    cofitok = _object(report.get("cofitok"), f"{capacity} CoFiTok config")
    dense = _object(report.get("dense"), f"{capacity} dense config")
    if (
        report.get("status") != "pass"
        or report.get("mismatches") != []
        or recipe.get("valid") is not True
        or recipe.get("stage") != expected_stage
        or int(cofitok.get("parameter_count", -1)) != expected_counts["cofitok"]
        or int(dense.get("parameter_count", -1))
        != expected_counts["dense_identity"]
        or abs(float(report.get("relative_parameter_gap", 1.0))) > 0.02
    ):
        raise ValueError(f"{capacity} matched config validation differs")
    effective = _object(recipe.get("effective_batches"), f"{capacity} effective batches")
    for method in METHOD_NAMES:
        row = _object(effective.get(method), f"{capacity} {method} batch")
        if int(row.get("effective_batch_size", -1)) != EFFECTIVE_BATCH:
            raise ValueError(f"{capacity} {method} effective batch differs")
    return {
        "status": "pass",
        "recipe_stage": expected_stage,
        "parameter_counts": expected_counts,
        "relative_parameter_gap": float(report["relative_parameter_gap"]),
        "effective_batches": copy.deepcopy(effective),
        "recipe_schema": recipe.get("schema"),
    }


def _validate_decision_sources(
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    decision_validation: Mapping[str, Any],
    decision_validation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_decision_contract(decision)
    if (
        decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("role") != DECISION_ROLE
        or validated.get("decision") != "capacity_screen"
        or validated.get("next_stage", {}).get(
            "capacity_screen_preparation_allowed"
        )
        is not True
    ):
        raise ValueError("capacity screen requires the capacity-screen route")
    normalized_decision = _identity(decision_identity, "capacity decision")
    normalized_validation = _identity(
        decision_validation_identity,
        "capacity decision validation",
    )
    if (
        decision_validation.get("schema_version") != DECISION_VALIDATION_SCHEMA
        or decision_validation.get("role") != DECISION_VALIDATION_ROLE
        or decision_validation.get("status") != "pass"
        or decision_validation.get("scientific_route") != "capacity_screen"
        or decision_validation.get("decision") != normalized_decision
        or decision_validation.get("generation_advantage_proven") is not False
    ):
        raise ValueError("capacity decision validation differs")
    return {
        "decision": normalized_decision,
        "decision_validation": normalized_validation,
        "decision_git": copy.deepcopy(validated["decision_git"]),
    }


def build_capacity_screen_preparation(
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    decision_validation: Mapping[str, Any],
    decision_validation_identity: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    config_identities: Mapping[str, Mapping[str, Any]],
    pair_validations: Mapping[str, Mapping[str, Any]],
    pair_validation_identities: Mapping[str, Mapping[str, Any]],
    preparation_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    source = _validate_decision_sources(
        decision=decision,
        decision_identity=decision_identity,
        decision_validation=decision_validation,
        decision_validation_identity=decision_validation_identity,
    )
    if set(configs) != set(ARM_NAMES) or set(config_identities) != set(ARM_NAMES):
        raise ValueError("capacity screen config arm set differs")
    if set(pair_validations) != set(CAPACITY_NAMES) or set(
        pair_validation_identities
    ) != set(CAPACITY_NAMES):
        raise ValueError("capacity screen pair-validation set differs")
    root = _absolute(output_root, "capacity screen output root")
    normalized_git = _git(preparation_git, "capacity screen preparation")
    validated_pairs = {
        capacity: _validate_pair_report(pair_validations[capacity], capacity=capacity)
        for capacity in CAPACITY_NAMES
    }
    normalized_pair_ids = {
        capacity: _identity(
            pair_validation_identities[capacity],
            f"{capacity} pair validation",
        )
        for capacity in CAPACITY_NAMES
    }

    normalized_config_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        spec = ARM_SPECS[arm]
        config = _object(configs[arm], f"{arm} config")
        data = _object(config.get("data"), f"{arm} data")
        model = _object(config.get("model"), f"{arm} model")
        runtime = _object(config.get("runtime"), f"{arm} runtime")
        optimization = _object(config.get("optimization"), f"{arm} optimization")
        if (
            data.get("dataset") != DATASET
            or bool(data.get("class_conditional")) is not True
            or int(model.get("base_channels", -1)) != spec["base_channels"]
            or int(runtime.get("steps", -1)) != CONFIGURED_STEPS
            or runtime.get("protected_checkpoint_steps") != [STOP_STEP]
            or int(runtime.get("seed", -1)) != TRAINING_SEED
            or int(data.get("batch_size", 0))
            * int(optimization.get("gradient_accumulation_steps", 0))
            != EFFECTIVE_BATCH
        ):
            raise ValueError(f"{arm} training selection differs")
        normalized_config_ids[arm] = _identity(
            config_identities[arm], f"{arm} config"
        )

    differences: dict[str, list[str]] = {}
    for method in METHOD_NAMES:
        base = configs[f"base128_{method}"]
        large = configs[f"base256_{method}"]
        observed = _differences(base, large)
        if observed != ALLOWED_WITHIN_METHOD_CAPACITY_DIFFERENCES:
            raise ValueError(
                f"{method} capacity intervention differs: {sorted(observed)}"
            )
        differences[method] = sorted(observed)

    evaluation_contract = {
        "arms": list(ARM_NAMES),
        "checkpoint_step": STOP_STEP,
        "weights": "ema",
        "sampler": "ddim",
        "sample_steps": SAMPLE_STEPS,
        "samples_per_arm": SAMPLE_COUNT,
        "sampling_batch_size": SAMPLE_BATCH_SIZE,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "precision": "bf16",
        "seed": SAMPLE_SEED,
        "class_schedule": "balanced_modulo",
        "fixed_random_stream_across_arms": True,
        "fid_is_precision_recall_required": True,
        "class_fidelity_required": True,
        "checkpoint_evaluation_required_for_all_arms": True,
        "checkpoint_evaluation_images_per_arm": MECHANISM_IMAGES,
        "checkpoint_evaluation_timestep": MECHANISM_TIMESTEP,
        "cofitok_random_orders": MECHANISM_RANDOM_ORDERS,
        "dense_random_orders": 0,
        "rollout_required_for_all_arms": True,
        "rollout_images_per_arm": ROLLOUT_IMAGES,
        "rollout_batch_size": ROLLOUT_BATCH_SIZE,
        "rollout_seed": ROLLOUT_SEED,
        "primary_estimand": (
            "difference_in_differences_base256_minus_base128_"
            "by_cofitok_minus_dense"
        ),
    }
    result = {
        "schema_version": PREPARATION_SCHEMA,
        "role": PREPARATION_ROLE,
        "status": "prepared",
        "scientific_status": "capacity_screen_prepared",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation_git": normalized_git,
        "source_evidence": source,
        "selection": {
            "dataset": DATASET,
            "output_root": root,
            "configured_training_horizon": CONFIGURED_STEPS,
            "stop_after_step": STOP_STEP,
            "effective_batch_size": EFFECTIVE_BATCH,
            "images_seen_per_arm": STOP_STEP * EFFECTIVE_BATCH,
            "fresh_initialization_required_for_all_arms": True,
            "fresh_training_arms": list(ARM_NAMES),
            "capacity_intervention": ["model.base_channels"],
            "base_channels": {"base128": 128, "base256": 256},
            "parameter_counts": {
                arm: ARM_SPECS[arm]["parameter_count"] for arm in ARM_NAMES
            },
        },
        "configs": normalized_config_ids,
        "pair_validations": normalized_pair_ids,
        "validated_pairs": validated_pairs,
        "within_method_capacity_differences": differences,
        "evaluation_contract": evaluation_contract,
        "decision_contract": {
            "screen_may_only_prepare_confirmation": True,
            "confirmation_requires_separate_exact_authorization": True,
            "confirmation_samples_per_arm": 10_000,
            "screen_cannot_authorize_full_300k": True,
            "full_300k_requires_later_confirmatory_noncollapse_gate": True,
        },
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }
    validate_capacity_screen_preparation_contract(result, expected_output_root=root)
    return result


def validate_capacity_screen_preparation_contract(
    report: Mapping[str, Any],
    *,
    expected_output_root: str | None = None,
) -> dict[str, Any]:
    row = _object(report, "capacity screen preparation")
    selection = _object(row.get("selection"), "capacity screen selection")
    evaluation = _object(
        row.get("evaluation_contract"), "capacity screen evaluation contract"
    )
    decision = _object(
        row.get("decision_contract"), "capacity screen decision contract"
    )
    root = _absolute(str(selection.get("output_root", "")), "capacity screen root")
    if expected_output_root is not None and root != expected_output_root:
        raise ValueError("capacity screen output root differs")
    if (
        row.get("schema_version") != PREPARATION_SCHEMA
        or row.get("role") != PREPARATION_ROLE
        or row.get("status") != "prepared"
        or row.get("scientific_status") != "capacity_screen_prepared"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("authorization_boundary") != PREPARATION_BOUNDARY
        or selection.get("dataset") != DATASET
        or int(selection.get("configured_training_horizon", -1))
        != CONFIGURED_STEPS
        or int(selection.get("stop_after_step", -1)) != STOP_STEP
        or int(selection.get("effective_batch_size", -1)) != EFFECTIVE_BATCH
        or selection.get("fresh_training_arms") != list(ARM_NAMES)
        or selection.get("fresh_initialization_required_for_all_arms") is not True
        or evaluation.get("arms") != list(ARM_NAMES)
        or int(evaluation.get("checkpoint_step", -1)) != STOP_STEP
        or int(evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or int(evaluation.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(evaluation.get("sampling_batch_size", -1)) != SAMPLE_BATCH_SIZE
        or evaluation.get("weights") != "ema"
        or evaluation.get("class_fidelity_required") is not True
        or evaluation.get("fid_is_precision_recall_required") is not True
        or evaluation.get("fixed_random_stream_across_arms") is not True
        or evaluation.get("checkpoint_evaluation_required_for_all_arms")
        is not True
        or int(evaluation.get("checkpoint_evaluation_images_per_arm", -1))
        != MECHANISM_IMAGES
        or int(evaluation.get("checkpoint_evaluation_timestep", -1))
        != MECHANISM_TIMESTEP
        or evaluation.get("rollout_required_for_all_arms") is not True
        or int(evaluation.get("rollout_images_per_arm", -1)) != ROLLOUT_IMAGES
        or int(evaluation.get("rollout_batch_size", -1)) != ROLLOUT_BATCH_SIZE
        or int(evaluation.get("rollout_seed", -1)) != ROLLOUT_SEED
        or decision.get("screen_may_only_prepare_confirmation") is not True
        or decision.get("confirmation_requires_separate_exact_authorization")
        is not True
        or int(decision.get("confirmation_samples_per_arm", -1)) != 10_000
        or decision.get("screen_cannot_authorize_full_300k") is not True
    ):
        raise ValueError("capacity screen preparation contract differs")
    if set(row.get("configs", {})) != set(ARM_NAMES):
        raise ValueError("capacity screen preparation config set differs")
    if set(row.get("pair_validations", {})) != set(CAPACITY_NAMES):
        raise ValueError("capacity screen pair-validation set differs")
    _git(row.get("preparation_git"), "capacity screen preparation")
    return copy.deepcopy(row)


__all__ = [
    "ALLOWED_WITHIN_METHOD_CAPACITY_DIFFERENCES",
    "ARM_NAMES",
    "ARM_SPECS",
    "CAPACITY_NAMES",
    "CONFIGURED_STEPS",
    "DATASET",
    "EFFECTIVE_BATCH",
    "GUIDANCE_RESCALE",
    "GUIDANCE_SCALE",
    "MECHANISM_IMAGES",
    "MECHANISM_RANDOM_ORDERS",
    "MECHANISM_TIMESTEP",
    "METHOD_NAMES",
    "PREPARATION_BOUNDARY",
    "PREPARATION_ROLE",
    "PREPARATION_SCHEMA",
    "ROLLOUT_IMAGES",
    "ROLLOUT_BATCH_SIZE",
    "ROLLOUT_SEED",
    "SAMPLE_BATCH_SIZE",
    "SAMPLE_COUNT",
    "SAMPLE_SEED",
    "SAMPLE_STEPS",
    "STOP_STEP",
    "TRAINING_SEED",
    "build_capacity_screen_preparation",
    "validate_capacity_screen_preparation_contract",
]
