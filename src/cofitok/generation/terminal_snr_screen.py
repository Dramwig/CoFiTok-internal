"""Source-bound preparation for the matched terminal-SNR endpoint screen."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.configs import config_from_dict, config_to_dict
from cofitok.generation.terminal_snr_reassessment import (
    CONTROL_ENDPOINT_FRACTION,
    SCREEN_THRESHOLDS,
    SELECTED_ENDPOINT_FRACTION,
    SELECTED_INTERVENTION,
    validate_terminal_snr_reassessment,
    validate_terminal_snr_reassessment_validation,
)


PREPARATION_SCHEMA = "cofitok_generation_terminal_snr_screen_preparation_v1"
PREPARATION_ROLE = "source_bound_fresh_four_arm_terminal_snr_screen_preparation"

DATASET = "imagenet_256"
CONFIGURED_STEPS = 100_000
STOP_STEP = 10_000
EFFECTIVE_BATCH = 64
TRAINING_SEED = 2027
SAMPLE_COUNT = 1_000
SAMPLE_STEPS = 100
SAMPLE_SEED = 2027
SAMPLE_BATCH_SIZE = 4
GUIDANCE_SCALE = 1.5
GUIDANCE_RESCALE = 0.0
MECHANISM_IMAGES = 256
MECHANISM_TIMESTEP = 500
MECHANISM_RANDOM_ORDERS = 4
ROLLOUT_IMAGES = 64
ROLLOUT_BATCH_SIZE = 2
ROLLOUT_SEED = 2029

CONDITION_NAMES = ("control", "endpoint0975")
METHOD_NAMES = ("cofitok", "dense_identity")
CONDITION_ENDPOINTS = {
    "control": CONTROL_ENDPOINT_FRACTION,
    "endpoint0975": SELECTED_ENDPOINT_FRACTION,
}
ARM_SPECS: dict[str, dict[str, Any]] = {
    "control_cofitok": {
        "condition": "control",
        "method": "cofitok",
        "endpoint_fraction": CONTROL_ENDPOINT_FRACTION,
        "base_channels": 128,
        "parameter_count": 62_834_083,
        "recipe_stage": "stability_capacity_reference",
        "prefix_budget": 8,
    },
    "control_dense_identity": {
        "condition": "control",
        "method": "dense_identity",
        "endpoint_fraction": CONTROL_ENDPOINT_FRACTION,
        "base_channels": 128,
        "parameter_count": 62_824_707,
        "recipe_stage": "stability_capacity_reference",
        "prefix_budget": 1,
    },
    "endpoint0975_cofitok": {
        "condition": "endpoint0975",
        "method": "cofitok",
        "endpoint_fraction": SELECTED_ENDPOINT_FRACTION,
        "base_channels": 128,
        "parameter_count": 62_834_083,
        "recipe_stage": "stability_capacity_reference",
        "prefix_budget": 8,
    },
    "endpoint0975_dense_identity": {
        "condition": "endpoint0975",
        "method": "dense_identity",
        "endpoint_fraction": SELECTED_ENDPOINT_FRACTION,
        "base_channels": 128,
        "parameter_count": 62_824_707,
        "recipe_stage": "stability_capacity_reference",
        "prefix_budget": 1,
    },
}
ARM_NAMES = tuple(ARM_SPECS)
ALLOWED_WITHIN_METHOD_DIFFERENCES = {
    "diffusion.cosine_endpoint_fraction",
    "name",
}

PREPARATION_BOUNDARY = {
    "terminal_snr_screen_prepared": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "confirmation_preparation_allowed": False,
    "confirmation_launch_allowed": False,
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
        or not _hex(row.get("sha256"), length=64)
    ):
        raise ValueError(f"{name} identity is malformed")
    return {key: row[key] for key in ("path", "bytes", "sha256")}


def _content_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("bytes", "sha256")}


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


def _resolved_config(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        return config_to_dict(config_from_dict(dict(value)))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a valid experiment config") from error


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
    report: Mapping[str, Any], *, condition: str
) -> dict[str, Any]:
    if condition not in CONDITION_NAMES:
        raise ValueError("terminal-SNR condition name differs")
    expected_counts = {
        method: ARM_SPECS[f"{condition}_{method}"]["parameter_count"]
        for method in METHOD_NAMES
    }
    recipe = _object(report.get("training_recipe"), f"{condition} recipe")
    cofitok = _object(report.get("cofitok"), f"{condition} CoFiTok config")
    dense = _object(report.get("dense"), f"{condition} dense config")
    matched_diffusion = _object(
        report.get("matched_diffusion"), f"{condition} matched diffusion"
    )
    if (
        report.get("status") != "pass"
        or report.get("mismatches") != []
        or recipe.get("valid") is not True
        or recipe.get("stage") != "stability_capacity_reference"
        or int(cofitok.get("parameter_count", -1)) != expected_counts["cofitok"]
        or int(dense.get("parameter_count", -1))
        != expected_counts["dense_identity"]
        or abs(float(report.get("relative_parameter_gap", 1.0))) > 0.02
        or matched_diffusion.get("schedule_type") != "cosine"
        or matched_diffusion.get("prediction_target") != "epsilon"
        or matched_diffusion.get("cosine_endpoint_fraction")
        != CONDITION_ENDPOINTS[condition]
    ):
        raise ValueError(f"{condition} matched config validation differs")
    effective = _object(recipe.get("effective_batches"), f"{condition} batches")
    for method in METHOD_NAMES:
        row = _object(effective.get(method), f"{condition} {method} batch")
        if int(row.get("effective_batch_size", -1)) != EFFECTIVE_BATCH:
            raise ValueError(f"{condition} {method} effective batch differs")
    return {
        "status": "pass",
        "recipe_stage": "stability_capacity_reference",
        "endpoint_fraction": CONDITION_ENDPOINTS[condition],
        "parameter_counts": expected_counts,
        "relative_parameter_gap": float(report["relative_parameter_gap"]),
        "effective_batches": copy.deepcopy(effective),
        "recipe_schema": recipe.get("schema"),
    }


def _validate_reassessment_sources(
    *,
    reassessment: Mapping[str, Any],
    reassessment_identity: Mapping[str, Any],
    reassessment_validation: Mapping[str, Any],
    reassessment_validation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    reassessment_id = _identity(reassessment_identity, "terminal-SNR reassessment")
    validation_id = _identity(
        reassessment_validation_identity, "terminal-SNR reassessment validation"
    )
    decision = validate_terminal_snr_reassessment(reassessment)
    validation = validate_terminal_snr_reassessment_validation(
        reassessment_validation,
        decision=decision,
        decision_identity=reassessment_id,
    )
    next_stage = _object(decision.get("next_stage"), "terminal-SNR next stage")
    if (
        decision.get("status") != "completed"
        or decision.get("scientific_status")
        != "bounded_screen_selected_not_executed"
        or decision.get("selected_intervention", {}).get("id")
        != SELECTED_INTERVENTION
        or next_stage.get("route")
        != "build_separate_source_bound_terminal_snr_screen_preparation"
        or next_stage.get("source_bound_preparation_may_be_built") is not True
        or next_stage.get("execution_ready") is not False
        or validation.get("status") != "pass"
        or validation.get("decision") != reassessment_id
    ):
        raise ValueError("terminal-SNR reassessment does not allow preparation")
    return {
        "reassessment": reassessment_id,
        "reassessment_validation": validation_id,
        "reassessment_git": copy.deepcopy(decision["decision_git"]),
        "selected_intervention": copy.deepcopy(decision["selected_intervention"]),
        "bounded_screen_contract": copy.deepcopy(decision["bounded_screen_contract"]),
        "config_contract": copy.deepcopy(
            _object(decision["source_evidence"], "terminal-SNR sources")[
                "config_contract"
            ]
        ),
    }


def build_terminal_snr_screen_preparation(
    *,
    reassessment: Mapping[str, Any],
    reassessment_identity: Mapping[str, Any],
    reassessment_validation: Mapping[str, Any],
    reassessment_validation_identity: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    config_identities: Mapping[str, Mapping[str, Any]],
    pair_validations: Mapping[str, Mapping[str, Any]],
    pair_validation_identities: Mapping[str, Mapping[str, Any]],
    preparation_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    source = _validate_reassessment_sources(
        reassessment=reassessment,
        reassessment_identity=reassessment_identity,
        reassessment_validation=reassessment_validation,
        reassessment_validation_identity=reassessment_validation_identity,
    )
    if set(configs) != set(ARM_NAMES) or set(config_identities) != set(ARM_NAMES):
        raise ValueError("terminal-SNR screen config arm set differs")
    if set(pair_validations) != set(CONDITION_NAMES) or set(
        pair_validation_identities
    ) != set(CONDITION_NAMES):
        raise ValueError("terminal-SNR screen pair-validation set differs")
    root = _absolute(output_root, "terminal-SNR screen output root")
    normalized_git = _git(preparation_git, "terminal-SNR screen preparation")
    validated_pairs = {
        condition: _validate_pair_report(
            pair_validations[condition], condition=condition
        )
        for condition in CONDITION_NAMES
    }
    pair_ids = {
        condition: _identity(
            pair_validation_identities[condition],
            f"{condition} pair validation",
        )
        for condition in CONDITION_NAMES
    }

    source_contract = _object(source["config_contract"], "source config contract")
    source_config_groups = {
        "control": _object(
            source_contract.get("control_configs"), "source control configs"
        ),
        "endpoint0975": _object(
            source_contract.get("intervention_configs"),
            "source intervention configs",
        ),
    }
    normalized_configs: dict[str, dict[str, Any]] = {}
    resolved: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        spec = ARM_SPECS[arm]
        config = _resolved_config(configs[arm], f"{arm} config")
        resolved[arm] = config
        data = _object(config.get("data"), f"{arm} data")
        diffusion = _object(config.get("diffusion"), f"{arm} diffusion")
        model = _object(config.get("model"), f"{arm} model")
        runtime = _object(config.get("runtime"), f"{arm} runtime")
        optimization = _object(config.get("optimization"), f"{arm} optimization")
        if (
            data.get("dataset") != DATASET
            or data.get("class_conditional") is not True
            or diffusion.get("schedule_type") != "cosine"
            or diffusion.get("prediction_target") != "epsilon"
            or diffusion.get("cosine_endpoint_fraction")
            != spec["endpoint_fraction"]
            or int(model.get("base_channels", -1)) != 128
            or int(runtime.get("steps", -1)) != CONFIGURED_STEPS
            or runtime.get("protected_checkpoint_steps") != [STOP_STEP]
            or int(runtime.get("seed", -1)) != TRAINING_SEED
            or int(data.get("batch_size", 0))
            * int(optimization.get("gradient_accumulation_steps", 0))
            != EFFECTIVE_BATCH
        ):
            raise ValueError(f"{arm} training selection differs")
        descriptor = _identity(config_identities[arm], f"{arm} config")
        source_descriptor = _identity(
            source_config_groups[spec["condition"]][spec["method"]],
            f"source {arm} config",
        )
        if _content_identity(descriptor) != _content_identity(source_descriptor):
            raise ValueError(f"{arm} config differs from the reassessment")
        normalized_configs[arm] = descriptor

    differences: dict[str, list[str]] = {}
    for method in METHOD_NAMES:
        control = resolved[f"control_{method}"]
        intervention = resolved[f"endpoint0975_{method}"]
        observed = _differences(control, intervention)
        if observed != ALLOWED_WITHIN_METHOD_DIFFERENCES:
            raise ValueError(
                f"{method} endpoint intervention differs: {sorted(observed)}"
            )
        differences[method] = sorted(observed)

    source_screen = _object(
        source["bounded_screen_contract"], "source bounded-screen contract"
    )
    if (
        source_screen.get("thresholds") != SCREEN_THRESHOLDS
        or source_screen.get("training_steps_per_arm") != STOP_STEP
        or source_screen.get("evaluation_samples_per_arm") != SAMPLE_COUNT
        or source_screen.get("sample_steps") != SAMPLE_STEPS
        or source_screen.get("sampler") != "ddim"
        or source_screen.get("weights") != "ema"
        or source_screen.get("guidance_scale") != GUIDANCE_SCALE
        or source_screen.get("precision") != "bf16"
        or source_screen.get("seed") != TRAINING_SEED
    ):
        raise ValueError("terminal-SNR bounded-screen contract differs")

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
        "terminal_raw_x0_clip_fraction_source": "free_sampling_rollout.steps[0]",
        "primary_estimand": "within_method_endpoint0975_minus_control",
    }
    report = {
        "schema_version": PREPARATION_SCHEMA,
        "role": PREPARATION_ROLE,
        "status": "prepared",
        "scientific_status": "terminal_snr_screen_prepared",
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
            "single_scientific_config_field": (
                "diffusion.cosine_endpoint_fraction"
            ),
            "conditions": {
                condition: {"cosine_endpoint_fraction": endpoint}
                for condition, endpoint in CONDITION_ENDPOINTS.items()
            },
            "parameter_counts": {
                arm: ARM_SPECS[arm]["parameter_count"] for arm in ARM_NAMES
            },
        },
        "configs": normalized_configs,
        "pair_validations": pair_ids,
        "validated_pairs": validated_pairs,
        "within_method_condition_differences": differences,
        "evaluation_contract": evaluation_contract,
        "thresholds": copy.deepcopy(SCREEN_THRESHOLDS),
        "decision_contract": {
            "screen_may_only_prepare_frozen_confirmation": True,
            "confirmation_requires_separate_exact_authorization": True,
            "confirmation_samples_per_arm": 10_000,
            "screen_cannot_authorize_full_300k": True,
            "full_300k_requires_passing_frozen_confirmation": True,
        },
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }
    return validate_terminal_snr_screen_preparation_contract(
        report, expected_output_root=root
    )


def validate_terminal_snr_screen_preparation_contract(
    report: Mapping[str, Any], *, expected_output_root: str | None = None
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR screen preparation")
    selection = _object(row.get("selection"), "terminal-SNR selection")
    evaluation = _object(row.get("evaluation_contract"), "terminal-SNR evaluation")
    decision = _object(row.get("decision_contract"), "terminal-SNR decision")
    root = _absolute(str(selection.get("output_root", "")), "terminal-SNR root")
    if expected_output_root is not None and root != expected_output_root:
        raise ValueError("terminal-SNR screen output root differs")
    if (
        row.get("schema_version") != PREPARATION_SCHEMA
        or row.get("role") != PREPARATION_ROLE
        or row.get("status") != "prepared"
        or row.get("scientific_status") != "terminal_snr_screen_prepared"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("authorization_boundary") != PREPARATION_BOUNDARY
        or row.get("thresholds") != SCREEN_THRESHOLDS
        or selection.get("dataset") != DATASET
        or selection.get("single_scientific_config_field")
        != "diffusion.cosine_endpoint_fraction"
        or selection.get("conditions")
        != {
            condition: {"cosine_endpoint_fraction": endpoint}
            for condition, endpoint in CONDITION_ENDPOINTS.items()
        }
        or int(selection.get("configured_training_horizon", -1))
        != CONFIGURED_STEPS
        or int(selection.get("stop_after_step", -1)) != STOP_STEP
        or int(selection.get("effective_batch_size", -1)) != EFFECTIVE_BATCH
        or int(selection.get("images_seen_per_arm", -1))
        != STOP_STEP * EFFECTIVE_BATCH
        or selection.get("fresh_training_arms") != list(ARM_NAMES)
        or selection.get("fresh_initialization_required_for_all_arms") is not True
        or evaluation.get("arms") != list(ARM_NAMES)
        or int(evaluation.get("checkpoint_step", -1)) != STOP_STEP
        or int(evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or int(evaluation.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(evaluation.get("sampling_batch_size", -1)) != SAMPLE_BATCH_SIZE
        or int(evaluation.get("seed", -1)) != SAMPLE_SEED
        or evaluation.get("weights") != "ema"
        or evaluation.get("class_fidelity_required") is not True
        or evaluation.get("fid_is_precision_recall_required") is not True
        or evaluation.get("fixed_random_stream_across_arms") is not True
        or evaluation.get("checkpoint_evaluation_required_for_all_arms")
        is not True
        or evaluation.get("rollout_required_for_all_arms") is not True
        or decision.get("screen_may_only_prepare_frozen_confirmation") is not True
        or decision.get("confirmation_requires_separate_exact_authorization")
        is not True
        or int(decision.get("confirmation_samples_per_arm", -1)) != 10_000
        or decision.get("screen_cannot_authorize_full_300k") is not True
        or decision.get("full_300k_requires_passing_frozen_confirmation")
        is not True
    ):
        raise ValueError("terminal-SNR screen preparation contract differs")
    if set(row.get("configs", {})) != set(ARM_NAMES):
        raise ValueError("terminal-SNR preparation config set differs")
    if set(row.get("pair_validations", {})) != set(CONDITION_NAMES):
        raise ValueError("terminal-SNR pair-validation set differs")
    _git(row.get("preparation_git"), "terminal-SNR screen preparation")
    return copy.deepcopy(row)


__all__ = [
    "ALLOWED_WITHIN_METHOD_DIFFERENCES",
    "ARM_NAMES",
    "ARM_SPECS",
    "CONDITION_ENDPOINTS",
    "CONDITION_NAMES",
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
    "ROLLOUT_BATCH_SIZE",
    "ROLLOUT_IMAGES",
    "ROLLOUT_SEED",
    "SAMPLE_BATCH_SIZE",
    "SAMPLE_COUNT",
    "SAMPLE_SEED",
    "SAMPLE_STEPS",
    "STOP_STEP",
    "TRAINING_SEED",
    "build_terminal_snr_screen_preparation",
    "validate_terminal_snr_screen_preparation_contract",
]
