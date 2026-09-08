"""Non-authorizing preparation for fresh endpoint-0.975 large-capacity training.

This artifact is intentionally narrower than the legacy capacity-scaling path.
It consumes the physically replayable terminal-SNR frozen confirmation and
selects a *freshly initialized* matched base-256 / 300K pair.  It does not
authorize a checkout mutation, GPU use, training, sampling, evaluation, or
release; those actions require a later source-bound evidence and authorization
chain.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

import torch

from cofitok.configs import config_from_dict, config_to_dict
from cofitok.generation.terminal_snr_confirmation import SAMPLE_COUNT
from cofitok.generation.terminal_snr_confirmation_result import (
    replay_terminal_snr_confirmation_result,
    validate_terminal_snr_confirmation_result_contract,
    validate_terminal_snr_confirmation_validation_receipt,
)
from cofitok.generation_pair import generation_pair_contract
from cofitok.generation_recipe import (
    GENERATION_TRAINING_RECIPE_SCHEMA,
    generation_training_recipe_contract,
)
from cofitok.models import CoFiTokTiny


PREPARATION_SCHEMA = (
    "cofitok_generation_terminal_snr_large_capacity_preparation_v1"
)
PREPARATION_ROLE = (
    "source_bound_fresh_terminal_snr_endpoint0975_large_capacity_preparation"
)
FULL_STAGE_DIRNAME = "terminal_snr_endpoint0975_large_capacity_300k_v1"

METHODS = ("cofitok", "dense_identity")
CONFIG_NAMES = {
    "cofitok": (
        "imagenet256_terminal_snr_endpoint0975_rgbtail3_rollout_x0_u2_"
        "ema_teacher_k8_300k"
    ),
    "dense_identity": (
        "imagenet256_terminal_snr_endpoint0975_rollout_x0_u2_"
        "ema_teacher_dense_300k"
    ),
}
CONFIG_FILENAMES = {
    method: f"{name}.json" for method, name in CONFIG_NAMES.items()
}
EXPECTED_PARAMETER_COUNTS = {
    "cofitok": 250_153_763,
    "dense_identity": 250_135_043,
}
BASE_CHANNELS = 256
EFFECTIVE_BATCH_SIZE = 64
TARGET_STEPS = 300_000
MILESTONE_STEPS = (50_000, 100_000, 200_000, 300_000)
ENDPOINT_FRACTION = 0.975

TREND_EVALUATION_CONTRACT = {
    "weights": "ema",
    "sampler": "ddim",
    "samples_per_method": 2_048,
    "sample_steps": 50,
    "batch_size": 32,
    "guidance_scale": 1.5,
    "guidance_rescale": 0.0,
    "cfg_batch_mode": "batched",
    "eta": 0.0,
    "clip_x0": True,
    "precision": "bf16",
    "seed": 0,
    "start_index": 0,
    "class_schedule": "balanced_modulo",
    "fixed_random_stream_across_methods": True,
    "fid_is_required": True,
    "checkpoint_mechanism_diagnostics_required": True,
    "rollout_stability_required": True,
    "formal_gate_substitute": False,
}

FORMAL_EVALUATION_CONTRACT = {
    "methods": list(METHODS),
    "checkpoint_step": TARGET_STEPS,
    "weights": "ema",
    "sampler": "ddim",
    "samples_per_method": 50_000,
    "sample_steps": 250,
    "guidance_scale": 1.5,
    "guidance_rescale": 0.0,
    "cfg_batch_mode": "batched",
    "eta": 0.0,
    "clip_x0": True,
    "precision": "bf16",
    "seed": 0,
    "start_index": 0,
    "class_schedule": "balanced_modulo",
    "requested_samples_per_class": 50,
    "fixed_random_stream_across_methods": True,
    "per_method_real_forward_preflight_required": True,
    "sampling_progress_required": True,
    "fid_is_precision_recall_required": True,
    "class_fidelity_required": True,
    "checkpoint_mechanism_diagnostics_required": True,
    "rollout_stability_required": True,
    "final_release_gate_required": True,
}

READINESS_REQUIREMENTS = [
    "exact_clean_execution_checkout_identity",
    "matched_pair_validation_receipt",
    "gpu_runtime_selection_with_real_forward_benchmarks",
    "full_stage_storage_capacity_preflight",
    "gpu_idle_process_ownership_and_output_absence_live_snapshot",
    "user_stage_authorization_bound_to_the_explicit_300k_goal",
    "separate_source_bound_execution_authorization",
    "immutable_launch_receipt",
]

PREPARATION_BOUNDARY = {
    "large_capacity_readiness_prepared": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "checkpoint_mutation_allowed": False,
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
        or not PurePosixPath(row["path"]).is_absolute()
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
    return copy.deepcopy(row)


def _absolute_posix(value: Any, name: str) -> str:
    if not isinstance(value, str) or not PurePosixPath(value).is_absolute():
        raise ValueError(f"{name} must be an absolute POSIX path")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolved_config(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        return config_to_dict(config_from_dict(dict(value)))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a valid experiment config") from error


def _parameter_count(value: Mapping[str, Any]) -> int:
    config = config_from_dict(dict(value))
    with torch.device("meta"):
        model = CoFiTokTiny(config.model)
    return sum(parameter.numel() for parameter in model.parameters())


def validate_terminal_snr_large_capacity_config_pair(
    *,
    cofitok_config: Mapping[str, Any],
    dense_config: Mapping[str, Any],
) -> dict[str, Any]:
    raw = {
        "cofitok": _object(cofitok_config, "large-capacity CoFiTok config"),
        "dense_identity": _object(
            dense_config, "large-capacity dense config"
        ),
    }
    resolved = {
        method: _resolved_config(value, f"large-capacity {method} config")
        for method, value in raw.items()
    }
    pair = generation_pair_contract(
        resolved["cofitok"], resolved["dense_identity"]
    )
    recipe = generation_training_recipe_contract(
        resolved["cofitok"],
        resolved["dense_identity"],
        stage="stability_full",
    )
    parameter_counts = {
        method: _parameter_count(value) for method, value in raw.items()
    }
    for method in METHODS:
        value = raw[method]
        diffusion = _object(value.get("diffusion"), f"{method} diffusion")
        model = _object(value.get("model"), f"{method} model")
        runtime = _object(value.get("runtime"), f"{method} runtime")
        data = _object(value.get("data"), f"{method} data")
        optimization = _object(
            value.get("optimization"), f"{method} optimization"
        )
        if (
            value.get("name") != CONFIG_NAMES[method]
            or diffusion.get("num_train_timesteps") != 1_000
            or diffusion.get("beta_start") != 0.0001
            or diffusion.get("beta_end") != 0.02
            or diffusion.get("schedule_type") != "cosine"
            or diffusion.get("prediction_target") != "epsilon"
            or diffusion.get("cosine_endpoint_fraction") != ENDPOINT_FRACTION
            or data.get("dataset") != "imagenet_256"
            or int(model.get("base_channels", -1)) != BASE_CHANNELS
            or int(runtime.get("steps", -1)) != TARGET_STEPS
            or runtime.get("protected_checkpoint_steps")
            != list(MILESTONE_STEPS)
            or int(data.get("batch_size", 0))
            * int(optimization.get("gradient_accumulation_steps", 0))
            != EFFECTIVE_BATCH_SIZE
            or parameter_counts[method] != EXPECTED_PARAMETER_COUNTS[method]
        ):
            raise ValueError(f"{method} endpoint0975 300K config differs")
    cofitok_model = _object(raw["cofitok"].get("model"), "CoFiTok model")
    if (
        cofitok_model.get("synthesis_mode") != "fixed_basis"
        or cofitok_model.get("synthesis_kernel_size") != 1
        or cofitok_model.get("gamma_mode") != "fixed_one"
        or cofitok_model.get("token_channel_schedule")
        != [4, 4, 8, 8, 8, 1, 1, 1]
        or cofitok_model.get("token_spatial_strides")
        != [16, 16, 8, 8, 4, 1, 1, 1]
    ):
        raise ValueError("large-capacity CoFiTok synthesis contract differs")
    if pair.get("valid") is not True or pair.get("issues") != []:
        raise ValueError("large-capacity matched pair contract differs")
    if (
        recipe.get("valid") is not True
        or recipe.get("issues") != []
        or recipe.get("stage") != "stability_full"
        or recipe.get("effective_batches", {}).get("cofitok", {}).get(
            "effective_batch_size"
        )
        != EFFECTIVE_BATCH_SIZE
        or recipe.get("effective_batches", {}).get("dense_identity", {}).get(
            "effective_batch_size"
        )
        != EFFECTIVE_BATCH_SIZE
    ):
        raise ValueError("large-capacity 300K training recipe differs")
    dense_count = parameter_counts["dense_identity"]
    relative_gap = (parameter_counts["cofitok"] - dense_count) / dense_count
    if abs(relative_gap) > 0.02:
        raise ValueError("large-capacity parameter gap exceeds two percent")
    return {
        "status": "pass",
        "recipe_schema": recipe["schema"],
        "recipe_stage": "stability_full",
        "parameter_counts": parameter_counts,
        "relative_parameter_gap": relative_gap,
        "base_channels": BASE_CHANNELS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "diffusion": {
            "num_train_timesteps": 1_000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_type": "cosine",
            "prediction_target": "epsilon",
            "cosine_endpoint_fraction": ENDPOINT_FRACTION,
        },
        "fresh_initialization_required": True,
        "resume_checkpoint_allowed": False,
        "cofitok_synthesis": {
            "mode": "fixed_basis",
            "kernel_size": 1,
            "gamma_mode": "fixed_one",
            "current_token_only": True,
            "zero_token_maps_to_zero": True,
        },
        "pair_contract_sha256": _canonical_sha256(pair),
        "recipe_contract_sha256": _canonical_sha256(recipe),
    }


def _expected_validation_path(result_path: str) -> str:
    path = PurePosixPath(result_path)
    if path.name != "terminal_snr_confirmation_result.json":
        raise ValueError("terminal-SNR confirmation result filename differs")
    return path.with_name("terminal_snr_confirmation_result.validation.json").as_posix()


def _milestone_contract(output_root: str) -> list[dict[str, Any]]:
    root = PurePosixPath(output_root)
    rows: list[dict[str, Any]] = []
    for step in MILESTONE_STEPS:
        checkpoints = {}
        for method in METHODS:
            checkpoint = (
                root
                / "training"
                / method
                / f"checkpoint_step_{step:08d}.pt"
            ).as_posix()
            checkpoints[method] = {
                "path": checkpoint,
                "integrity_sidecar": f"{checkpoint}.integrity.json",
            }
        rows.append(
            {
                "step": step,
                "images_seen_per_method": step * EFFECTIVE_BATCH_SIZE,
                "checkpoints": checkpoints,
                "trend_evaluation": copy.deepcopy(TREND_EVALUATION_CONTRACT),
            }
        )
    return rows


def build_terminal_snr_large_capacity_preparation(
    *,
    confirmation_result: Mapping[str, Any],
    confirmation_result_identity: Mapping[str, Any],
    confirmation_validation: Mapping[str, Any],
    confirmation_validation_identity: Mapping[str, Any],
    cofitok_config: Mapping[str, Any],
    cofitok_config_identity: Mapping[str, Any],
    dense_config: Mapping[str, Any],
    dense_config_identity: Mapping[str, Any],
    preparation_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    result = validate_terminal_snr_confirmation_result_contract(
        confirmation_result
    )
    replayed = replay_terminal_snr_confirmation_result(result)
    if replayed != result:
        raise ValueError("terminal-SNR confirmation differs from physical replay")
    result_id = _identity(
        confirmation_result_identity, "terminal-SNR confirmation result"
    )
    validation_id = _identity(
        confirmation_validation_identity,
        "terminal-SNR confirmation validation",
    )
    expected_validation_path = _expected_validation_path(result_id["path"])
    if validation_id["path"] != expected_validation_path:
        raise ValueError("terminal-SNR confirmation validation is not adjacent")
    validator_git = _git(
        confirmation_validation.get("validator_git"),
        "terminal-SNR confirmation validator",
    )
    validation = validate_terminal_snr_confirmation_validation_receipt(
        confirmation_validation,
        result=result,
        result_identity=result_id,
        validator_git=validator_git,
    )
    next_stage = _object(result.get("next_stage"), "confirmation next stage")
    claim = _object(result.get("claim_policy"), "confirmation claim policy")
    result_git = _git(result.get("result_git"), "confirmation result Git")
    if (
        result.get("scientific_status") != "confirmation_pass"
        or result.get("confirmation_pass") is not True
        or result.get("support_collapse_resolved") is not True
        or result.get("failed_checks") != []
        or result.get("decision")
        != "prepare_separate_large_capacity_readiness"
        or next_stage.get("route") != "large_capacity_readiness_preparation"
        or next_stage.get("support_collapse_resolved") is not True
        or next_stage.get("large_capacity_readiness_preparation_allowed")
        is not True
        or next_stage.get("large_capacity_readiness_launch_allowed") is not False
        or next_stage.get("separate_source_bound_authorization_required")
        is not True
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or claim.get("confirmation_is_large_capacity_qualification_only")
        is not True
        or claim.get("support_collapse_resolved") is not True
        or validation.get("status") != "pass"
        or validation.get("confirmation_pass") is not True
        or validation.get("support_collapse_resolved") is not True
        or validation.get("failed_checks") != []
        or validation.get("result") != result_id
        or validation.get("result_git") != result_git
        or validator_git != result_git
    ):
        raise ValueError(
            "large-capacity preparation requires the exact passing confirmation"
        )

    config_ids = {
        "cofitok": _identity(
            cofitok_config_identity, "large-capacity CoFiTok config"
        ),
        "dense_identity": _identity(
            dense_config_identity, "large-capacity dense config"
        ),
    }
    for method, descriptor in config_ids.items():
        if PurePosixPath(descriptor["path"]).name != CONFIG_FILENAMES[method]:
            raise ValueError(f"{method} large-capacity config path differs")
    config_validation = validate_terminal_snr_large_capacity_config_pair(
        cofitok_config=cofitok_config,
        dense_config=dense_config,
    )
    root = _absolute_posix(output_root, "large-capacity output root")
    if PurePosixPath(root).name != FULL_STAGE_DIRNAME:
        raise ValueError("large-capacity output root basename differs")
    training_dirs = {
        method: (PurePosixPath(root) / "training" / method).as_posix()
        for method in METHODS
    }
    preparation_checkout = _git(
        preparation_git, "large-capacity preparation checkout"
    )
    report = {
        "schema_version": PREPARATION_SCHEMA,
        "role": PREPARATION_ROLE,
        "status": "prepared",
        "scientific_status": "large_capacity_readiness_prepared",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation_git": preparation_checkout,
        "source_evidence": {
            "terminal_snr_confirmation_result": result_id,
            "terminal_snr_confirmation_validation": validation_id,
            "confirmation_result_git": result_git,
            "confirmation_validator_git": validator_git,
            "configs": config_ids,
        },
        "qualification": {
            "scientific_status": result["scientific_status"],
            "confirmation_pass": True,
            "support_collapse_resolved": True,
            "failed_checks": [],
            "check_count": len(result["checks"]),
            "confirmation_samples_per_arm": SAMPLE_COUNT,
        },
        "selection": {
            "dataset": "imagenet_256",
            "condition": "endpoint0975",
            "cosine_endpoint_fraction": ENDPOINT_FRACTION,
            "methods": list(METHODS),
            "base_channels": BASE_CHANNELS,
            "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "configured_training_steps": TARGET_STEPS,
            "images_seen_per_method": TARGET_STEPS * EFFECTIVE_BATCH_SIZE,
            "fresh_initialization_required": True,
            "resume_checkpoint_allowed": False,
            "output_root": root,
            "training_run_dirs": training_dirs,
            "config_names": copy.deepcopy(CONFIG_NAMES),
            "config_validation": config_validation,
        },
        "milestone_contract": _milestone_contract(root),
        "formal_evaluation_contract": copy.deepcopy(
            FORMAL_EVALUATION_CONTRACT
        ),
        "readiness_requirements": copy.deepcopy(READINESS_REQUIREMENTS),
        "next_stage": {
            "route": "build_separate_source_bound_large_capacity_stage_authorization",
            "preparation_selected": True,
            "readiness_evidence_still_required": True,
            "user_stage_authorization_required": True,
            "execution_authorization_required": True,
            "immutable_launch_receipt_required": True,
            "execution_ready": False,
            "remote_mutation_allowed": False,
            "gpu_execution_allowed": False,
            "training_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "release_allowed": False,
        },
        "claim_policy": {
            "role": "non_claim_large_capacity_readiness_preparation",
            "formal_generation_claim_allowed": False,
            "generation_advantage_proven": False,
            "confirmation_is_not_300k_execution_authorization": True,
            "terminal_completion_audit_required": True,
        },
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }
    return validate_terminal_snr_large_capacity_preparation_contract(
        report, expected_output_root=root
    )


def validate_terminal_snr_large_capacity_preparation_contract(
    report: Mapping[str, Any], *, expected_output_root: str | None = None
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR large-capacity preparation")
    sources = _object(row.get("source_evidence"), "preparation sources")
    qualification = _object(row.get("qualification"), "qualification")
    selection = _object(row.get("selection"), "large-capacity selection")
    next_stage = _object(row.get("next_stage"), "large-capacity next stage")
    claim = _object(row.get("claim_policy"), "large-capacity claim policy")
    config_validation = _object(
        selection.get("config_validation"), "large-capacity config validation"
    )
    root = _absolute_posix(selection.get("output_root"), "large-capacity root")
    if expected_output_root is not None and root != expected_output_root:
        raise ValueError("large-capacity expected output root differs")
    if PurePosixPath(root).name != FULL_STAGE_DIRNAME:
        raise ValueError("large-capacity output root basename differs")
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
            "qualification",
            "selection",
            "milestone_contract",
            "formal_evaluation_contract",
            "readiness_requirements",
            "next_stage",
            "claim_policy",
            "authorization_boundary",
        }
        or set(sources)
        != {
            "terminal_snr_confirmation_result",
            "terminal_snr_confirmation_validation",
            "confirmation_result_git",
            "confirmation_validator_git",
            "configs",
        }
        or set(qualification)
        != {
            "scientific_status",
            "confirmation_pass",
            "support_collapse_resolved",
            "failed_checks",
            "check_count",
            "confirmation_samples_per_arm",
        }
        or set(selection)
        != {
            "dataset",
            "condition",
            "cosine_endpoint_fraction",
            "methods",
            "base_channels",
            "parameter_counts",
            "effective_batch_size",
            "configured_training_steps",
            "images_seen_per_method",
            "fresh_initialization_required",
            "resume_checkpoint_allowed",
            "output_root",
            "training_run_dirs",
            "config_names",
            "config_validation",
        }
        or set(config_validation)
        != {
            "status",
            "recipe_schema",
            "recipe_stage",
            "parameter_counts",
            "relative_parameter_gap",
            "base_channels",
            "effective_batch_size",
            "diffusion",
            "fresh_initialization_required",
            "resume_checkpoint_allowed",
            "cofitok_synthesis",
            "pair_contract_sha256",
            "recipe_contract_sha256",
        }
        or set(next_stage)
        != {
            "route",
            "preparation_selected",
            "readiness_evidence_still_required",
            "user_stage_authorization_required",
            "execution_authorization_required",
            "immutable_launch_receipt_required",
            "execution_ready",
            "remote_mutation_allowed",
            "gpu_execution_allowed",
            "training_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_allowed",
            "release_allowed",
        }
        or set(claim)
        != {
            "role",
            "formal_generation_claim_allowed",
            "generation_advantage_proven",
            "confirmation_is_not_300k_execution_authorization",
            "terminal_completion_audit_required",
        }
        or row.get("schema_version") != PREPARATION_SCHEMA
        or row.get("role") != PREPARATION_ROLE
        or row.get("status") != "prepared"
        or row.get("scientific_status") != "large_capacity_readiness_prepared"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("authorization_boundary") != PREPARATION_BOUNDARY
        or qualification.get("scientific_status") != "confirmation_pass"
        or qualification.get("confirmation_pass") is not True
        or qualification.get("support_collapse_resolved") is not True
        or qualification.get("failed_checks") != []
        or int(qualification.get("check_count", 0)) < 1
        or int(qualification.get("confirmation_samples_per_arm", -1))
        != SAMPLE_COUNT
        or selection.get("dataset") != "imagenet_256"
        or selection.get("condition") != "endpoint0975"
        or selection.get("cosine_endpoint_fraction") != ENDPOINT_FRACTION
        or selection.get("methods") != list(METHODS)
        or int(selection.get("base_channels", -1)) != BASE_CHANNELS
        or selection.get("parameter_counts") != EXPECTED_PARAMETER_COUNTS
        or int(selection.get("effective_batch_size", -1))
        != EFFECTIVE_BATCH_SIZE
        or int(selection.get("configured_training_steps", -1)) != TARGET_STEPS
        or int(selection.get("images_seen_per_method", -1))
        != TARGET_STEPS * EFFECTIVE_BATCH_SIZE
        or selection.get("fresh_initialization_required") is not True
        or selection.get("resume_checkpoint_allowed") is not False
        or selection.get("config_names") != CONFIG_NAMES
        or config_validation.get("status") != "pass"
        or config_validation.get("parameter_counts")
        != EXPECTED_PARAMETER_COUNTS
        or config_validation.get("recipe_schema")
        != GENERATION_TRAINING_RECIPE_SCHEMA
        or config_validation.get("recipe_stage") != "stability_full"
        or config_validation.get("relative_parameter_gap")
        != (
            EXPECTED_PARAMETER_COUNTS["cofitok"]
            - EXPECTED_PARAMETER_COUNTS["dense_identity"]
        )
        / EXPECTED_PARAMETER_COUNTS["dense_identity"]
        or int(config_validation.get("base_channels", -1)) != BASE_CHANNELS
        or int(config_validation.get("effective_batch_size", -1))
        != EFFECTIVE_BATCH_SIZE
        or config_validation.get("diffusion")
        != {
            "num_train_timesteps": 1_000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_type": "cosine",
            "prediction_target": "epsilon",
            "cosine_endpoint_fraction": ENDPOINT_FRACTION,
        }
        or config_validation.get("fresh_initialization_required") is not True
        or config_validation.get("resume_checkpoint_allowed") is not False
        or config_validation.get("cofitok_synthesis")
        != {
            "mode": "fixed_basis",
            "kernel_size": 1,
            "gamma_mode": "fixed_one",
            "current_token_only": True,
            "zero_token_maps_to_zero": True,
        }
        or not _hex(config_validation.get("pair_contract_sha256"), 64)
        or not _hex(config_validation.get("recipe_contract_sha256"), 64)
        or row.get("formal_evaluation_contract")
        != FORMAL_EVALUATION_CONTRACT
        or row.get("readiness_requirements") != READINESS_REQUIREMENTS
        or next_stage.get("route")
        != "build_separate_source_bound_large_capacity_stage_authorization"
        or next_stage.get("preparation_selected") is not True
        or next_stage.get("readiness_evidence_still_required") is not True
        or next_stage.get("user_stage_authorization_required") is not True
        or next_stage.get("execution_authorization_required") is not True
        or next_stage.get("immutable_launch_receipt_required") is not True
        or next_stage.get("execution_ready") is not False
        or next_stage.get("remote_mutation_allowed") is not False
        or next_stage.get("gpu_execution_allowed") is not False
        or next_stage.get("training_launch_allowed") is not False
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or next_stage.get("promotion_allowed") is not False
        or next_stage.get("release_allowed") is not False
        or claim.get("formal_generation_claim_allowed") is not False
        or claim.get("generation_advantage_proven") is not False
        or claim.get("confirmation_is_not_300k_execution_authorization")
        is not True
        or claim.get("terminal_completion_audit_required") is not True
    ):
        raise ValueError("terminal-SNR large-capacity preparation contract differs")
    _git(row.get("preparation_git"), "large-capacity preparation Git")
    result_id = _identity(
        sources.get("terminal_snr_confirmation_result"),
        "terminal-SNR confirmation result",
    )
    validation_id = _identity(
        sources.get("terminal_snr_confirmation_validation"),
        "terminal-SNR confirmation validation",
    )
    if validation_id["path"] != _expected_validation_path(result_id["path"]):
        raise ValueError("terminal-SNR confirmation validation is not adjacent")
    result_git = _git(sources.get("confirmation_result_git"), "result Git")
    validator_git = _git(
        sources.get("confirmation_validator_git"), "validator Git"
    )
    if result_git != validator_git:
        raise ValueError("terminal-SNR confirmation validator Git differs")
    configs = _object(sources.get("configs"), "large-capacity configs")
    if set(configs) != set(METHODS):
        raise ValueError("large-capacity config source set differs")
    for method in METHODS:
        descriptor = _identity(configs[method], f"{method} config")
        if PurePosixPath(descriptor["path"]).name != CONFIG_FILENAMES[method]:
            raise ValueError(f"{method} config filename differs")
    training_dirs = _object(
        selection.get("training_run_dirs"), "large-capacity training dirs"
    )
    expected_training_dirs = {
        method: (PurePosixPath(root) / "training" / method).as_posix()
        for method in METHODS
    }
    if training_dirs != expected_training_dirs:
        raise ValueError("large-capacity training directory plan differs")
    if row.get("milestone_contract") != _milestone_contract(root):
        raise ValueError("large-capacity milestone contract differs")
    return copy.deepcopy(row)


__all__ = [
    "BASE_CHANNELS",
    "CONFIG_FILENAMES",
    "CONFIG_NAMES",
    "EFFECTIVE_BATCH_SIZE",
    "ENDPOINT_FRACTION",
    "EXPECTED_PARAMETER_COUNTS",
    "FORMAL_EVALUATION_CONTRACT",
    "FULL_STAGE_DIRNAME",
    "METHODS",
    "MILESTONE_STEPS",
    "PREPARATION_BOUNDARY",
    "PREPARATION_ROLE",
    "PREPARATION_SCHEMA",
    "READINESS_REQUIREMENTS",
    "TARGET_STEPS",
    "TREND_EVALUATION_CONTRACT",
    "build_terminal_snr_large_capacity_preparation",
    "validate_terminal_snr_large_capacity_config_pair",
    "validate_terminal_snr_large_capacity_preparation_contract",
]
