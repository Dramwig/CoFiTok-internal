from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_CONFIGURED_STEPS,
    CAPACITY_PROBE_DATASET,
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_LARGE_CHANNELS,
    CAPACITY_PROBE_PARAMETER_COUNTS,
    CAPACITY_PROBE_PREPARATION_ROLE,
    CAPACITY_PROBE_SAMPLES_PER_ARM,
    CAPACITY_PROBE_STOP_STEP,
    validate_capacity_probe_preparation_contract,
)


CAPACITY_PROBE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION = 1
CAPACITY_PROBE_EXECUTION_AUTHORIZATION_ROLE = (
    "stability_full_data_capacity_probe_execution_authorization"
)
CAPACITY_PROBE_EXECUTION_SCOPE = (
    "stability_full_data_250m_capacity_probe_10k_execution_only"
)
CAPACITY_PROBE_LAUNCH_RECEIPT_SCHEMA_VERSION = 1
CAPACITY_PROBE_LAUNCH_RECEIPT_ROLE = (
    "stability_full_data_capacity_probe_launch_receipt"
)
CAPACITY_PROBE_RECIPE_STAGE = "stability_capacity_probe"
CAPACITY_PROBE_STORAGE_STAGE = (
    "stability_full_data_capacity_probe_250m_10k_execution"
)
CAPACITY_PROBE_MIN_STORAGE_SAMPLES = 4 * CAPACITY_PROBE_SAMPLES_PER_ARM
CAPACITY_PROBE_MIN_CHECKPOINT_COUNT = 4
CAPACITY_PROBE_SOURCE_NAMES = {
    "preparation",
    "execution_authorization",
    "cofitok_config",
    "dense_config",
    "config_validation",
    "storage_capacity",
    "runtime_selection",
}

STANDING_AUTHORIZATION_ROLE = "cofitok_standing_experiment_authorization_record"
STANDING_AUTHORIZATION_EXACT_TEXT = "之后不要我授权你直接运行需要的实验"
STANDING_AUTHORIZATION_INTERPRETATION = (
    "Authorize Codex to directly run future experiments that are necessary for "
    "the active CoFiTok long-term objective without requesting per-stage approval."
)
STANDING_AUTHORIZATION_SAFETY_BOUNDARIES = {
    "unrelated_project_processes_must_not_be_modified": True,
    "formal_remote_checkout_must_not_be_modified": True,
    "locked_evidence_must_not_be_overwritten": True,
    "independent_clean_checkout_required": True,
    "exact_revision_stage_and_output_binding_required": True,
    "stage_must_remain_non_authorizing_when_protocol_declares_non_authorizing": True,
}

EXECUTION_AUTHORIZATION_BOUNDARY = {
    "capacity_probe_execution_allowed": True,
    "gpu_execution_allowed": True,
    "training_launch_allowed": True,
    "scope_limited_to_capacity_probe": True,
    "intentional_stop_step_required": CAPACITY_PROBE_STOP_STEP,
    "configured_100k_completion_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
}

LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY = {
    "capacity_probe_execution_authorized": True,
    "training_must_stop_at_step": CAPACITY_PROBE_STOP_STEP,
    "configured_100k_completion_allowed": False,
    "full_300k_launch_allowed": False,
    "scaling_authorization_created": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
    "new_source_compatible_decision_required": True,
}


def _hex(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _hex(digest, length=64)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not _hex(revision, length=40)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def validate_standing_experiment_authorization(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    instruction = report.get("instruction")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != STANDING_AUTHORIZATION_ROLE
        or report.get("status") != "active"
        or not isinstance(instruction, Mapping)
        or instruction.get("language") != "zh-CN"
        or instruction.get("exact_text") != STANDING_AUTHORIZATION_EXACT_TEXT
        or instruction.get("interpretation") != STANDING_AUTHORIZATION_INTERPRETATION
        or not str(instruction.get("received_at", "")).strip()
        or report.get("preserved_safety_boundaries")
        != STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing experiment authorization contract differs")
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": copy.deepcopy(dict(instruction)),
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def build_capacity_probe_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    execution_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    normalized_preparation = _identity(
        preparation_identity,
        label="capacity probe preparation",
    )
    normalized_standing = _identity(
        standing_authorization_identity,
        label="standing experiment authorization",
    )
    selection = validate_capacity_probe_preparation_contract(
        preparation,
        expected_output_root=output_root,
    )
    standing = validate_standing_experiment_authorization(standing_authorization)
    git = _git(execution_git, label="capacity probe execution authorization")
    return {
        "schema_version": CAPACITY_PROBE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        "status": "authorized",
        "role": CAPACITY_PROBE_EXECUTION_AUTHORIZATION_ROLE,
        "scope": CAPACITY_PROBE_EXECUTION_SCOPE,
        "git": git,
        "preparation": normalized_preparation,
        "standing_authorization": {
            "source": normalized_standing,
            "validated_record": standing,
        },
        "selection": selection,
        "output_root": output_root,
        "authorization_boundary": copy.deepcopy(EXECUTION_AUTHORIZATION_BOUNDARY),
    }


def validate_capacity_probe_execution_authorization(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_PROBE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION
        or report.get("status") != "authorized"
        or report.get("role") != CAPACITY_PROBE_EXECUTION_AUTHORIZATION_ROLE
        or report.get("scope") != CAPACITY_PROBE_EXECUTION_SCOPE
        or report.get("authorization_boundary") != EXECUTION_AUTHORIZATION_BOUNDARY
        or report.get("output_root") != expected_output_root
        or report.get("preparation")
        != _identity(preparation_identity, label="capacity probe preparation")
    ):
        raise ValueError("capacity probe execution authorization contract differs")
    git = _git(report.get("git", {}), label="capacity probe execution authorization")
    if git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity probe execution authorization Git identity differs")
    standing = report.get("standing_authorization")
    if not isinstance(standing, Mapping):
        raise ValueError("capacity probe execution lacks standing authorization")
    source = _identity(
        standing.get("source", {}),
        label="standing experiment authorization",
    )
    validated = standing.get("validated_record")
    if not isinstance(validated, Mapping):
        raise ValueError("capacity probe standing authorization replay is missing")
    normalized = validate_standing_experiment_authorization(validated)
    selection = report.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("dataset") != CAPACITY_PROBE_DATASET
        or int(selection.get("configured_training_horizon", -1))
        != CAPACITY_PROBE_CONFIGURED_STEPS
        or int(selection.get("stop_after_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or int(selection.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or selection.get("output_root") != expected_output_root
    ):
        raise ValueError("capacity probe execution authorization selection differs")
    return {
        "scope": CAPACITY_PROBE_EXECUTION_SCOPE,
        "git": git,
        "output_root": expected_output_root,
        "standing_authorization": {"source": source, "validated_record": normalized},
        "authorization_boundary": copy.deepcopy(EXECUTION_AUTHORIZATION_BOUNDARY),
    }


def _validate_config_pair(report: Mapping[str, Any]) -> dict[str, Any]:
    recipe = report.get("training_recipe")
    cofitok = report.get("cofitok")
    dense = report.get("dense")
    if (
        report.get("status") != "pass"
        or report.get("mismatches") != []
        or not isinstance(recipe, Mapping)
        or recipe.get("valid") is not True
        or recipe.get("stage") != CAPACITY_PROBE_RECIPE_STAGE
        or not isinstance(cofitok, Mapping)
        or not isinstance(dense, Mapping)
        or int(cofitok.get("parameter_count", -1))
        != CAPACITY_PROBE_PARAMETER_COUNTS["base256"]["cofitok"]
        or int(dense.get("parameter_count", -1))
        != CAPACITY_PROBE_PARAMETER_COUNTS["base256"]["dense_identity"]
        or abs(float(report.get("relative_parameter_gap", 1.0))) > 0.02
    ):
        raise ValueError("capacity probe launch config validation differs")
    return {
        "status": "pass",
        "recipe_stage": CAPACITY_PROBE_RECIPE_STAGE,
        "parameter_counts": {
            "cofitok": int(cofitok["parameter_count"]),
            "dense_identity": int(dense["parameter_count"]),
        },
        "relative_parameter_gap": float(report["relative_parameter_gap"]),
        "effective_batches": copy.deepcopy(recipe.get("effective_batches")),
        "recipe_schema": recipe.get("schema"),
    }


def _validate_storage_capacity(
    report: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    expected_storage_path: str,
) -> dict[str, Any]:
    filesystem = report.get("filesystem")
    plan = report.get("plan")
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or report.get("stage") != CAPACITY_PROBE_STORAGE_STAGE
        or report.get("status") != "pass"
        or report.get("git") != dict(expected_git)
        or not isinstance(filesystem, Mapping)
        or not isinstance(plan, Mapping)
        or str(filesystem.get("path", "")) != expected_storage_path
        or int(plan.get("sample_count", -1)) < CAPACITY_PROBE_MIN_STORAGE_SAMPLES
        or int(plan.get("checkpoint_count", -1))
        < CAPACITY_PROBE_MIN_CHECKPOINT_COUNT
        or float(plan.get("checkpoint_size_multiplier", 0.0)) < 4.0
        or int(report.get("headroom_bytes", -1)) < 0
    ):
        raise ValueError("capacity probe launch storage evidence differs")
    return {
        "storage_path": expected_storage_path,
        "free_bytes": int(filesystem["free_bytes"]),
        "required_free_bytes": int(plan["required_free_bytes"]),
        "headroom_bytes": int(report["headroom_bytes"]),
    }


def _validate_runtime_selection(
    report: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    expected_config_sha256: Mapping[str, str],
    expected_run_dirs: list[str],
    expected_benchmark_root: str,
) -> dict[str, Any]:
    selected = report.get("selected")
    lock = report.get("selection_lock")
    if not isinstance(selected, Mapping) or not isinstance(lock, Mapping):
        raise ValueError("capacity probe runtime selection is incomplete")
    micro_batch = int(selected.get("micro_batch_size", -1))
    accumulation = int(selected.get("gradient_accumulation_steps", -1))
    if (
        report.get("status") != "selected"
        or report.get("git_revision") != expected_git["revision"]
        or report.get("config_sha256") != dict(expected_config_sha256)
        or str(report.get("benchmark_root", "")) != expected_benchmark_root
        or micro_batch < 1
        or accumulation < 1
        or micro_batch * accumulation != CAPACITY_PROBE_EFFECTIVE_BATCH
        or lock.get("training_run_dirs") != expected_run_dirs
        or int(lock.get("training_target_steps", -1))
        != CAPACITY_PROBE_CONFIGURED_STEPS
        or int(lock.get("expected_effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or lock.get("git") != dict(expected_git)
        or lock.get("config_sha256") != dict(expected_config_sha256)
        or str(lock.get("benchmark_root", "")) != expected_benchmark_root
    ):
        raise ValueError("capacity probe runtime selection contract differs")
    environment_sha = str(report.get("runtime_environment_sha256", ""))
    if not _hex(environment_sha, length=64):
        raise ValueError("capacity probe runtime environment identity is malformed")
    return {
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro_batch * accumulation,
        "runtime_environment_sha256": environment_sha,
    }


def build_capacity_probe_launch_receipt(
    *,
    preparation: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    config_validation: Mapping[str, Any],
    storage_capacity: Mapping[str, Any],
    runtime_selection: Mapping[str, Any],
    source_identities: Mapping[str, Mapping[str, Any]],
    expected_revision: str,
    expected_branch: str,
    output_root: str,
    storage_path: str,
    training_run_dirs: list[str],
    benchmark_root: str,
    training_state_absent_at_launch: bool,
) -> dict[str, Any]:
    if set(source_identities) != CAPACITY_PROBE_SOURCE_NAMES:
        raise ValueError("capacity probe launch receipt source set differs")
    normalized_sources = {
        name: _identity(identity, label=name)
        for name, identity in source_identities.items()
    }
    selection = validate_capacity_probe_preparation_contract(
        preparation,
        expected_output_root=output_root,
    )
    expected_git = _git(
        {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        },
        label="capacity probe launch",
    )
    authorization = validate_capacity_probe_execution_authorization(
        execution_authorization,
        preparation_identity=normalized_sources["preparation"],
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_output_root=output_root,
    )
    matched_config = _validate_config_pair(config_validation)
    if matched_config != preparation.get("matched_training_contract", {}).get(
        "base256"
    ):
        raise ValueError(
            "capacity probe launch config validation differs from preparation"
        )
    expected_config_sha256 = {
        "cofitok": normalized_sources["cofitok_config"]["sha256"],
        "dense_identity": normalized_sources["dense_config"]["sha256"],
    }
    preparation_configs = preparation.get("matched_training_contract", {}).get(
        "capacity_configs"
    )
    if not isinstance(preparation_configs, Mapping):
        raise ValueError("capacity probe preparation config identities are missing")
    preparation_config_content = {
        method: {
            key: value.get(key)
            for key in ("bytes", "sha256")
        }
        for method, value in preparation_configs.items()
        if isinstance(value, Mapping)
    }
    launch_config_content = {
        "cofitok": {
            key: normalized_sources["cofitok_config"][key]
            for key in ("bytes", "sha256")
        },
        "dense_identity": {
            key: normalized_sources["dense_config"][key]
            for key in ("bytes", "sha256")
        },
    }
    if preparation_config_content != launch_config_content:
        raise ValueError("capacity probe launch config identities differ")
    if len(training_run_dirs) != 2 or len(set(training_run_dirs)) != 2:
        raise ValueError("capacity probe requires exactly two fresh training runs")
    storage = _validate_storage_capacity(
        storage_capacity,
        expected_git=expected_git,
        expected_storage_path=storage_path,
    )
    runtime = _validate_runtime_selection(
        runtime_selection,
        expected_git=expected_git,
        expected_config_sha256=expected_config_sha256,
        expected_run_dirs=training_run_dirs,
        expected_benchmark_root=benchmark_root,
    )
    if training_state_absent_at_launch is not True:
        raise ValueError("capacity probe launch requires absent initial training state")
    return {
        "schema_version": CAPACITY_PROBE_LAUNCH_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "role": CAPACITY_PROBE_LAUNCH_RECEIPT_ROLE,
        "stage": CAPACITY_PROBE_RECIPE_STAGE,
        "git": expected_git,
        "source_reports": normalized_sources,
        "authorization": authorization,
        "selection": selection,
        "matched_config": matched_config,
        "runtime_selection": runtime,
        "storage_capacity": storage,
        "output_root": output_root,
        "storage_path": storage_path,
        "training_run_dirs": training_run_dirs,
        "benchmark_root": benchmark_root,
        "training_state_absent_at_launch": True,
        "authorization_boundary": copy.deepcopy(
            LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY
        ),
    }


def validate_capacity_probe_launch_receipt_contract(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_PROBE_LAUNCH_RECEIPT_SCHEMA_VERSION
        or report.get("status") != "pass"
        or report.get("role") != CAPACITY_PROBE_LAUNCH_RECEIPT_ROLE
        or report.get("stage") != CAPACITY_PROBE_RECIPE_STAGE
        or report.get("training_state_absent_at_launch") is not True
        or report.get("authorization_boundary")
        != LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("capacity probe launch receipt contract differs")
    git = _git(report.get("git", {}), label="capacity probe launch receipt")
    if git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity probe launch receipt Git identity differs")
    sources = report.get("source_reports")
    if not isinstance(sources, Mapping) or set(sources) != CAPACITY_PROBE_SOURCE_NAMES:
        raise ValueError("capacity probe launch receipt source set differs")
    for name, identity in sources.items():
        _identity(identity, label=f"capacity probe launch source {name}")
    selection = report.get("selection")
    runtime = report.get("runtime_selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("dataset") != CAPACITY_PROBE_DATASET
        or int(selection.get("stop_after_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or int(selection.get("base_channels", [0, 0])[-1])
        != CAPACITY_PROBE_LARGE_CHANNELS
        or not isinstance(runtime, Mapping)
        or int(runtime.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
    ):
        raise ValueError("capacity probe launch receipt selection differs")
    output_root = str(report.get("output_root", ""))
    if not output_root or not PurePosixPath(output_root).is_absolute():
        raise ValueError("capacity probe launch output root is not absolute")
    return {
        "git": git,
        "output_root": output_root,
        "stop_after_step": CAPACITY_PROBE_STOP_STEP,
        "effective_batch_size": CAPACITY_PROBE_EFFECTIVE_BATCH,
        "authorization_boundary": copy.deepcopy(
            LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY
        ),
    }
