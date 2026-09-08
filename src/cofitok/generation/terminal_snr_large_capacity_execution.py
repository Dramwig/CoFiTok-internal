"""User-stage authorization boundary for terminal-SNR large-capacity work.

The stage authorization created here is deliberately *not* a training or
execution authorization.  It binds the user's explicit 250M/300K goal to one
passing terminal-SNR preparation and one exact clean execution checkout, then
permits only the prelaunch evidence collection needed to construct a later
source-bound execution authorization and immutable launch receipt.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation.terminal_snr_large_capacity import (
    BASE_CHANNELS,
    CONFIG_FILENAMES,
    EFFECTIVE_BATCH_SIZE,
    ENDPOINT_FRACTION,
    EXPECTED_PARAMETER_COUNTS,
    FORMAL_EVALUATION_CONTRACT,
    METHODS,
    MILESTONE_STEPS,
    TARGET_STEPS,
    validate_terminal_snr_large_capacity_config_pair,
    validate_terminal_snr_large_capacity_preparation_contract,
)


STAGE_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_terminal_snr_large_capacity_stage_authorization_v1"
)
STAGE_AUTHORIZATION_ROLE = (
    "user_goal_bound_terminal_snr_large_capacity_prelaunch_approval"
)
STAGE_AUTHORIZATION_SCOPE = (
    "fresh_endpoint0975_matched_base256_250m_300k_prelaunch_evidence_only"
)
PAIR_VALIDATION_SCHEMA = (
    "cofitok_generation_terminal_snr_large_capacity_pair_validation_v1"
)
PAIR_VALIDATION_ROLE = (
    "source_bound_terminal_snr_large_capacity_matched_pair_validation"
)

GOAL_BINDING = {
    "project": "CoFiTok",
    "training": {
        "initialization": "fresh",
        "dataset": "imagenet_256",
        "condition": "terminal_snr_endpoint0975",
        "methods": list(METHODS),
        "base_channels": BASE_CHANNELS,
        "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
        "steps_per_method": TARGET_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    },
    "formal_evaluation": copy.deepcopy(FORMAL_EVALUATION_CONTRACT),
    "terminal_condition": (
        "release requires a passing final gate, release-authorized EMA "
        "artifacts, locked paper integration, and terminal completion audit"
    ),
}

STAGE_BOUNDARY = {
    "user_goal_bound": True,
    "decision_is_execution_authorization": False,
    "readiness_evidence_collection_allowed": True,
    "pair_validation_allowed": True,
    "gpu_runtime_preflight_allowed": True,
    "storage_preflight_allowed": True,
    "live_snapshot_capture_allowed": True,
    "remote_checkout_mutation_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "checkpoint_mutation_allowed": False,
    "resume_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "separate_execution_authorization_required": True,
    "immutable_launch_receipt_required": True,
}

PAIR_VALIDATION_BOUNDARY = {
    "matched_pair_validated": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "checkpoint_mutation_allowed": False,
    "resume_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "runtime_selection_still_required": True,
    "storage_capacity_still_required": True,
    "live_snapshot_still_required": True,
    "separate_execution_authorization_required": True,
    "immutable_launch_receipt_required": True,
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
    path = str(row.get("path", ""))
    if (
        not path
        or not (Path(path).is_absolute() or PurePosixPath(path).is_absolute())
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


def _absolute(value: Any, name: str) -> str:
    text = str(value or "")
    if not text or not (
        Path(text).is_absolute() or PurePosixPath(text).is_absolute()
    ):
        raise ValueError(f"{name} must be absolute")
    return text


def _explicit_goal_instruction(value: Any) -> str:
    text = str(value or "").strip()
    normalized = text.lower()
    required = ("250m", "300k", "50k", "ddim-250", "terminal")
    if not text or any(token not in normalized for token in required):
        raise ValueError(
            "large-capacity source instruction must bind the explicit "
            "250M/300K, 50K DDIM-250, and terminal-audit goal"
        )
    return text


def _selection(
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    prepared = validate_terminal_snr_large_capacity_preparation_contract(
        preparation
    )
    root = _absolute(output_root, "large-capacity output root")
    selected = _object(prepared.get("selection"), "large-capacity selection")
    sources = _object(prepared.get("source_evidence"), "preparation sources")
    configs = _object(sources.get("configs"), "large-capacity configs")
    if (
        selected.get("output_root") != root
        or selected.get("fresh_initialization_required") is not True
        or selected.get("resume_checkpoint_allowed") is not False
        or selected.get("condition") != "endpoint0975"
        or selected.get("methods") != list(METHODS)
        or int(selected.get("configured_training_steps", -1)) != TARGET_STEPS
        or set(configs) != set(METHODS)
    ):
        raise ValueError("large-capacity preparation selection differs")
    config_ids = {
        method: _identity(configs[method], f"{method} prepared config")
        for method in METHODS
    }
    return {
        "preparation": _identity(
            preparation_identity, "large-capacity preparation"
        ),
        "execution_checkout": _git(
            execution_checkout, "large-capacity execution checkout"
        ),
        "output_root": root,
        "configs": config_ids,
        "condition": "terminal_snr_endpoint0975",
        "cosine_endpoint_fraction": ENDPOINT_FRACTION,
        "methods": list(METHODS),
        "base_channels": BASE_CHANNELS,
        "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "configured_training_steps": TARGET_STEPS,
        "milestone_steps": list(MILESTONE_STEPS),
        "fresh_initialization_required": True,
        "resume_allowed": False,
        "formal_evaluation": copy.deepcopy(FORMAL_EVALUATION_CONTRACT),
    }


def build_terminal_snr_large_capacity_stage_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
    approved_by: str,
    approved_at: str,
    source_instruction: str,
) -> dict[str, Any]:
    report = {
        "schema_version": STAGE_AUTHORIZATION_SCHEMA,
        "role": STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "selection": _selection(
            preparation,
            preparation_identity,
            execution_checkout,
            output_root,
        ),
        "approval_record": {
            "approved_by": str(approved_by).strip(),
            "approved_at": str(approved_at).strip(),
            "source_instruction": _explicit_goal_instruction(source_instruction),
            "goal_binding": copy.deepcopy(GOAL_BINDING),
        },
        "next_stage": {
            "route": "collect_source_bound_large_capacity_prelaunch_evidence",
            "pair_validation_required": True,
            "runtime_selection_required": True,
            "storage_capacity_required": True,
            "live_snapshot_required": True,
            "separate_execution_authorization_required": True,
            "immutable_launch_receipt_required": True,
            "execution_ready": False,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(STAGE_BOUNDARY),
    }
    return validate_terminal_snr_large_capacity_stage_authorization(
        report,
        preparation=preparation,
        preparation_identity=preparation_identity,
        execution_checkout=execution_checkout,
        expected_output_root=output_root,
    )


def validate_terminal_snr_large_capacity_stage_authorization(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "large-capacity stage authorization")
    approval = _object(row.get("approval_record"), "large-capacity approval")
    next_stage = _object(row.get("next_stage"), "large-capacity next stage")
    expected_selection = _selection(
        preparation,
        preparation_identity,
        execution_checkout,
        expected_output_root,
    )
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "scope",
            "selection",
            "approval_record",
            "next_stage",
            "authorization_boundary",
        }
        or set(approval)
        != {"approved_by", "approved_at", "source_instruction", "goal_binding"}
        or set(next_stage)
        != {
            "route",
            "pair_validation_required",
            "runtime_selection_required",
            "storage_capacity_required",
            "live_snapshot_required",
            "separate_execution_authorization_required",
            "immutable_launch_receipt_required",
            "execution_ready",
            "training_launch_allowed",
            "full_300k_launch_allowed",
        }
        or row.get("schema_version") != STAGE_AUTHORIZATION_SCHEMA
        or row.get("role") != STAGE_AUTHORIZATION_ROLE
        or row.get("status") != "approved"
        or row.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or row.get("selection") != expected_selection
        or row.get("authorization_boundary") != STAGE_BOUNDARY
        or not str(approval.get("approved_by", "")).strip()
        or not str(approval.get("approved_at", "")).strip()
        or _explicit_goal_instruction(approval.get("source_instruction"))
        != approval.get("source_instruction")
        or approval.get("goal_binding") != GOAL_BINDING
        or next_stage.get("route")
        != "collect_source_bound_large_capacity_prelaunch_evidence"
        or next_stage.get("pair_validation_required") is not True
        or next_stage.get("runtime_selection_required") is not True
        or next_stage.get("storage_capacity_required") is not True
        or next_stage.get("live_snapshot_required") is not True
        or next_stage.get("separate_execution_authorization_required") is not True
        or next_stage.get("immutable_launch_receipt_required") is not True
        or next_stage.get("execution_ready") is not False
        or next_stage.get("training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("terminal-SNR large-capacity stage authorization differs")
    return copy.deepcopy(row)


def _pair_validation_report(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    cofitok_config: Mapping[str, Any],
    cofitok_config_identity: Mapping[str, Any],
    dense_config: Mapping[str, Any],
    dense_config_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    prepared = validate_terminal_snr_large_capacity_preparation_contract(
        preparation
    )
    prepared_id = _identity(
        preparation_identity, "large-capacity preparation"
    )
    execution = _git(
        execution_checkout, "large-capacity execution checkout"
    )
    root = _absolute(output_root, "large-capacity output root")
    stage = validate_terminal_snr_large_capacity_stage_authorization(
        stage_authorization,
        preparation=prepared,
        preparation_identity=prepared_id,
        execution_checkout=execution,
        expected_output_root=root,
    )
    stage_id = _identity(
        stage_authorization_identity, "large-capacity stage authorization"
    )
    prepared_sources = _object(
        prepared.get("source_evidence"), "large-capacity preparation sources"
    )
    prepared_configs = _object(
        prepared_sources.get("configs"), "large-capacity prepared configs"
    )
    config_ids = {
        "cofitok": _identity(
            cofitok_config_identity, "large-capacity CoFiTok config"
        ),
        "dense_identity": _identity(
            dense_config_identity, "large-capacity dense config"
        ),
    }
    if set(prepared_configs) != set(METHODS) or any(
        config_ids[method]
        != _identity(prepared_configs[method], f"prepared {method} config")
        for method in METHODS
    ):
        raise ValueError("large-capacity pair-validation config identity differs")
    for method in METHODS:
        if PurePosixPath(config_ids[method]["path"]).name != CONFIG_FILENAMES[method]:
            raise ValueError(
                f"large-capacity {method} pair-validation config path differs"
            )
    validation = validate_terminal_snr_large_capacity_config_pair(
        cofitok_config=cofitok_config,
        dense_config=dense_config,
    )
    prepared_selection = _object(
        prepared.get("selection"), "large-capacity prepared selection"
    )
    if validation != prepared_selection.get("config_validation"):
        raise ValueError(
            "large-capacity pair validation differs from preparation"
        )
    return {
        "schema_version": PAIR_VALIDATION_SCHEMA,
        "role": PAIR_VALIDATION_ROLE,
        "status": "pass",
        "execution_checkout": execution,
        "source_evidence": {
            "preparation": prepared_id,
            "stage_authorization": stage_id,
            "configs": config_ids,
        },
        "selection": {
            "output_root": root,
            "condition": "terminal_snr_endpoint0975",
            "methods": list(METHODS),
            "base_channels": BASE_CHANNELS,
            "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "configured_training_steps": TARGET_STEPS,
            "milestone_steps": list(MILESTONE_STEPS),
            "fresh_initialization_required": True,
            "resume_allowed": False,
        },
        "validation": validation,
        "next_stage": {
            "route": "collect_large_capacity_runtime_storage_and_live_snapshot",
            "runtime_selection_required": True,
            "storage_capacity_required": True,
            "live_snapshot_required": True,
            "separate_execution_authorization_required": True,
            "immutable_launch_receipt_required": True,
            "execution_ready": False,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(PAIR_VALIDATION_BOUNDARY),
    }


def build_terminal_snr_large_capacity_pair_validation(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    cofitok_config: Mapping[str, Any],
    cofitok_config_identity: Mapping[str, Any],
    dense_config: Mapping[str, Any],
    dense_config_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    report = _pair_validation_report(
        preparation=preparation,
        preparation_identity=preparation_identity,
        stage_authorization=stage_authorization,
        stage_authorization_identity=stage_authorization_identity,
        cofitok_config=cofitok_config,
        cofitok_config_identity=cofitok_config_identity,
        dense_config=dense_config,
        dense_config_identity=dense_config_identity,
        execution_checkout=execution_checkout,
        output_root=output_root,
    )
    return validate_terminal_snr_large_capacity_pair_validation(
        report,
        preparation=preparation,
        preparation_identity=preparation_identity,
        stage_authorization=stage_authorization,
        stage_authorization_identity=stage_authorization_identity,
        cofitok_config=cofitok_config,
        cofitok_config_identity=cofitok_config_identity,
        dense_config=dense_config,
        dense_config_identity=dense_config_identity,
        execution_checkout=execution_checkout,
        expected_output_root=output_root,
    )


def validate_terminal_snr_large_capacity_pair_validation(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    cofitok_config: Mapping[str, Any],
    cofitok_config_identity: Mapping[str, Any],
    dense_config: Mapping[str, Any],
    dense_config_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "large-capacity pair validation")
    expected = _pair_validation_report(
        preparation=preparation,
        preparation_identity=preparation_identity,
        stage_authorization=stage_authorization,
        stage_authorization_identity=stage_authorization_identity,
        cofitok_config=cofitok_config,
        cofitok_config_identity=cofitok_config_identity,
        dense_config=dense_config,
        dense_config_identity=dense_config_identity,
        execution_checkout=execution_checkout,
        output_root=expected_output_root,
    )
    if row != expected:
        raise ValueError("terminal-SNR large-capacity pair validation differs")
    return copy.deepcopy(row)


__all__ = [
    "GOAL_BINDING",
    "PAIR_VALIDATION_BOUNDARY",
    "PAIR_VALIDATION_ROLE",
    "PAIR_VALIDATION_SCHEMA",
    "STAGE_AUTHORIZATION_ROLE",
    "STAGE_AUTHORIZATION_SCHEMA",
    "STAGE_AUTHORIZATION_SCOPE",
    "STAGE_BOUNDARY",
    "build_terminal_snr_large_capacity_pair_validation",
    "build_terminal_snr_large_capacity_stage_authorization",
    "validate_terminal_snr_large_capacity_pair_validation",
    "validate_terminal_snr_large_capacity_stage_authorization",
]
