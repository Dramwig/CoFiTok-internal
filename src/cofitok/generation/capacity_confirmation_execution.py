"""Authorization and immutable launch receipt for capacity confirmation.

The confirmation evaluates four frozen step-10K checkpoints.  It can use the
GPU for sampling and evaluation, but it cannot train, mutate checkpoints, or
authorize the later 250M/300K run.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation.capacity_confirmation import (
    SAMPLE_BATCH_SIZE,
    SAMPLE_COUNT,
    SAMPLE_STEPS,
    validate_capacity_confirmation_preparation_contract,
)
from cofitok.generation.capacity_screen import ARM_NAMES, STOP_STEP


STAGE_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_capacity_confirmation_stage_authorization_v1"
)
STAGE_AUTHORIZATION_ROLE = "user_created_capacity_confirmation_execution_approval"
STAGE_AUTHORIZATION_SCOPE = "frozen_four_arm_capacity_confirmation_10k_only"
EXECUTION_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_capacity_confirmation_execution_authorization_v1"
)
EXECUTION_AUTHORIZATION_ROLE = (
    "exact_source_bound_capacity_confirmation_authorization"
)
LAUNCH_RECEIPT_SCHEMA = "cofitok_generation_capacity_confirmation_launch_receipt_v1"
LAUNCH_RECEIPT_ROLE = "immutable_capacity_confirmation_launch_receipt"
LIVE_SNAPSHOT_SCHEMA = "cofitok_generation_capacity_confirmation_live_snapshot_v1"
LIVE_SNAPSHOT_ROLE = "capacity_confirmation_live_prelaunch_snapshot"
MIN_FREE_BYTES = 120 * 1024**3

STAGE_BOUNDARY = {
    "decision_is_execution_authorization": True,
    "remote_mutation_allowed": True,
    "gpu_execution_allowed": True,
    "training_launch_allowed": False,
    "sampling_launch_allowed": True,
    "evaluation_launch_allowed": True,
    "checkpoint_mutation_allowed": False,
    "scope_limited_to_capacity_confirmation": True,
    "large_capacity_readiness_preparation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
EXECUTION_BOUNDARY = copy.deepcopy(STAGE_BOUNDARY)
LAUNCH_BOUNDARY = {
    "capacity_confirmation_execution_authorized": True,
    "frozen_checkpoint_training_allowed": False,
    "confirmation_samples_per_arm": SAMPLE_COUNT,
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


def capacity_confirmation_execution_lock_path(output_root: str) -> str:
    root = PurePosixPath(_absolute(output_root, "capacity confirmation root"))
    return (
        root.parent.parent
        / f".{root.parent.name}.{root.name}.capacity_confirmation_execution.lock"
    ).as_posix()


def capacity_confirmation_control_root(output_root: str) -> str:
    root = PurePosixPath(_absolute(output_root, "capacity confirmation root"))
    return (
        root.parent.parent
        / f".{root.parent.name}.{root.name}.capacity_confirmation_control"
    ).as_posix()


def _expected_output_dirs(output_root: str) -> dict[str, str]:
    root = PurePosixPath(_absolute(output_root, "capacity confirmation root"))
    return {arm: (root / arm).as_posix() for arm in ARM_NAMES}


def validate_capacity_confirmation_stage_authorization(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "capacity confirmation stage authorization")
    prep = _identity(preparation_identity, "capacity confirmation preparation")
    git = _git(execution_checkout, "capacity confirmation checkout")
    root = _absolute(expected_output_root, "capacity confirmation root")
    expected_selection = {
        "preparation": prep,
        "execution_checkout": git,
        "output_root": root,
        "frozen_arms": list(ARM_NAMES),
        "checkpoint_step": STOP_STEP,
        "samples_per_arm": SAMPLE_COUNT,
        "sample_steps": SAMPLE_STEPS,
        "sampling_batch_size": SAMPLE_BATCH_SIZE,
    }
    approval = _object(row.get("approval_record"), "capacity confirmation approval")
    if (
        row.get("schema_version") != STAGE_AUTHORIZATION_SCHEMA
        or row.get("role") != STAGE_AUTHORIZATION_ROLE
        or row.get("status") != "approved"
        or row.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or row.get("selection") != expected_selection
        or row.get("authorization_boundary") != STAGE_BOUNDARY
        or not str(approval.get("approved_by", "")).strip()
        or not str(approval.get("approved_at", "")).strip()
        or not str(approval.get("source_instruction", "")).strip()
    ):
        raise ValueError("capacity confirmation stage authorization differs")
    return copy.deepcopy(row)


def build_capacity_confirmation_stage_authorization(
    *,
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
    approved_by: str,
    approved_at: str,
    source_instruction: str,
) -> dict[str, Any]:
    prep = _identity(preparation_identity, "capacity confirmation preparation")
    git = _git(execution_checkout, "capacity confirmation checkout")
    root = _absolute(output_root, "capacity confirmation root")
    report = {
        "schema_version": STAGE_AUTHORIZATION_SCHEMA,
        "role": STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "selection": {
            "preparation": prep,
            "execution_checkout": git,
            "output_root": root,
            "frozen_arms": list(ARM_NAMES),
            "checkpoint_step": STOP_STEP,
            "samples_per_arm": SAMPLE_COUNT,
            "sample_steps": SAMPLE_STEPS,
            "sampling_batch_size": SAMPLE_BATCH_SIZE,
        },
        "approval_record": {
            "approved_by": str(approved_by).strip(),
            "approved_at": str(approved_at).strip(),
            "source_instruction": str(source_instruction).strip(),
        },
        "authorization_boundary": copy.deepcopy(STAGE_BOUNDARY),
    }
    return validate_capacity_confirmation_stage_authorization(
        report,
        preparation_identity=prep,
        execution_checkout=git,
        expected_output_root=root,
    )


def build_capacity_confirmation_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    prepared = validate_capacity_confirmation_preparation_contract(
        preparation, expected_output_root=output_root
    )
    prep_id = _identity(preparation_identity, "capacity confirmation preparation")
    stage_id = _identity(
        stage_authorization_identity, "capacity confirmation stage authorization"
    )
    git = _git(execution_checkout, "capacity confirmation checkout")
    validated_stage = validate_capacity_confirmation_stage_authorization(
        stage_authorization,
        preparation_identity=prep_id,
        execution_checkout=git,
        expected_output_root=output_root,
    )
    return {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA,
        "role": EXECUTION_AUTHORIZATION_ROLE,
        "status": "authorized",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "preparation": prep_id,
        "stage_authorization": stage_id,
        "validated_stage_authorization": validated_stage,
        "execution_checkout": git,
        "frozen_arms": copy.deepcopy(prepared["frozen_arms"]),
        "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        "output_root": _absolute(output_root, "capacity confirmation root"),
        "authorization_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
    }


def validate_capacity_confirmation_execution_authorization(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    expected_execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "capacity confirmation execution authorization")
    if (
        row.get("schema_version") != EXECUTION_AUTHORIZATION_SCHEMA
        or row.get("role") != EXECUTION_AUTHORIZATION_ROLE
        or row.get("status") != "authorized"
        or row.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or row.get("preparation")
        != _identity(preparation_identity, "capacity confirmation preparation")
        or row.get("execution_checkout")
        != _git(expected_execution_checkout, "capacity confirmation checkout")
        or row.get("output_root")
        != _absolute(expected_output_root, "capacity confirmation root")
        or row.get("authorization_boundary") != EXECUTION_BOUNDARY
        or set(_object(row.get("frozen_arms"), "capacity frozen arms"))
        != set(ARM_NAMES)
    ):
        raise ValueError("capacity confirmation execution authorization differs")
    return copy.deepcopy(row)


def validate_capacity_confirmation_live_snapshot(
    report: Mapping[str, Any],
    *,
    expected_output_root: str,
    expected_execution_checkout: Mapping[str, Any],
    expected_execution_lock: str,
) -> dict[str, Any]:
    row = _object(report, "capacity confirmation live snapshot")
    inventory = row.get("gpu_inventory")
    if not isinstance(inventory, list) or len(inventory) != 1:
        raise ValueError("capacity confirmation requires exactly one target GPU")
    gpu = _object(inventory[0], "capacity confirmation GPU")
    root = _absolute(expected_output_root, "capacity confirmation root")
    execution = _git(
        expected_execution_checkout,
        "capacity confirmation live checkout",
    )
    exact_lock = capacity_confirmation_execution_lock_path(root)
    if (
        row.get("schema_version") != LIVE_SNAPSHOT_SCHEMA
        or row.get("role") != LIVE_SNAPSHOT_ROLE
        or row.get("status") != "pass"
        or row.get("execution_checkout") != execution
        or row.get("output_root") != root
        or row.get("execution_lock") != exact_lock
        or _absolute(expected_execution_lock, "capacity confirmation lock")
        != exact_lock
        or not str(row.get("captured_at", "")).strip()
        or int(gpu.get("memory_used_mib", -1)) < 0
        or int(gpu.get("memory_used_mib", 17)) > 16
        or int(gpu.get("utilization_percent", 6)) > 5
        or int(gpu.get("memory_total_mib", 0)) < 1
        or row.get("gpu_compute_processes") != []
        or row.get("conflicting_processes") != []
        or row.get("output_root_absent") is not True
        or row.get("execution_lock_free") is not True
        or int(row.get("free_bytes", 0)) < MIN_FREE_BYTES
        or not _hex(row.get("runtime_environment_sha256"), 64)
        or not _hex(row.get("dataset_identity_sha256"), 64)
    ):
        raise ValueError("capacity confirmation live prelaunch snapshot differs")
    return copy.deepcopy(row)


def _validate_storage(report: Mapping[str, Any], *, output_root: str) -> dict[str, Any]:
    filesystem = _object(report.get("filesystem"), "capacity confirmation filesystem")
    plan = _object(report.get("plan"), "capacity confirmation storage plan")
    filesystem_path = PurePosixPath(str(filesystem.get("path", "")))
    output_path = PurePosixPath(output_root)
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or report.get("status") != "pass"
        or int(report.get("headroom_bytes", -1)) < 0
        or int(filesystem.get("free_bytes", 0)) < MIN_FREE_BYTES
        or int(plan.get("sample_count", -1)) < SAMPLE_COUNT * len(ARM_NAMES)
        or (filesystem_path != output_path and filesystem_path not in output_path.parents)
    ):
        raise ValueError("capacity confirmation storage evidence differs")
    return {
        "filesystem": copy.deepcopy(filesystem),
        "plan": copy.deepcopy(plan),
        "headroom_bytes": int(report["headroom_bytes"]),
    }


def build_capacity_confirmation_launch_receipt(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    storage_capacity: Mapping[str, Any],
    storage_capacity_identity: Mapping[str, Any],
    live_snapshot: Mapping[str, Any],
    live_snapshot_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
    output_dirs: Mapping[str, str],
    execution_lock: str,
    evaluation_state_absent_at_launch: bool,
) -> dict[str, Any]:
    prepared = validate_capacity_confirmation_preparation_contract(
        preparation, expected_output_root=output_root
    )
    prep_id = _identity(preparation_identity, "capacity confirmation preparation")
    authorization_id = _identity(
        execution_authorization_identity,
        "capacity confirmation execution authorization",
    )
    storage_id = _identity(storage_capacity_identity, "capacity confirmation storage")
    live_id = _identity(live_snapshot_identity, "capacity confirmation snapshot")
    git = _git(execution_checkout, "capacity confirmation launch checkout")
    authorization = validate_capacity_confirmation_execution_authorization(
        execution_authorization,
        preparation_identity=prep_id,
        expected_execution_checkout=git,
        expected_output_root=output_root,
    )
    if set(output_dirs) != set(ARM_NAMES):
        raise ValueError("capacity confirmation output arm set differs")
    root = _absolute(output_root, "capacity confirmation root")
    normalized_dirs = {
        arm: _absolute(output_dirs[arm], f"{arm} confirmation output")
        for arm in ARM_NAMES
    }
    if normalized_dirs != _expected_output_dirs(root):
        raise ValueError("capacity confirmation output directories differ")
    exact_lock = capacity_confirmation_execution_lock_path(root)
    if _absolute(execution_lock, "capacity confirmation lock") != exact_lock:
        raise ValueError("capacity confirmation execution lock differs")
    storage = _validate_storage(storage_capacity, output_root=root)
    live = validate_capacity_confirmation_live_snapshot(
        live_snapshot,
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=exact_lock,
    )
    if evaluation_state_absent_at_launch is not True:
        raise ValueError("capacity confirmation requires absent evaluation state")
    return {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "pass",
        "stage": "capacity_confirmation",
        "execution_checkout": git,
        "source_evidence": {
            "preparation": prep_id,
            "execution_authorization": authorization_id,
            "storage_capacity": storage_id,
            "live_snapshot": live_id,
            "capacity_screen_arm_validations": copy.deepcopy(
                prepared["source_evidence"]["capacity_screen_arm_validations"]
            ),
        },
        "authorization": authorization,
        "frozen_arms": copy.deepcopy(prepared["frozen_arms"]),
        "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        "storage_capacity": storage,
        "live_snapshot": live,
        "runtime_environment_sha256": live["runtime_environment_sha256"],
        "dataset_identity_sha256": live["dataset_identity_sha256"],
        "output_root": root,
        "output_dirs": normalized_dirs,
        "execution_lock": exact_lock,
        "evaluation_state_absent_at_launch": True,
        "authorization_boundary": copy.deepcopy(LAUNCH_BOUNDARY),
    }


def validate_capacity_confirmation_launch_receipt_contract(
    report: Mapping[str, Any],
    *,
    expected_execution_checkout: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "capacity confirmation launch receipt")
    git = _git(expected_execution_checkout, "capacity confirmation checkout")
    root = _absolute(row.get("output_root"), "capacity confirmation root")
    exact_lock = capacity_confirmation_execution_lock_path(root)
    live = validate_capacity_confirmation_live_snapshot(
        _object(row.get("live_snapshot"), "capacity confirmation live snapshot"),
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=exact_lock,
    )
    output_dirs = _object(row.get("output_dirs"), "capacity confirmation outputs")
    if (
        row.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or row.get("role") != LAUNCH_RECEIPT_ROLE
        or row.get("status") != "pass"
        or row.get("stage") != "capacity_confirmation"
        or row.get("execution_checkout") != git
        or row.get("evaluation_state_absent_at_launch") is not True
        or row.get("authorization_boundary") != LAUNCH_BOUNDARY
        or row.get("execution_lock") != exact_lock
        or output_dirs != _expected_output_dirs(root)
        or int(
            _object(
                row.get("evaluation_contract"), "capacity confirmation evaluation"
            ).get("samples_per_arm", -1)
        )
        != SAMPLE_COUNT
        or not _hex(row.get("runtime_environment_sha256"), 64)
        or not _hex(row.get("dataset_identity_sha256"), 64)
        or live.get("runtime_environment_sha256")
        != row.get("runtime_environment_sha256")
        or live.get("dataset_identity_sha256")
        != row.get("dataset_identity_sha256")
    ):
        raise ValueError("capacity confirmation launch receipt contract differs")
    return copy.deepcopy(row)


__all__ = [
    "EXECUTION_AUTHORIZATION_ROLE",
    "EXECUTION_AUTHORIZATION_SCHEMA",
    "EXECUTION_BOUNDARY",
    "LAUNCH_BOUNDARY",
    "LAUNCH_RECEIPT_ROLE",
    "LAUNCH_RECEIPT_SCHEMA",
    "LIVE_SNAPSHOT_ROLE",
    "LIVE_SNAPSHOT_SCHEMA",
    "MIN_FREE_BYTES",
    "STAGE_AUTHORIZATION_ROLE",
    "STAGE_AUTHORIZATION_SCHEMA",
    "STAGE_AUTHORIZATION_SCOPE",
    "STAGE_BOUNDARY",
    "build_capacity_confirmation_execution_authorization",
    "build_capacity_confirmation_launch_receipt",
    "build_capacity_confirmation_stage_authorization",
    "capacity_confirmation_control_root",
    "capacity_confirmation_execution_lock_path",
    "validate_capacity_confirmation_execution_authorization",
    "validate_capacity_confirmation_launch_receipt_contract",
    "validate_capacity_confirmation_live_snapshot",
    "validate_capacity_confirmation_stage_authorization",
]
