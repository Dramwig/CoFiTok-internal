"""Exact authorization and immutable launch receipt for the capacity screen."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation.capacity_screen import (
    ARM_NAMES,
    ARM_SPECS,
    CAPACITY_NAMES,
    CONFIGURED_STEPS,
    EFFECTIVE_BATCH,
    SAMPLE_COUNT,
    SAMPLE_STEPS,
    STOP_STEP,
    validate_capacity_screen_preparation_contract,
)


STAGE_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_capacity_screen_stage_authorization_v1"
)
STAGE_AUTHORIZATION_ROLE = "user_created_capacity_screen_execution_approval"
STAGE_AUTHORIZATION_SCOPE = "fresh_four_arm_capacity_screen_10k_execution_only"
EXECUTION_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_capacity_screen_execution_authorization_v1"
)
EXECUTION_AUTHORIZATION_ROLE = "exact_source_bound_capacity_screen_authorization"
LAUNCH_RECEIPT_SCHEMA = "cofitok_generation_capacity_screen_launch_receipt_v1"
LAUNCH_RECEIPT_ROLE = "immutable_capacity_screen_launch_receipt"
LIVE_SNAPSHOT_SCHEMA = "cofitok_generation_capacity_screen_live_snapshot_v1"
LIVE_SNAPSHOT_ROLE = "capacity_screen_live_prelaunch_snapshot"
MIN_FREE_BYTES = 120 * 1024**3

STAGE_BOUNDARY = {
    "decision_is_execution_authorization": True,
    "remote_mutation_allowed": True,
    "gpu_execution_allowed": True,
    "training_launch_allowed": True,
    "sampling_launch_allowed": True,
    "evaluation_launch_allowed": True,
    "scope_limited_to_capacity_screen": True,
    "intentional_stop_step_required": STOP_STEP,
    "configured_100k_completion_allowed": False,
    "capacity_confirmation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
EXECUTION_BOUNDARY = copy.deepcopy(STAGE_BOUNDARY)
LAUNCH_BOUNDARY = {
    "capacity_screen_execution_authorized": True,
    "training_must_stop_at_step": STOP_STEP,
    "screen_samples_per_arm": SAMPLE_COUNT,
    "configured_100k_completion_allowed": False,
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


def _absolute(value: Any, name: str) -> str:
    text = str(value or "")
    if not text or not (Path(text).is_absolute() or PurePosixPath(text).is_absolute()):
        raise ValueError(f"{name} must be absolute")
    return text


def _content_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("bytes", "sha256")}


def _runtime_git(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Git subset emitted by the shared runtime selector.

    ``checkout_identity`` also records the tree, while the selector's frozen
    lock intentionally stores the canonical training Git tuple only.  Requiring
    full dictionary equality would reject every real selector artifact.
    """

    git = _git(value, "capacity runtime checkout")
    return {
        "revision": git["revision"],
        "branch": git["branch"],
        "tracked_dirty": False,
    }


def capacity_screen_execution_lock_path(output_root: str) -> str:
    """Return the one allowed sibling execution lock for a result root."""

    root = _absolute(output_root, "capacity screen output root")
    posix_root = PurePosixPath(root)
    if not posix_root.is_absolute() or not posix_root.name:
        raise ValueError("capacity screen output root must be an absolute POSIX path")
    return (
        posix_root.parent / f".{posix_root.name}.capacity_screen_execution.lock"
    ).as_posix()


def capacity_screen_control_root(output_root: str) -> str:
    """Return the sibling directory reserved for prelaunch/control artifacts."""

    root = PurePosixPath(_absolute(output_root, "capacity screen output root"))
    if not root.name:
        raise ValueError("capacity screen output root must not be a filesystem root")
    return (root.parent / f".{root.name}.capacity_screen_control").as_posix()


def capacity_screen_benchmark_root(output_root: str) -> str:
    return (PurePosixPath(capacity_screen_control_root(output_root)) / "runtime_benchmarks").as_posix()


def _expected_run_dirs(output_root: str) -> dict[str, str]:
    root = PurePosixPath(_absolute(output_root, "capacity screen output root"))
    return {
        arm: (root / "training" / arm).as_posix()
        for arm in ARM_NAMES
    }


def _path_is_within(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return candidate == parent or parent in candidate.parents


def validate_capacity_screen_stage_authorization(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "capacity screen stage authorization")
    prep = _identity(preparation_identity, "capacity screen preparation")
    git = _git(execution_checkout, "capacity screen execution checkout")
    output_root = _absolute(expected_output_root, "capacity screen output root")
    expected_selection = {
        "preparation": prep,
        "execution_checkout": git,
        "output_root": output_root,
        "fresh_training_arms": list(ARM_NAMES),
        "configured_training_horizon": CONFIGURED_STEPS,
        "intentional_stop_step": STOP_STEP,
        "effective_batch_size": EFFECTIVE_BATCH,
        "screen_samples_per_arm": SAMPLE_COUNT,
        "screen_sample_steps": SAMPLE_STEPS,
    }
    approval = _object(row.get("approval_record"), "capacity screen approval record")
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
        raise ValueError("capacity screen stage authorization differs")
    return copy.deepcopy(row)


def build_capacity_screen_stage_authorization(
    *,
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
    approved_by: str,
    approved_at: str,
    source_instruction: str,
) -> dict[str, Any]:
    prep = _identity(preparation_identity, "capacity screen preparation")
    git = _git(execution_checkout, "capacity screen execution checkout")
    root = _absolute(output_root, "capacity screen output root")
    report = {
        "schema_version": STAGE_AUTHORIZATION_SCHEMA,
        "role": STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "selection": {
            "preparation": prep,
            "execution_checkout": git,
            "output_root": root,
            "fresh_training_arms": list(ARM_NAMES),
            "configured_training_horizon": CONFIGURED_STEPS,
            "intentional_stop_step": STOP_STEP,
            "effective_batch_size": EFFECTIVE_BATCH,
            "screen_samples_per_arm": SAMPLE_COUNT,
            "screen_sample_steps": SAMPLE_STEPS,
        },
        "approval_record": {
            "approved_by": str(approved_by).strip(),
            "approved_at": str(approved_at).strip(),
            "source_instruction": str(source_instruction).strip(),
        },
        "authorization_boundary": copy.deepcopy(STAGE_BOUNDARY),
    }
    return validate_capacity_screen_stage_authorization(
        report,
        preparation_identity=prep,
        execution_checkout=git,
        expected_output_root=root,
    )


def build_capacity_screen_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
    output_root: str,
) -> dict[str, Any]:
    prepared = validate_capacity_screen_preparation_contract(
        preparation,
        expected_output_root=output_root,
    )
    prep_id = _identity(preparation_identity, "capacity screen preparation")
    stage_id = _identity(stage_authorization_identity, "capacity stage authorization")
    git = _git(execution_checkout, "capacity screen execution checkout")
    validated_stage = validate_capacity_screen_stage_authorization(
        stage_authorization,
        preparation_identity=prep_id,
        execution_checkout=git,
        expected_output_root=output_root,
    )
    if set(config_identities) != set(ARM_NAMES):
        raise ValueError("capacity screen execution config set differs")
    configs = {
        arm: _identity(config_identities[arm], f"{arm} execution config")
        for arm in ARM_NAMES
    }
    prepared_configs = _object(prepared.get("configs"), "prepared capacity configs")
    for arm in ARM_NAMES:
        if _content_identity(configs[arm]) != _content_identity(
            _identity(prepared_configs[arm], f"prepared {arm} config")
        ):
            raise ValueError(f"{arm} execution config differs from preparation")
    return {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA,
        "role": EXECUTION_AUTHORIZATION_ROLE,
        "status": "authorized",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "preparation": prep_id,
        "stage_authorization": stage_id,
        "validated_stage_authorization": validated_stage,
        "execution_checkout": git,
        "configs": configs,
        "selection": copy.deepcopy(prepared["selection"]),
        "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        "output_root": _absolute(output_root, "capacity screen output root"),
        "authorization_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
    }


def validate_capacity_screen_execution_authorization(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    expected_execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "capacity screen execution authorization")
    if (
        row.get("schema_version") != EXECUTION_AUTHORIZATION_SCHEMA
        or row.get("role") != EXECUTION_AUTHORIZATION_ROLE
        or row.get("status") != "authorized"
        or row.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or row.get("preparation")
        != _identity(preparation_identity, "capacity screen preparation")
        or row.get("execution_checkout")
        != _git(expected_execution_checkout, "capacity execution checkout")
        or row.get("output_root")
        != _absolute(expected_output_root, "capacity screen output root")
        or row.get("authorization_boundary") != EXECUTION_BOUNDARY
        or set(row.get("configs", {})) != set(ARM_NAMES)
    ):
        raise ValueError("capacity screen execution authorization contract differs")
    return copy.deepcopy(row)


def validate_capacity_screen_live_snapshot(
    report: Mapping[str, Any],
    *,
    expected_output_root: str,
    expected_execution_checkout: Mapping[str, Any],
    expected_execution_lock: str,
) -> dict[str, Any]:
    row = _object(report, "capacity screen live snapshot")
    gpu = row.get("gpu_inventory")
    if not isinstance(gpu, list) or len(gpu) != 1:
        raise ValueError("capacity screen requires exactly one target GPU")
    device = _object(gpu[0], "capacity screen GPU inventory")
    output_root = _absolute(expected_output_root, "capacity screen output root")
    execution_git = _git(
        expected_execution_checkout,
        "capacity screen live execution checkout",
    )
    exact_lock = capacity_screen_execution_lock_path(output_root)
    supplied_lock = _absolute(
        expected_execution_lock,
        "capacity screen execution lock",
    )
    if (
        row.get("schema_version") != LIVE_SNAPSHOT_SCHEMA
        or row.get("role") != LIVE_SNAPSHOT_ROLE
        or row.get("status") != "pass"
        or row.get("execution_checkout") != execution_git
        or row.get("output_root") != output_root
        or row.get("execution_lock") != exact_lock
        or supplied_lock != exact_lock
        or not str(row.get("captured_at", "")).strip()
        or int(device.get("memory_used_mib", -1)) < 0
        or int(device.get("memory_used_mib", 17)) > 16
        or int(device.get("utilization_percent", 6)) > 5
        or int(device.get("memory_total_mib", 0)) < 1
        or row.get("gpu_compute_processes") != []
        or row.get("conflicting_processes") != []
        or row.get("output_root_absent") is not True
        or row.get("execution_lock_free") is not True
        or int(row.get("free_bytes", 0)) < MIN_FREE_BYTES
        or not _hex(row.get("runtime_environment_sha256"), length=64)
        or not _hex(row.get("dataset_identity_sha256"), length=64)
    ):
        raise ValueError("capacity screen live prelaunch snapshot differs")
    return copy.deepcopy(row)


def validate_capacity_screen_runtime_selection(
    report: Mapping[str, Any],
    *,
    execution_git: Mapping[str, Any],
    base256_config_identities: Mapping[str, Mapping[str, Any]],
    base256_run_dirs: list[str],
    benchmark_root: str,
) -> dict[str, Any]:
    selected = _object(report.get("selected"), "capacity runtime selected row")
    lock = _object(report.get("selection_lock"), "capacity runtime selection lock")
    micro_batch = int(selected.get("micro_batch_size", -1))
    accumulation = int(selected.get("gradient_accumulation_steps", -1))
    expected_config_sha = {
        "cofitok": base256_config_identities["cofitok"]["sha256"],
        "dense_identity": base256_config_identities["dense_identity"]["sha256"],
    }
    if (
        report.get("status") != "selected"
        or report.get("git_revision") != execution_git["revision"]
        or report.get("config_sha256") != expected_config_sha
        or report.get("benchmark_root") != benchmark_root
        or micro_batch < 1
        or accumulation < 1
        or micro_batch * accumulation != EFFECTIVE_BATCH
        or lock.get("training_run_dirs") != base256_run_dirs
        or int(lock.get("training_target_steps", -1)) != CONFIGURED_STEPS
        or int(lock.get("expected_effective_batch_size", -1)) != EFFECTIVE_BATCH
        or lock.get("git") != _runtime_git(execution_git)
        or lock.get("config_sha256") != expected_config_sha
        or lock.get("benchmark_root") != benchmark_root
        or not _hex(report.get("runtime_environment_sha256"), length=64)
        or not _hex(report.get("dataset_identity_sha256"), length=64)
    ):
        raise ValueError("capacity screen runtime selection differs")
    return {
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": EFFECTIVE_BATCH,
        "runtime_environment_sha256": report["runtime_environment_sha256"],
        "dataset_identity_sha256": report["dataset_identity_sha256"],
        "selection_sha256": report.get("selection_sha256"),
    }


def _validate_storage(report: Mapping[str, Any], *, output_root: str) -> dict[str, Any]:
    filesystem = _object(report.get("filesystem"), "capacity storage filesystem")
    plan = _object(report.get("plan"), "capacity storage plan")
    filesystem_path = str(filesystem.get("path", ""))
    output_parent = PurePosixPath(output_root).parent.as_posix()
    if (
        report.get("status") != "pass"
        or int(report.get("schema_version", -1)) != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or int(report.get("headroom_bytes", -1)) < 0
        or int(filesystem.get("free_bytes", 0)) < MIN_FREE_BYTES
        or int(plan.get("sample_count", -1)) < SAMPLE_COUNT * len(ARM_NAMES)
        or int(plan.get("checkpoint_count", -1)) < len(ARM_NAMES) * 2
        or not _path_is_within(output_parent, filesystem_path)
    ):
        raise ValueError("capacity screen storage evidence differs")
    return {
        "filesystem": copy.deepcopy(filesystem),
        "plan": copy.deepcopy(plan),
        "headroom_bytes": int(report["headroom_bytes"]),
    }


def build_capacity_screen_launch_receipt(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    runtime_selection: Mapping[str, Any],
    runtime_selection_identity: Mapping[str, Any],
    storage_capacity: Mapping[str, Any],
    storage_capacity_identity: Mapping[str, Any],
    live_snapshot: Mapping[str, Any],
    live_snapshot_identity: Mapping[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
    execution_checkout: Mapping[str, Any],
    output_root: str,
    execution_lock: str,
    run_dirs: Mapping[str, str],
    benchmark_root: str,
    training_state_absent_at_launch: bool,
) -> dict[str, Any]:
    root = _absolute(output_root, "capacity screen output root")
    prepared = validate_capacity_screen_preparation_contract(
        preparation,
        expected_output_root=root,
    )
    prep_id = _identity(preparation_identity, "capacity screen preparation")
    auth_id = _identity(
        execution_authorization_identity,
        "capacity screen execution authorization",
    )
    runtime_id = _identity(runtime_selection_identity, "capacity runtime selection")
    storage_id = _identity(storage_capacity_identity, "capacity storage evidence")
    live_id = _identity(live_snapshot_identity, "capacity live snapshot")
    git = _git(execution_checkout, "capacity screen launch checkout")
    authorization = validate_capacity_screen_execution_authorization(
        execution_authorization,
        preparation_identity=prep_id,
        expected_execution_checkout=git,
        expected_output_root=root,
    )
    if set(config_identities) != set(ARM_NAMES) or set(run_dirs) != set(ARM_NAMES):
        raise ValueError("capacity screen launch arm set differs")
    normalized_configs = {
        arm: _identity(config_identities[arm], f"{arm} launch config")
        for arm in ARM_NAMES
    }
    if normalized_configs != authorization["configs"]:
        raise ValueError("capacity screen launch configs differ from authorization")
    normalized_runs = {
        arm: _absolute(run_dirs[arm], f"{arm} run directory") for arm in ARM_NAMES
    }
    expected_runs = _expected_run_dirs(root)
    if normalized_runs != expected_runs:
        raise ValueError("capacity screen run directories differ from the fixed layout")
    exact_lock = capacity_screen_execution_lock_path(root)
    if _absolute(execution_lock, "capacity screen execution lock") != exact_lock:
        raise ValueError("capacity screen execution lock is not the exact sibling lock")
    base256_configs = {
        "cofitok": normalized_configs["base256_cofitok"],
        "dense_identity": normalized_configs["base256_dense_identity"],
    }
    base256_runs = [
        normalized_runs["base256_cofitok"],
        normalized_runs["base256_dense_identity"],
    ]
    runtime = validate_capacity_screen_runtime_selection(
        runtime_selection,
        execution_git=git,
        base256_config_identities=base256_configs,
        base256_run_dirs=base256_runs,
        benchmark_root=_absolute(benchmark_root, "capacity benchmark root"),
    )
    normalized_benchmark_root = _absolute(
        benchmark_root,
        "capacity benchmark root",
    )
    if _path_is_within(normalized_benchmark_root, root):
        raise ValueError("capacity runtime benchmarks must stay outside the result root")
    storage = _validate_storage(storage_capacity, output_root=root)
    live = validate_capacity_screen_live_snapshot(
        live_snapshot,
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=exact_lock,
    )
    if live["runtime_environment_sha256"] != runtime["runtime_environment_sha256"]:
        raise ValueError("capacity launch runtime environment changed")
    if live["dataset_identity_sha256"] != runtime["dataset_identity_sha256"]:
        raise ValueError("capacity launch dataset identity changed")
    if training_state_absent_at_launch is not True:
        raise ValueError("capacity screen requires absent initial training state")
    return {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "pass",
        "stage": "capacity_screen",
        "execution_checkout": git,
        "source_evidence": {
            "preparation": prep_id,
            "execution_authorization": auth_id,
            "runtime_selection": runtime_id,
            "storage_capacity": storage_id,
            "live_snapshot": live_id,
            "configs": normalized_configs,
        },
        "authorization": authorization,
        "selection": copy.deepcopy(prepared["selection"]),
        "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        "runtime_selection": runtime,
        "runtime_environment_sha256": runtime["runtime_environment_sha256"],
        "dataset_identity_sha256": runtime["dataset_identity_sha256"],
        "storage_capacity": storage,
        "live_snapshot": live,
        "output_root": root,
        "execution_lock": exact_lock,
        "run_dirs": normalized_runs,
        "benchmark_root": normalized_benchmark_root,
        "training_state_absent_at_launch": True,
        "authorization_boundary": copy.deepcopy(LAUNCH_BOUNDARY),
    }


def validate_capacity_screen_launch_receipt_contract(
    report: Mapping[str, Any],
    *,
    expected_execution_checkout: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "capacity screen launch receipt")
    git = _git(expected_execution_checkout, "capacity launch checkout")
    root = _absolute(row.get("output_root"), "capacity screen output root")
    exact_lock = capacity_screen_execution_lock_path(root)
    expected_runs = _expected_run_dirs(root)
    runtime = _object(row.get("runtime_selection"), "capacity runtime selection")
    live = validate_capacity_screen_live_snapshot(
        _object(row.get("live_snapshot"), "capacity live snapshot"),
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=exact_lock,
    )
    benchmark_root = _absolute(
        row.get("benchmark_root"),
        "capacity benchmark root",
    )
    if (
        row.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or row.get("role") != LAUNCH_RECEIPT_ROLE
        or row.get("status") != "pass"
        or row.get("stage") != "capacity_screen"
        or row.get("execution_checkout") != git
        or row.get("training_state_absent_at_launch") is not True
        or row.get("authorization_boundary") != LAUNCH_BOUNDARY
        or row.get("execution_lock") != exact_lock
        or _object(row.get("run_dirs"), "capacity run dirs") != expected_runs
        or int(runtime.get("effective_batch_size", -1)) != EFFECTIVE_BATCH
        or not _hex(runtime.get("runtime_environment_sha256"), length=64)
        or not _hex(runtime.get("dataset_identity_sha256"), length=64)
        or row.get("runtime_environment_sha256")
        != runtime.get("runtime_environment_sha256")
        or row.get("dataset_identity_sha256")
        != runtime.get("dataset_identity_sha256")
        or live.get("runtime_environment_sha256")
        != runtime.get("runtime_environment_sha256")
        or live.get("dataset_identity_sha256")
        != runtime.get("dataset_identity_sha256")
        or _path_is_within(benchmark_root, root)
    ):
        raise ValueError("capacity screen launch receipt contract differs")
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
    "build_capacity_screen_stage_authorization",
    "build_capacity_screen_execution_authorization",
    "build_capacity_screen_launch_receipt",
    "capacity_screen_benchmark_root",
    "capacity_screen_control_root",
    "capacity_screen_execution_lock_path",
    "validate_capacity_screen_execution_authorization",
    "validate_capacity_screen_launch_receipt_contract",
    "validate_capacity_screen_live_snapshot",
    "validate_capacity_screen_runtime_selection",
    "validate_capacity_screen_stage_authorization",
]
