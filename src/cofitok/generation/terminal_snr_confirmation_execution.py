"""Fail-closed authorization and launch receipt for terminal-SNR confirmation.

The authorized stage may only sample and evaluate the four checkpoints frozen
by a passing terminal-SNR screen. It cannot train, mutate checkpoints, signal
processes, authorize 250M/300K training, export, or release artifacts.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.data.provenance import validate_dataset_provenance
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.exposure_capacity_authorization import (
    identity as artifact_identity,
    read_object,
)
from cofitok.generation.terminal_snr_confirmation import (
    SAMPLE_BATCH_SIZE,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    SAMPLE_STEPS,
    validate_terminal_snr_confirmation_preparation_contract,
)
from cofitok.generation.terminal_snr_screen import (
    ARM_NAMES,
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


STAGE_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_terminal_snr_confirmation_stage_authorization_v1"
)
STAGE_AUTHORIZATION_ROLE = (
    "user_created_terminal_snr_confirmation_execution_approval"
)
STAGE_AUTHORIZATION_SCOPE = "frozen_terminal_snr_four_arm_confirmation_10k_only"
EXECUTION_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_terminal_snr_confirmation_execution_authorization_v1"
)
EXECUTION_AUTHORIZATION_ROLE = (
    "exact_source_bound_terminal_snr_confirmation_authorization"
)
LAUNCH_RECEIPT_SCHEMA = (
    "cofitok_generation_terminal_snr_confirmation_launch_receipt_v1"
)
LAUNCH_RECEIPT_ROLE = "immutable_terminal_snr_confirmation_launch_receipt"
LIVE_SNAPSHOT_SCHEMA = (
    "cofitok_generation_terminal_snr_confirmation_live_snapshot_v1"
)
LIVE_SNAPSHOT_ROLE = "terminal_snr_confirmation_live_prelaunch_snapshot"
MIN_FREE_BYTES = 120 * 1024**3

STAGE_BOUNDARY = {
    "decision_is_execution_authorization": True,
    "remote_mutation_allowed": True,
    "gpu_execution_allowed": True,
    "training_launch_allowed": False,
    "sampling_launch_allowed": True,
    "evaluation_launch_allowed": True,
    "checkpoint_mutation_allowed": False,
    "scope_limited_to_terminal_snr_confirmation": True,
    "large_capacity_readiness_preparation_allowed": False,
    "large_capacity_readiness_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
EXECUTION_BOUNDARY = copy.deepcopy(STAGE_BOUNDARY)
LAUNCH_BOUNDARY = {
    "terminal_snr_confirmation_execution_authorized": True,
    "frozen_checkpoint_training_allowed": False,
    "checkpoint_mutation_allowed": False,
    "confirmation_samples_per_arm": SAMPLE_COUNT,
    "large_capacity_readiness_preparation_allowed": False,
    "large_capacity_readiness_launch_allowed": False,
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


def terminal_snr_confirmation_execution_lock_path(output_root: str) -> str:
    root = PurePosixPath(_absolute(output_root, "terminal-SNR confirmation root"))
    return (
        root.parent.parent
        / f".{root.parent.name}.{root.name}.terminal_snr_confirmation_execution.lock"
    ).as_posix()


def terminal_snr_confirmation_control_root(output_root: str) -> str:
    root = PurePosixPath(_absolute(output_root, "terminal-SNR confirmation root"))
    return (
        root.parent.parent
        / f".{root.parent.name}.{root.name}.terminal_snr_confirmation_control"
    ).as_posix()


def _expected_output_dirs(output_root: str) -> dict[str, str]:
    root = PurePosixPath(_absolute(output_root, "terminal-SNR confirmation root"))
    return {arm: (root / arm).as_posix() for arm in ARM_NAMES}


def _stage_selection(
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    return {
        "preparation": _identity(
            preparation_identity, "terminal-SNR confirmation preparation"
        ),
        "execution_checkout": _git(
            execution_checkout, "terminal-SNR confirmation checkout"
        ),
        "output_root": _absolute(output_root, "terminal-SNR confirmation root"),
        "frozen_arms": list(ARM_NAMES),
        "checkpoint_step": STOP_STEP,
        "samples_per_arm": SAMPLE_COUNT,
        "sample_steps": SAMPLE_STEPS,
        "sampling_batch_size": SAMPLE_BATCH_SIZE,
        "sampling_seed": SAMPLE_SEED,
    }


def validate_terminal_snr_confirmation_stage_authorization(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation stage authorization")
    expected_selection = _stage_selection(
        preparation_identity, execution_checkout, expected_output_root
    )
    approval = _object(row.get("approval_record"), "terminal-SNR confirmation approval")
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "scope",
            "selection",
            "approval_record",
            "authorization_boundary",
        }
        or set(approval)
        != {"approved_by", "approved_at", "source_instruction"}
        or row.get("schema_version") != STAGE_AUTHORIZATION_SCHEMA
        or row.get("role") != STAGE_AUTHORIZATION_ROLE
        or row.get("status") != "approved"
        or row.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or row.get("selection") != expected_selection
        or row.get("authorization_boundary") != STAGE_BOUNDARY
        or not str(approval.get("approved_by", "")).strip()
        or not str(approval.get("approved_at", "")).strip()
        or not str(approval.get("source_instruction", "")).strip()
    ):
        raise ValueError("terminal-SNR confirmation stage authorization differs")
    return copy.deepcopy(row)


def build_terminal_snr_confirmation_stage_authorization(
    *,
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
        "selection": _stage_selection(
            preparation_identity, execution_checkout, output_root
        ),
        "approval_record": {
            "approved_by": str(approved_by).strip(),
            "approved_at": str(approved_at).strip(),
            "source_instruction": str(source_instruction).strip(),
        },
        "authorization_boundary": copy.deepcopy(STAGE_BOUNDARY),
    }
    return validate_terminal_snr_confirmation_stage_authorization(
        report,
        preparation_identity=preparation_identity,
        execution_checkout=execution_checkout,
        expected_output_root=output_root,
    )


def build_terminal_snr_confirmation_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    prepared = validate_terminal_snr_confirmation_preparation_contract(
        preparation, expected_output_root=output_root
    )
    prep_id = _identity(
        preparation_identity, "terminal-SNR confirmation preparation"
    )
    stage_id = _identity(
        stage_authorization_identity, "terminal-SNR confirmation stage authorization"
    )
    git = _git(execution_checkout, "terminal-SNR confirmation checkout")
    validated_stage = validate_terminal_snr_confirmation_stage_authorization(
        stage_authorization,
        preparation_identity=prep_id,
        execution_checkout=git,
        expected_output_root=output_root,
    )
    report = {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA,
        "role": EXECUTION_AUTHORIZATION_ROLE,
        "status": "authorized",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "preparation": prep_id,
        "stage_authorization": stage_id,
        "validated_stage_authorization": validated_stage,
        "execution_checkout": git,
        "screen_execution_checkout": copy.deepcopy(
            prepared["screen_execution_checkout"]
        ),
        "frozen_arms": copy.deepcopy(prepared["frozen_arms"]),
        "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        "thresholds": copy.deepcopy(prepared["thresholds"]),
        "output_root": _absolute(output_root, "terminal-SNR confirmation root"),
        "authorization_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
    }
    return validate_terminal_snr_confirmation_execution_authorization(
        report,
        preparation=prepared,
        preparation_identity=prep_id,
        expected_execution_checkout=git,
        expected_output_root=output_root,
    )


def validate_terminal_snr_confirmation_execution_authorization(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any] | None = None,
    preparation_identity: Mapping[str, Any],
    expected_execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation execution authorization")
    prep_id = _identity(
        preparation_identity, "terminal-SNR confirmation preparation"
    )
    git = _git(expected_execution_checkout, "terminal-SNR confirmation checkout")
    root = _absolute(expected_output_root, "terminal-SNR confirmation root")
    stage = _object(
        row.get("validated_stage_authorization"),
        "terminal-SNR validated stage authorization",
    )
    validate_terminal_snr_confirmation_stage_authorization(
        stage,
        preparation_identity=prep_id,
        execution_checkout=git,
        expected_output_root=root,
    )
    evaluation = _object(
        row.get("evaluation_contract"), "terminal-SNR confirmation evaluation"
    )
    if preparation is not None:
        prepared = validate_terminal_snr_confirmation_preparation_contract(
            preparation, expected_output_root=root
        )
        if (
            row.get("screen_execution_checkout")
            != prepared.get("screen_execution_checkout")
            or row.get("frozen_arms") != prepared.get("frozen_arms")
            or evaluation != prepared.get("evaluation_contract")
            or row.get("thresholds") != prepared.get("thresholds")
        ):
            raise ValueError(
                "terminal-SNR confirmation authorization differs from preparation"
            )
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "scope",
            "preparation",
            "stage_authorization",
            "validated_stage_authorization",
            "execution_checkout",
            "screen_execution_checkout",
            "frozen_arms",
            "evaluation_contract",
            "thresholds",
            "output_root",
            "authorization_boundary",
        }
        or row.get("schema_version") != EXECUTION_AUTHORIZATION_SCHEMA
        or row.get("role") != EXECUTION_AUTHORIZATION_ROLE
        or row.get("status") != "authorized"
        or row.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or row.get("preparation") != prep_id
        or row.get("execution_checkout") != git
        or row.get("output_root") != root
        or row.get("authorization_boundary") != EXECUTION_BOUNDARY
        or not _identity(
            row.get("stage_authorization"),
            "terminal-SNR confirmation stage authorization",
        )
        or not _git(
            row.get("screen_execution_checkout"),
            "terminal-SNR screen execution checkout",
        )
        or set(_object(row.get("frozen_arms"), "terminal-SNR frozen arms"))
        != set(ARM_NAMES)
        or evaluation.get("arms") != list(ARM_NAMES)
        or int(evaluation.get("checkpoint_step", -1)) != STOP_STEP
        or int(evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or int(evaluation.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(evaluation.get("sampling_batch_size", -1)) != SAMPLE_BATCH_SIZE
        or int(evaluation.get("seed", -1)) != SAMPLE_SEED
        or row.get("thresholds") != SCREEN_THRESHOLDS
    ):
        raise ValueError("terminal-SNR confirmation execution authorization differs")
    return copy.deepcopy(row)


def _load_bound_object(identity_row: Mapping[str, Any], name: str) -> dict[str, Any]:
    expected = _identity(identity_row, name)
    if artifact_identity(expected["path"]) != expected:
        raise ValueError(f"{name} physical identity differs")
    return read_object(expected["path"], name=name)


def validate_terminal_snr_confirmation_execution_authorization_physical(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    expected_execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    """Replay the separately user-created stage authorization from disk."""

    row = validate_terminal_snr_confirmation_execution_authorization(
        report,
        preparation=preparation,
        preparation_identity=preparation_identity,
        expected_execution_checkout=expected_execution_checkout,
        expected_output_root=expected_output_root,
    )
    stage = _load_bound_object(
        row["stage_authorization"],
        "terminal-SNR confirmation stage authorization",
    )
    validated_stage = validate_terminal_snr_confirmation_stage_authorization(
        stage,
        preparation_identity=preparation_identity,
        execution_checkout=expected_execution_checkout,
        expected_output_root=expected_output_root,
    )
    if row["validated_stage_authorization"] != validated_stage:
        raise ValueError(
            "terminal-SNR confirmation embedded stage authorization differs from disk"
        )
    return copy.deepcopy(row)


def validate_terminal_snr_confirmation_live_snapshot(
    report: Mapping[str, Any],
    *,
    expected_output_root: str,
    expected_execution_checkout: Mapping[str, Any],
    expected_execution_lock: str,
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation live snapshot")
    inventory = row.get("gpu_inventory")
    if not isinstance(inventory, list) or len(inventory) != 1:
        raise ValueError("terminal-SNR confirmation requires exactly one target GPU")
    gpu = _object(inventory[0], "terminal-SNR confirmation GPU")
    root = _absolute(expected_output_root, "terminal-SNR confirmation root")
    execution = _git(
        expected_execution_checkout, "terminal-SNR confirmation live checkout"
    )
    sources = _object(
        row.get("source_evidence"), "terminal-SNR confirmation live sources"
    )
    if set(sources) != {
        "preparation",
        "execution_authorization",
        "reference_config",
        "frozen_checkpoint_verification",
    }:
        raise ValueError("terminal-SNR confirmation live source set differs")
    for key in ("preparation", "execution_authorization", "reference_config"):
        _identity(sources.get(key), f"terminal-SNR confirmation live {key}")
    frozen_verification = _object(
        sources.get("frozen_checkpoint_verification"),
        "terminal-SNR confirmation frozen checkpoint verification",
    )
    if set(frozen_verification) != set(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation frozen verification set differs")
    for arm in ARM_NAMES:
        verified = _object(frozen_verification[arm], f"{arm} frozen verification")
        if (
            set(verified) != {"checkpoint", "integrity_manifest", "step"}
            or int(verified.get("step", -1)) != STOP_STEP
        ):
            raise ValueError(f"{arm} frozen verification differs")
        _identity(verified.get("checkpoint"), f"{arm} verified checkpoint")
        _identity(
            verified.get("integrity_manifest"), f"{arm} verified checkpoint integrity"
        )
    runtime_environment = _object(
        row.get("runtime_environment"), "terminal-SNR confirmation runtime environment"
    )
    dataset_provenance = _object(
        row.get("dataset_provenance"), "terminal-SNR confirmation dataset provenance"
    )
    dataset = validate_dataset_provenance(
        dataset_provenance, expected_dataset="imagenet_256"
    )
    exact_lock = terminal_snr_confirmation_execution_lock_path(root)
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "execution_checkout",
            "source_evidence",
            "output_root",
            "execution_lock",
            "gpu_inventory",
            "gpu_compute_processes",
            "conflicting_processes",
            "output_root_absent",
            "execution_lock_free",
            "free_bytes",
            "runtime_environment",
            "runtime_environment_sha256",
            "dataset_provenance",
            "dataset_identity_sha256",
            "captured_at",
        }
        or row.get("schema_version") != LIVE_SNAPSHOT_SCHEMA
        or row.get("role") != LIVE_SNAPSHOT_ROLE
        or row.get("status") != "pass"
        or row.get("execution_checkout") != execution
        or row.get("output_root") != root
        or row.get("execution_lock") != exact_lock
        or _absolute(expected_execution_lock, "terminal-SNR confirmation lock")
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
        or runtime_environment_sha256(runtime_environment)
        != row.get("runtime_environment_sha256")
        or dataset.get("identity_sha256") != row.get("dataset_identity_sha256")
    ):
        raise ValueError("terminal-SNR confirmation live prelaunch snapshot differs")
    return copy.deepcopy(row)


def _validate_storage(report: Mapping[str, Any], *, output_root: str) -> dict[str, Any]:
    filesystem = _object(report.get("filesystem"), "terminal-SNR confirmation filesystem")
    plan = _object(report.get("plan"), "terminal-SNR confirmation storage plan")
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
        raise ValueError("terminal-SNR confirmation storage evidence differs")
    return {
        "filesystem": copy.deepcopy(filesystem),
        "plan": copy.deepcopy(plan),
        "headroom_bytes": int(report["headroom_bytes"]),
    }


def _validate_storage_summary(
    report: Mapping[str, Any], *, output_root: str
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation storage summary")
    filesystem = _object(
        row.get("filesystem"), "terminal-SNR confirmation filesystem"
    )
    plan = _object(row.get("plan"), "terminal-SNR confirmation storage plan")
    filesystem_path = PurePosixPath(str(filesystem.get("path", "")))
    output_path = PurePosixPath(output_root)
    if (
        set(row) != {"filesystem", "plan", "headroom_bytes"}
        or int(row.get("headroom_bytes", -1)) < 0
        or int(filesystem.get("free_bytes", 0)) < MIN_FREE_BYTES
        or int(plan.get("sample_count", -1)) < SAMPLE_COUNT * len(ARM_NAMES)
        or (filesystem_path != output_path and filesystem_path not in output_path.parents)
    ):
        raise ValueError("terminal-SNR confirmation storage summary differs")
    return copy.deepcopy(row)


def build_terminal_snr_confirmation_launch_receipt(
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
    prepared = validate_terminal_snr_confirmation_preparation_contract(
        preparation, expected_output_root=output_root
    )
    prep_id = _identity(
        preparation_identity, "terminal-SNR confirmation preparation"
    )
    authorization_id = _identity(
        execution_authorization_identity,
        "terminal-SNR confirmation execution authorization",
    )
    storage_id = _identity(
        storage_capacity_identity, "terminal-SNR confirmation storage"
    )
    live_id = _identity(live_snapshot_identity, "terminal-SNR confirmation snapshot")
    git = _git(execution_checkout, "terminal-SNR confirmation launch checkout")
    authorization = validate_terminal_snr_confirmation_execution_authorization(
        execution_authorization,
        preparation=prepared,
        preparation_identity=prep_id,
        expected_execution_checkout=git,
        expected_output_root=output_root,
    )
    if set(output_dirs) != set(ARM_NAMES):
        raise ValueError("terminal-SNR confirmation output arm set differs")
    root = _absolute(output_root, "terminal-SNR confirmation root")
    normalized_dirs = {
        arm: _absolute(output_dirs[arm], f"{arm} confirmation output")
        for arm in ARM_NAMES
    }
    if normalized_dirs != _expected_output_dirs(root):
        raise ValueError("terminal-SNR confirmation output directories differ")
    exact_lock = terminal_snr_confirmation_execution_lock_path(root)
    if _absolute(execution_lock, "terminal-SNR confirmation lock") != exact_lock:
        raise ValueError("terminal-SNR confirmation execution lock differs")
    storage = _validate_storage(storage_capacity, output_root=root)
    live = validate_terminal_snr_confirmation_live_snapshot(
        live_snapshot,
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=exact_lock,
    )
    if evaluation_state_absent_at_launch is not True:
        raise ValueError("terminal-SNR confirmation requires absent evaluation state")
    report = {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "pass",
        "stage": "terminal_snr_confirmation",
        "execution_checkout": git,
        "screen_execution_checkout": copy.deepcopy(
            prepared["screen_execution_checkout"]
        ),
        "source_evidence": {
            "preparation": prep_id,
            "execution_authorization": authorization_id,
            "storage_capacity": storage_id,
            "live_snapshot": live_id,
            "terminal_snr_screen_result": copy.deepcopy(
                prepared["source_evidence"]["terminal_snr_screen_result"]
            ),
            "terminal_snr_screen_result_validation": copy.deepcopy(
                prepared["source_evidence"][
                    "terminal_snr_screen_result_validation"
                ]
            ),
            "terminal_snr_screen_arm_validations": copy.deepcopy(
                prepared["source_evidence"][
                    "terminal_snr_screen_arm_validations"
                ]
            ),
        },
        "authorization": authorization,
        "frozen_arms": copy.deepcopy(prepared["frozen_arms"]),
        "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        "thresholds": copy.deepcopy(prepared["thresholds"]),
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
    return validate_terminal_snr_confirmation_launch_receipt_contract(
        report, expected_execution_checkout=git
    )


def validate_terminal_snr_confirmation_launch_receipt_contract(
    report: Mapping[str, Any],
    *,
    expected_execution_checkout: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR confirmation launch receipt")
    git = _git(expected_execution_checkout, "terminal-SNR confirmation checkout")
    root = _absolute(row.get("output_root"), "terminal-SNR confirmation root")
    exact_lock = terminal_snr_confirmation_execution_lock_path(root)
    live = validate_terminal_snr_confirmation_live_snapshot(
        _object(row.get("live_snapshot"), "terminal-SNR confirmation live snapshot"),
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=exact_lock,
    )
    sources = _object(
        row.get("source_evidence"), "terminal-SNR confirmation launch sources"
    )
    if set(sources) != {
        "preparation",
        "execution_authorization",
        "storage_capacity",
        "live_snapshot",
        "terminal_snr_screen_result",
        "terminal_snr_screen_result_validation",
        "terminal_snr_screen_arm_validations",
    }:
        raise ValueError("terminal-SNR confirmation launch source set differs")
    prep_id = _identity(sources.get("preparation"), "terminal-SNR preparation")
    authorization = validate_terminal_snr_confirmation_execution_authorization(
        _object(row.get("authorization"), "terminal-SNR confirmation authorization"),
        preparation_identity=prep_id,
        expected_execution_checkout=git,
        expected_output_root=root,
    )
    evaluation = _object(
        row.get("evaluation_contract"), "terminal-SNR confirmation evaluation"
    )
    screen_arm_ids = _object(
        sources.get("terminal_snr_screen_arm_validations"),
        "terminal-SNR screen arm validations",
    )
    storage = _validate_storage_summary(
        _object(row.get("storage_capacity"), "terminal-SNR storage capacity"),
        output_root=root,
    )
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "stage",
            "execution_checkout",
            "screen_execution_checkout",
            "source_evidence",
            "authorization",
            "frozen_arms",
            "evaluation_contract",
            "thresholds",
            "storage_capacity",
            "live_snapshot",
            "runtime_environment_sha256",
            "dataset_identity_sha256",
            "output_root",
            "output_dirs",
            "execution_lock",
            "evaluation_state_absent_at_launch",
            "authorization_boundary",
        }
        or row.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or row.get("role") != LAUNCH_RECEIPT_ROLE
        or row.get("status") != "pass"
        or row.get("stage") != "terminal_snr_confirmation"
        or row.get("execution_checkout") != git
        or row.get("screen_execution_checkout")
        != authorization.get("screen_execution_checkout")
        or row.get("evaluation_state_absent_at_launch") is not True
        or row.get("authorization_boundary") != LAUNCH_BOUNDARY
        or row.get("execution_lock") != exact_lock
        or row.get("output_dirs") != _expected_output_dirs(root)
        or set(screen_arm_ids) != set(ARM_NAMES)
        or set(_object(row.get("frozen_arms"), "terminal-SNR frozen arms"))
        != set(ARM_NAMES)
        or row.get("frozen_arms") != authorization.get("frozen_arms")
        or evaluation != authorization.get("evaluation_contract")
        or int(evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or int(evaluation.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(evaluation.get("seed", -1)) != SAMPLE_SEED
        or row.get("thresholds") != authorization.get("thresholds")
        or row.get("thresholds") != SCREEN_THRESHOLDS
        or row.get("storage_capacity") != storage
        or not _identity(
            sources.get("execution_authorization"),
            "terminal-SNR execution authorization",
        )
        or not _identity(
            sources.get("storage_capacity"), "terminal-SNR storage capacity"
        )
        or not _identity(sources.get("live_snapshot"), "terminal-SNR live snapshot")
        or not _identity(
            sources.get("terminal_snr_screen_result"),
            "terminal-SNR screen result",
        )
        or not _identity(
            sources.get("terminal_snr_screen_result_validation"),
            "terminal-SNR screen result validation",
        )
        or not _hex(row.get("runtime_environment_sha256"), 64)
        or not _hex(row.get("dataset_identity_sha256"), 64)
        or live.get("runtime_environment_sha256")
        != row.get("runtime_environment_sha256")
        or live.get("dataset_identity_sha256")
        != row.get("dataset_identity_sha256")
    ):
        raise ValueError("terminal-SNR confirmation launch receipt contract differs")
    for arm in ARM_NAMES:
        _identity(screen_arm_ids[arm], f"{arm} screen validation")
    return copy.deepcopy(row)


def validate_terminal_snr_confirmation_launch_receipt_physical(
    report: Mapping[str, Any],
    *,
    expected_execution_checkout: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay every confirmation launch source from its immutable identity."""

    row = validate_terminal_snr_confirmation_launch_receipt_contract(
        report, expected_execution_checkout=expected_execution_checkout
    )
    sources = _object(
        row["source_evidence"], "terminal-SNR confirmation launch sources"
    )
    root = str(row["output_root"])
    execution = _git(
        expected_execution_checkout, "terminal-SNR confirmation launch checkout"
    )

    preparation = _load_bound_object(
        sources["preparation"], "terminal-SNR confirmation preparation"
    )
    prepared = validate_terminal_snr_confirmation_preparation_contract(
        preparation, expected_output_root=root
    )
    if (
        row["screen_execution_checkout"] != prepared["screen_execution_checkout"]
        or row["frozen_arms"] != prepared["frozen_arms"]
        or row["evaluation_contract"] != prepared["evaluation_contract"]
        or row["thresholds"] != prepared["thresholds"]
    ):
        raise ValueError("terminal-SNR confirmation launch differs from preparation")

    authorization = _load_bound_object(
        sources["execution_authorization"],
        "terminal-SNR confirmation execution authorization",
    )
    validated_authorization = (
        validate_terminal_snr_confirmation_execution_authorization_physical(
            authorization,
            preparation=prepared,
            preparation_identity=sources["preparation"],
            expected_execution_checkout=execution,
            expected_output_root=root,
        )
    )
    if row["authorization"] != validated_authorization:
        raise ValueError(
            "terminal-SNR confirmation embedded authorization differs from disk"
        )

    storage_report = _load_bound_object(
        sources["storage_capacity"], "terminal-SNR confirmation storage capacity"
    )
    if row["storage_capacity"] != _validate_storage(
        storage_report, output_root=root
    ):
        raise ValueError("terminal-SNR confirmation storage differs from disk")

    live_report = _load_bound_object(
        sources["live_snapshot"], "terminal-SNR confirmation live snapshot"
    )
    validated_live = validate_terminal_snr_confirmation_live_snapshot(
        live_report,
        expected_output_root=root,
        expected_execution_checkout=execution,
        expected_execution_lock=row["execution_lock"],
    )
    if row["live_snapshot"] != validated_live:
        raise ValueError("terminal-SNR confirmation live snapshot differs from disk")
    live_sources = _object(
        validated_live["source_evidence"], "terminal-SNR confirmation live sources"
    )
    if (
        live_sources["preparation"] != sources["preparation"]
        or live_sources["execution_authorization"]
        != sources["execution_authorization"]
        or live_sources["reference_config"]
        != prepared["frozen_arms"]["control_cofitok"]["config"]
    ):
        raise ValueError("terminal-SNR confirmation live sources differ from launch")
    frozen_verification = _object(
        live_sources["frozen_checkpoint_verification"],
        "terminal-SNR confirmation frozen checkpoint verification",
    )
    for arm in ARM_NAMES:
        frozen = _object(prepared["frozen_arms"][arm], f"{arm} frozen arm")
        checkpoint = _object(frozen["checkpoint"], f"{arm} frozen checkpoint")
        verified = _object(frozen_verification[arm], f"{arm} frozen verification")
        expected_checkpoint_identity = {
            "path": checkpoint["path"],
            "bytes": checkpoint["bytes"],
            "sha256": checkpoint["sha256"],
        }
        if (
            verified["checkpoint"] != expected_checkpoint_identity
            or verified["integrity_manifest"] != checkpoint["integrity_manifest"]
            or int(verified["step"]) != STOP_STEP
        ):
            raise ValueError(f"{arm} frozen checkpoint differs from live verification")

    screen_result = _load_bound_object(
        sources["terminal_snr_screen_result"], "terminal-SNR screen result"
    )
    validated_screen_result = validate_terminal_snr_screen_result_contract(
        screen_result
    )
    if replay_terminal_snr_screen_result(validated_screen_result) != validated_screen_result:
        raise ValueError("terminal-SNR screen result differs from physical replay")
    if (
        validated_screen_result.get("screen_pass") is not True
        or validated_screen_result.get("failed_checks") != []
        or sources["terminal_snr_screen_result"]
        != prepared["source_evidence"]["terminal_snr_screen_result"]
    ):
        raise ValueError("terminal-SNR confirmation does not bind a passing screen")

    screen_result_validation = _load_bound_object(
        sources["terminal_snr_screen_result_validation"],
        "terminal-SNR screen result validation",
    )
    validated_screen_receipt = validate_terminal_snr_screen_validation_receipt(
        screen_result_validation,
        result=validated_screen_result,
        result_identity=sources["terminal_snr_screen_result"],
        validator_git=_git(
            screen_result_validation.get("validator_git"),
            "terminal-SNR screen result validator",
        ),
    )
    if (
        validated_screen_receipt.get("screen_pass") is not True
        or validated_screen_receipt.get("failed_checks") != []
        or sources["terminal_snr_screen_result_validation"]
        != prepared["source_evidence"]["terminal_snr_screen_result_validation"]
    ):
        raise ValueError("terminal-SNR screen validation receipt differs")

    screen_launch_identity = prepared["source_evidence"][
        "terminal_snr_screen_launch_receipt"
    ]
    screen_launch = _load_bound_object(
        screen_launch_identity, "terminal-SNR screen launch receipt"
    )
    validate_terminal_snr_screen_launch_receipt_physical(
        screen_launch,
        expected_execution_checkout=prepared["screen_execution_checkout"],
    )
    result_sources = _object(
        validated_screen_result["source_evidence"], "terminal-SNR screen result sources"
    )
    if result_sources["launch_receipt"] != screen_launch_identity:
        raise ValueError("terminal-SNR screen result binds another launch receipt")

    screen_arm_ids = _object(
        sources["terminal_snr_screen_arm_validations"],
        "terminal-SNR confirmation screen arm identities",
    )
    result_arm_ids = _object(
        result_sources["arm_validations"], "terminal-SNR screen result arms"
    )
    for arm in ARM_NAMES:
        if (
            screen_arm_ids[arm]
            != prepared["source_evidence"]["terminal_snr_screen_arm_validations"][arm]
            or screen_arm_ids[arm] != result_arm_ids[arm]
        ):
            raise ValueError(f"{arm} screen validation identity differs")
        screen_arm = _load_bound_object(
            screen_arm_ids[arm], f"{arm} terminal-SNR screen validation"
        )
        validated_screen_arm = validate_terminal_snr_screen_arm_validation(screen_arm)
        if (
            validated_screen_arm.get("arm") != arm
            or validated_screen_arm.get("execution_git")
            != prepared["screen_execution_checkout"]
            or _object(
                _object(validated_screen_arm["training"], f"{arm} screen training")[
                    "checkpoint"
                ],
                f"{arm} screen checkpoint",
            )
            != prepared["frozen_arms"][arm]["checkpoint"]
        ):
            raise ValueError(f"{arm} screen validation differs from frozen selection")
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
    "build_terminal_snr_confirmation_execution_authorization",
    "build_terminal_snr_confirmation_launch_receipt",
    "build_terminal_snr_confirmation_stage_authorization",
    "terminal_snr_confirmation_control_root",
    "terminal_snr_confirmation_execution_lock_path",
    "validate_terminal_snr_confirmation_execution_authorization",
    "validate_terminal_snr_confirmation_execution_authorization_physical",
    "validate_terminal_snr_confirmation_launch_receipt_contract",
    "validate_terminal_snr_confirmation_launch_receipt_physical",
    "validate_terminal_snr_confirmation_live_snapshot",
    "validate_terminal_snr_confirmation_stage_authorization",
]
