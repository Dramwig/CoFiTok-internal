"""Exact authorization and launch contracts for the terminal-SNR screen."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    identity as artifact_identity,
    read_object,
)
from cofitok.generation.terminal_snr_screen import (
    ARM_NAMES,
    CONFIGURED_STEPS,
    EFFECTIVE_BATCH,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    SAMPLE_STEPS,
    SCREEN_THRESHOLDS,
    STOP_STEP,
    validate_terminal_snr_screen_preparation_contract,
)


STAGE_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_terminal_snr_screen_stage_authorization_v1"
)
STAGE_AUTHORIZATION_ROLE = "user_created_terminal_snr_screen_execution_approval"
STAGE_AUTHORIZATION_SCOPE = "fresh_four_arm_terminal_snr_screen_10k_execution_only"
EXECUTION_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_terminal_snr_screen_execution_authorization_v1"
)
EXECUTION_AUTHORIZATION_ROLE = (
    "exact_source_bound_terminal_snr_screen_authorization"
)
LAUNCH_RECEIPT_SCHEMA = (
    "cofitok_generation_terminal_snr_screen_launch_receipt_v1"
)
LAUNCH_RECEIPT_ROLE = "immutable_terminal_snr_screen_launch_receipt"
LIVE_SNAPSHOT_SCHEMA = "cofitok_generation_terminal_snr_screen_live_snapshot_v1"
LIVE_SNAPSHOT_ROLE = "terminal_snr_screen_live_prelaunch_snapshot"
MIN_FREE_BYTES = 120 * 1024**3

STAGE_BOUNDARY = {
    "decision_is_execution_authorization": True,
    "remote_mutation_allowed": True,
    "gpu_execution_allowed": True,
    "training_launch_allowed": True,
    "sampling_launch_allowed": True,
    "evaluation_launch_allowed": True,
    "scope_limited_to_terminal_snr_screen": True,
    "intentional_stop_step_required": STOP_STEP,
    "configured_100k_completion_allowed": False,
    "frozen_confirmation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
EXECUTION_BOUNDARY = copy.deepcopy(STAGE_BOUNDARY)
LAUNCH_BOUNDARY = {
    "terminal_snr_screen_execution_authorized": True,
    "training_must_stop_at_step": STOP_STEP,
    "screen_samples_per_arm": SAMPLE_COUNT,
    "configured_100k_completion_allowed": False,
    "frozen_confirmation_allowed": False,
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
        or not (
            Path(row["path"]).is_absolute()
            or PurePosixPath(row["path"]).is_absolute()
        )
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
    if not text or not (
        Path(text).is_absolute() or PurePosixPath(text).is_absolute()
    ):
        raise ValueError(f"{name} must be absolute")
    return text


def _content_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("bytes", "sha256")}


def _runtime_git(value: Mapping[str, Any]) -> dict[str, Any]:
    git = _git(value, "terminal-SNR runtime checkout")
    return {
        "revision": git["revision"],
        "branch": git["branch"],
        "tracked_dirty": False,
    }


def terminal_snr_screen_execution_lock_path(output_root: str) -> str:
    root = PurePosixPath(_absolute(output_root, "terminal-SNR output root"))
    if not root.is_absolute() or not root.name:
        raise ValueError("terminal-SNR output root must be an absolute POSIX path")
    return (root.parent / f".{root.name}.terminal_snr_screen_execution.lock").as_posix()


def terminal_snr_screen_control_root(output_root: str) -> str:
    root = PurePosixPath(_absolute(output_root, "terminal-SNR output root"))
    if not root.name:
        raise ValueError("terminal-SNR output root must not be a filesystem root")
    return (root.parent / f".{root.name}.terminal_snr_screen_control").as_posix()


def terminal_snr_screen_benchmark_root(output_root: str) -> str:
    root = PurePosixPath(_absolute(output_root, "terminal-SNR output root"))
    return (
        root.parent / f".{root.name}.terminal_snr_runtime_benchmarks_v2"
    ).as_posix()


def _expected_run_dirs(output_root: str) -> dict[str, str]:
    root = PurePosixPath(_absolute(output_root, "terminal-SNR output root"))
    return {arm: (root / "training" / arm).as_posix() for arm in ARM_NAMES}


def _path_is_within(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return candidate == parent or parent in candidate.parents


def _stage_selection(
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    return {
        "preparation": _identity(
            preparation_identity, "terminal-SNR screen preparation"
        ),
        "execution_checkout": _git(
            execution_checkout, "terminal-SNR execution checkout"
        ),
        "output_root": _absolute(output_root, "terminal-SNR output root"),
        "fresh_training_arms": list(ARM_NAMES),
        "configured_training_horizon": CONFIGURED_STEPS,
        "intentional_stop_step": STOP_STEP,
        "effective_batch_size": EFFECTIVE_BATCH,
        "screen_samples_per_arm": SAMPLE_COUNT,
        "screen_sample_steps": SAMPLE_STEPS,
        "single_scientific_config_field": (
            "diffusion.cosine_endpoint_fraction"
        ),
    }


def validate_terminal_snr_screen_stage_authorization(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR stage authorization")
    approval = _object(row.get("approval_record"), "terminal-SNR approval record")
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
        or set(approval) != {"approved_by", "approved_at", "source_instruction"}
        or row.get("schema_version") != STAGE_AUTHORIZATION_SCHEMA
        or row.get("role") != STAGE_AUTHORIZATION_ROLE
        or row.get("status") != "approved"
        or row.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or row.get("selection")
        != _stage_selection(
            preparation_identity, execution_checkout, expected_output_root
        )
        or row.get("authorization_boundary") != STAGE_BOUNDARY
        or not str(approval.get("approved_by", "")).strip()
        or not str(approval.get("approved_at", "")).strip()
        or not str(approval.get("source_instruction", "")).strip()
    ):
        raise ValueError("terminal-SNR screen stage authorization differs")
    return copy.deepcopy(row)


def build_terminal_snr_screen_stage_authorization(
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
    return validate_terminal_snr_screen_stage_authorization(
        report,
        preparation_identity=preparation_identity,
        execution_checkout=execution_checkout,
        expected_output_root=output_root,
    )


def build_terminal_snr_screen_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
    output_root: str,
) -> dict[str, Any]:
    prepared = validate_terminal_snr_screen_preparation_contract(
        preparation, expected_output_root=output_root
    )
    prep_id = _identity(preparation_identity, "terminal-SNR preparation")
    stage_id = _identity(stage_authorization_identity, "terminal-SNR stage approval")
    git = _git(execution_checkout, "terminal-SNR execution checkout")
    validated_stage = validate_terminal_snr_screen_stage_authorization(
        stage_authorization,
        preparation_identity=prep_id,
        execution_checkout=git,
        expected_output_root=output_root,
    )
    if set(config_identities) != set(ARM_NAMES):
        raise ValueError("terminal-SNR execution config set differs")
    configs = {
        arm: _identity(config_identities[arm], f"{arm} execution config")
        for arm in ARM_NAMES
    }
    prepared_configs = _object(prepared.get("configs"), "prepared terminal configs")
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
        "thresholds": copy.deepcopy(prepared["thresholds"]),
        "output_root": _absolute(output_root, "terminal-SNR output root"),
        "authorization_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
    }


def validate_terminal_snr_screen_execution_authorization(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    expected_execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR execution authorization")
    prep_id = _identity(preparation_identity, "terminal-SNR preparation")
    git = _git(expected_execution_checkout, "terminal-SNR execution checkout")
    root = _absolute(expected_output_root, "terminal-SNR output root")
    stage_id = _identity(
        row.get("stage_authorization"), "terminal-SNR stage authorization"
    )
    del stage_id
    stage = validate_terminal_snr_screen_stage_authorization(
        _object(
            row.get("validated_stage_authorization"),
            "validated terminal-SNR stage authorization",
        ),
        preparation_identity=prep_id,
        execution_checkout=git,
        expected_output_root=root,
    )
    configs = _object(row.get("configs"), "terminal-SNR execution configs")
    selection = _object(row.get("selection"), "terminal-SNR execution selection")
    evaluation = _object(
        row.get("evaluation_contract"), "terminal-SNR execution evaluation"
    )
    if set(configs) == set(ARM_NAMES):
        configs = {
            arm: _identity(configs[arm], f"{arm} execution config")
            for arm in ARM_NAMES
        }
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
            "configs",
            "selection",
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
        or row.get("validated_stage_authorization") != stage
        or row.get("execution_checkout") != git
        or row.get("output_root") != root
        or row.get("authorization_boundary") != EXECUTION_BOUNDARY
        or set(configs) != set(ARM_NAMES)
        or row.get("configs") != configs
        or selection.get("output_root") != root
        or selection.get("fresh_training_arms") != list(ARM_NAMES)
        or selection.get("fresh_initialization_required_for_all_arms") is not True
        or selection.get("single_scientific_config_field")
        != "diffusion.cosine_endpoint_fraction"
        or int(selection.get("configured_training_horizon", -1))
        != CONFIGURED_STEPS
        or int(selection.get("stop_after_step", -1)) != STOP_STEP
        or int(selection.get("effective_batch_size", -1)) != EFFECTIVE_BATCH
        or int(selection.get("images_seen_per_arm", -1))
        != STOP_STEP * EFFECTIVE_BATCH
        or evaluation.get("arms") != list(ARM_NAMES)
        or int(evaluation.get("checkpoint_step", -1)) != STOP_STEP
        or int(evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or int(evaluation.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(evaluation.get("seed", -1)) != SAMPLE_SEED
        or evaluation.get("weights") != "ema"
        or evaluation.get("fixed_random_stream_across_arms") is not True
        or evaluation.get("fid_is_precision_recall_required") is not True
        or evaluation.get("class_fidelity_required") is not True
        or evaluation.get("checkpoint_evaluation_required_for_all_arms") is not True
        or evaluation.get("rollout_required_for_all_arms") is not True
        or row.get("thresholds") != SCREEN_THRESHOLDS
    ):
        raise ValueError("terminal-SNR execution authorization contract differs")
    return copy.deepcopy(row)


def validate_terminal_snr_screen_live_snapshot(
    report: Mapping[str, Any],
    *,
    expected_output_root: str,
    expected_execution_checkout: Mapping[str, Any],
    expected_execution_lock: str,
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR live snapshot")
    gpu = row.get("gpu_inventory")
    if not isinstance(gpu, list) or len(gpu) != 1:
        raise ValueError("terminal-SNR screen requires exactly one target GPU")
    device = _object(gpu[0], "terminal-SNR GPU inventory")
    root = _absolute(expected_output_root, "terminal-SNR output root")
    git = _git(expected_execution_checkout, "terminal-SNR live checkout")
    exact_lock = terminal_snr_screen_execution_lock_path(root)
    if (
        row.get("schema_version") != LIVE_SNAPSHOT_SCHEMA
        or row.get("role") != LIVE_SNAPSHOT_ROLE
        or row.get("status") != "pass"
        or row.get("execution_checkout") != git
        or row.get("output_root") != root
        or row.get("execution_lock") != exact_lock
        or _absolute(expected_execution_lock, "terminal-SNR execution lock")
        != exact_lock
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
        raise ValueError("terminal-SNR live prelaunch snapshot differs")
    return copy.deepcopy(row)


def validate_terminal_snr_screen_runtime_selection(
    report: Mapping[str, Any],
    *,
    execution_git: Mapping[str, Any],
    endpoint_config_identities: Mapping[str, Mapping[str, Any]],
    endpoint_run_dirs: list[str],
    benchmark_root: str,
) -> dict[str, Any]:
    selected = _object(report.get("selected"), "terminal-SNR runtime selection")
    lock = _object(report.get("selection_lock"), "terminal-SNR runtime lock")
    micro = int(selected.get("micro_batch_size", -1))
    accumulation = int(selected.get("gradient_accumulation_steps", -1))
    expected_config_sha = {
        method: endpoint_config_identities[method]["sha256"]
        for method in ("cofitok", "dense_identity")
    }
    if (
        report.get("status") != "selected"
        or report.get("git_revision") != execution_git["revision"]
        or report.get("config_sha256") != expected_config_sha
        or report.get("benchmark_root") != benchmark_root
        or micro < 1
        or accumulation < 1
        or micro * accumulation != EFFECTIVE_BATCH
        or lock.get("training_run_dirs") != endpoint_run_dirs
        or int(lock.get("training_target_steps", -1)) != CONFIGURED_STEPS
        or int(lock.get("expected_effective_batch_size", -1)) != EFFECTIVE_BATCH
        or lock.get("git") != _runtime_git(execution_git)
        or lock.get("config_sha256") != expected_config_sha
        or lock.get("benchmark_root") != benchmark_root
        or not _hex(report.get("runtime_environment_sha256"), length=64)
        or not _hex(report.get("dataset_identity_sha256"), length=64)
    ):
        raise ValueError("terminal-SNR runtime selection differs")
    return {
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": EFFECTIVE_BATCH,
        "runtime_environment_sha256": report["runtime_environment_sha256"],
        "dataset_identity_sha256": report["dataset_identity_sha256"],
        "selection_sha256": report.get("selection_sha256"),
    }


def _validate_storage(report: Mapping[str, Any], *, output_root: str) -> dict[str, Any]:
    filesystem = _object(report.get("filesystem"), "terminal-SNR storage filesystem")
    plan = _object(report.get("plan"), "terminal-SNR storage plan")
    output_parent = PurePosixPath(output_root).parent.as_posix()
    if (
        report.get("status") != "pass"
        or int(report.get("schema_version", -1)) != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or int(report.get("headroom_bytes", -1)) < 0
        or int(filesystem.get("free_bytes", 0)) < MIN_FREE_BYTES
        or int(plan.get("sample_count", -1)) < SAMPLE_COUNT * len(ARM_NAMES)
        or int(plan.get("checkpoint_count", -1)) < len(ARM_NAMES) * 2
        or not _path_is_within(output_parent, str(filesystem.get("path", "")))
    ):
        raise ValueError("terminal-SNR storage evidence differs")
    return {
        "filesystem": copy.deepcopy(filesystem),
        "plan": copy.deepcopy(plan),
        "headroom_bytes": int(report["headroom_bytes"]),
    }


def _validate_storage_summary(
    report: Mapping[str, Any], *, output_root: str
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR storage summary")
    filesystem = _object(row.get("filesystem"), "terminal-SNR storage filesystem")
    plan = _object(row.get("plan"), "terminal-SNR storage plan")
    output_parent = PurePosixPath(output_root).parent.as_posix()
    if (
        set(row) != {"filesystem", "plan", "headroom_bytes"}
        or int(row.get("headroom_bytes", -1)) < 0
        or int(filesystem.get("free_bytes", 0)) < MIN_FREE_BYTES
        or int(plan.get("sample_count", -1)) < SAMPLE_COUNT * len(ARM_NAMES)
        or int(plan.get("checkpoint_count", -1)) < len(ARM_NAMES) * 2
        or not _path_is_within(output_parent, str(filesystem.get("path", "")))
    ):
        raise ValueError("terminal-SNR storage summary differs")
    return copy.deepcopy(row)


def build_terminal_snr_screen_launch_receipt(
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
    root = _absolute(output_root, "terminal-SNR output root")
    prepared = validate_terminal_snr_screen_preparation_contract(
        preparation, expected_output_root=root
    )
    prep_id = _identity(preparation_identity, "terminal-SNR preparation")
    auth_id = _identity(execution_authorization_identity, "terminal-SNR authorization")
    runtime_id = _identity(runtime_selection_identity, "terminal-SNR runtime")
    storage_id = _identity(storage_capacity_identity, "terminal-SNR storage")
    live_id = _identity(live_snapshot_identity, "terminal-SNR live snapshot")
    git = _git(execution_checkout, "terminal-SNR launch checkout")
    authorization = validate_terminal_snr_screen_execution_authorization(
        execution_authorization,
        preparation_identity=prep_id,
        expected_execution_checkout=git,
        expected_output_root=root,
    )
    if set(config_identities) != set(ARM_NAMES) or set(run_dirs) != set(ARM_NAMES):
        raise ValueError("terminal-SNR launch arm set differs")
    configs = {
        arm: _identity(config_identities[arm], f"{arm} launch config")
        for arm in ARM_NAMES
    }
    if configs != authorization["configs"]:
        raise ValueError("terminal-SNR launch configs differ from authorization")
    normalized_runs = {
        arm: _absolute(run_dirs[arm], f"{arm} run directory")
        for arm in ARM_NAMES
    }
    if normalized_runs != _expected_run_dirs(root):
        raise ValueError("terminal-SNR run directories differ from fixed layout")
    exact_lock = terminal_snr_screen_execution_lock_path(root)
    if _absolute(execution_lock, "terminal-SNR execution lock") != exact_lock:
        raise ValueError("terminal-SNR execution lock is not the exact sibling lock")
    endpoint_configs = {
        "cofitok": configs["endpoint0975_cofitok"],
        "dense_identity": configs["endpoint0975_dense_identity"],
    }
    endpoint_runs = [
        normalized_runs["endpoint0975_cofitok"],
        normalized_runs["endpoint0975_dense_identity"],
    ]
    normalized_benchmark = _absolute(benchmark_root, "terminal-SNR benchmark root")
    runtime = validate_terminal_snr_screen_runtime_selection(
        runtime_selection,
        execution_git=git,
        endpoint_config_identities=endpoint_configs,
        endpoint_run_dirs=endpoint_runs,
        benchmark_root=normalized_benchmark,
    )
    if _path_is_within(normalized_benchmark, root):
        raise ValueError("terminal-SNR benchmarks must stay outside result root")
    storage = _validate_storage(storage_capacity, output_root=root)
    live = validate_terminal_snr_screen_live_snapshot(
        live_snapshot,
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=exact_lock,
    )
    if live["runtime_environment_sha256"] != runtime["runtime_environment_sha256"]:
        raise ValueError("terminal-SNR launch runtime environment changed")
    if live["dataset_identity_sha256"] != runtime["dataset_identity_sha256"]:
        raise ValueError("terminal-SNR launch dataset identity changed")
    if training_state_absent_at_launch is not True:
        raise ValueError("terminal-SNR screen requires absent initial training state")
    return {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "pass",
        "stage": "terminal_snr_screen",
        "execution_checkout": git,
        "source_evidence": {
            "preparation": prep_id,
            "execution_authorization": auth_id,
            "runtime_selection": runtime_id,
            "storage_capacity": storage_id,
            "live_snapshot": live_id,
            "configs": configs,
        },
        "authorization": authorization,
        "selection": copy.deepcopy(prepared["selection"]),
        "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        "thresholds": copy.deepcopy(prepared["thresholds"]),
        "runtime_selection": runtime,
        "runtime_environment_sha256": runtime["runtime_environment_sha256"],
        "dataset_identity_sha256": runtime["dataset_identity_sha256"],
        "storage_capacity": storage,
        "live_snapshot": live,
        "output_root": root,
        "execution_lock": exact_lock,
        "run_dirs": normalized_runs,
        "benchmark_root": normalized_benchmark,
        "training_state_absent_at_launch": True,
        "authorization_boundary": copy.deepcopy(LAUNCH_BOUNDARY),
    }


def validate_terminal_snr_screen_launch_receipt_contract(
    report: Mapping[str, Any],
    *,
    expected_execution_checkout: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR launch receipt")
    git = _git(expected_execution_checkout, "terminal-SNR launch checkout")
    root = _absolute(row.get("output_root"), "terminal-SNR output root")
    exact_lock = terminal_snr_screen_execution_lock_path(root)
    runtime = _object(row.get("runtime_selection"), "terminal-SNR runtime")
    sources = _object(row.get("source_evidence"), "terminal-SNR launch sources")
    if set(sources) != {
        "preparation",
        "execution_authorization",
        "runtime_selection",
        "storage_capacity",
        "live_snapshot",
        "configs",
    }:
        raise ValueError("terminal-SNR launch source set differs")
    preparation_id = _identity(sources.get("preparation"), "terminal-SNR preparation")
    for name in (
        "execution_authorization",
        "runtime_selection",
        "storage_capacity",
        "live_snapshot",
    ):
        _identity(sources.get(name), f"terminal-SNR {name}")
    configs = _object(sources.get("configs"), "terminal-SNR launch configs")
    if set(configs) != set(ARM_NAMES):
        raise ValueError("terminal-SNR launch config set differs")
    configs = {
        arm: _identity(configs[arm], f"{arm} launch config") for arm in ARM_NAMES
    }
    authorization = validate_terminal_snr_screen_execution_authorization(
        _object(row.get("authorization"), "terminal-SNR embedded authorization"),
        preparation_identity=preparation_id,
        expected_execution_checkout=git,
        expected_output_root=root,
    )
    live = validate_terminal_snr_screen_live_snapshot(
        _object(row.get("live_snapshot"), "terminal-SNR live snapshot"),
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=exact_lock,
    )
    benchmark = _absolute(row.get("benchmark_root"), "terminal-SNR benchmark root")
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
            "source_evidence",
            "authorization",
            "selection",
            "evaluation_contract",
            "thresholds",
            "runtime_selection",
            "runtime_environment_sha256",
            "dataset_identity_sha256",
            "storage_capacity",
            "live_snapshot",
            "output_root",
            "execution_lock",
            "run_dirs",
            "benchmark_root",
            "training_state_absent_at_launch",
            "authorization_boundary",
        }
        or row.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or row.get("role") != LAUNCH_RECEIPT_ROLE
        or row.get("status") != "pass"
        or row.get("stage") != "terminal_snr_screen"
        or row.get("execution_checkout") != git
        or row.get("training_state_absent_at_launch") is not True
        or row.get("authorization_boundary") != LAUNCH_BOUNDARY
        or row.get("execution_lock") != exact_lock
        or _object(row.get("run_dirs"), "terminal-SNR run dirs")
        != _expected_run_dirs(root)
        or authorization.get("configs") != configs
        or row.get("selection") != authorization.get("selection")
        or row.get("evaluation_contract")
        != authorization.get("evaluation_contract")
        or row.get("thresholds") != authorization.get("thresholds")
        or row.get("storage_capacity") != storage
        or int(runtime.get("effective_batch_size", -1)) != EFFECTIVE_BATCH
        or int(runtime.get("micro_batch_size", -1)) < 1
        or int(runtime.get("gradient_accumulation_steps", -1)) < 1
        or int(runtime.get("micro_batch_size", -1))
        * int(runtime.get("gradient_accumulation_steps", -1))
        != EFFECTIVE_BATCH
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
        or _path_is_within(benchmark, root)
    ):
        raise ValueError("terminal-SNR launch receipt contract differs")
    return copy.deepcopy(row)


def _load_bound_object(identity_row: Mapping[str, Any], name: str) -> dict[str, Any]:
    expected = _identity(identity_row, name)
    if artifact_identity(expected["path"]) != expected:
        raise ValueError(f"{name} physical identity differs")
    return read_object(expected["path"], name=name)


def validate_terminal_snr_screen_launch_receipt_physical(
    report: Mapping[str, Any],
    *,
    expected_execution_checkout: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay every launch source from its content-addressed disk identity."""

    row = validate_terminal_snr_screen_launch_receipt_contract(
        report, expected_execution_checkout=expected_execution_checkout
    )
    sources = _object(row["source_evidence"], "terminal-SNR launch sources")
    root = str(row["output_root"])
    git = _git(expected_execution_checkout, "terminal-SNR launch checkout")

    preparation = _load_bound_object(
        sources["preparation"], "terminal-SNR preparation"
    )
    prepared = validate_terminal_snr_screen_preparation_contract(
        preparation, expected_output_root=root
    )
    if (
        row["selection"] != prepared["selection"]
        or row["evaluation_contract"] != prepared["evaluation_contract"]
        or row["thresholds"] != prepared["thresholds"]
    ):
        raise ValueError("terminal-SNR launch differs from physical preparation")

    authorization = _load_bound_object(
        sources["execution_authorization"],
        "terminal-SNR execution authorization",
    )
    validated_authorization = validate_terminal_snr_screen_execution_authorization(
        authorization,
        preparation_identity=sources["preparation"],
        expected_execution_checkout=git,
        expected_output_root=root,
    )
    if row["authorization"] != validated_authorization:
        raise ValueError("terminal-SNR embedded authorization differs from disk")
    stage_id = validated_authorization["stage_authorization"]
    stage = _load_bound_object(stage_id, "terminal-SNR stage authorization")
    validated_stage = validate_terminal_snr_screen_stage_authorization(
        stage,
        preparation_identity=sources["preparation"],
        execution_checkout=git,
        expected_output_root=root,
    )
    if validated_authorization["validated_stage_authorization"] != validated_stage:
        raise ValueError("terminal-SNR embedded stage authorization differs from disk")

    config_ids = _object(sources["configs"], "terminal-SNR launch configs")
    for arm in ARM_NAMES:
        expected = _identity(config_ids[arm], f"{arm} launch config")
        if artifact_identity(expected["path"]) != expected:
            raise ValueError(f"{arm} physical config identity differs")

    runtime_report = _load_bound_object(
        sources["runtime_selection"], "terminal-SNR runtime selection"
    )
    expected_runtime = validate_terminal_snr_screen_runtime_selection(
        runtime_report,
        execution_git=git,
        endpoint_config_identities={
            "cofitok": config_ids["endpoint0975_cofitok"],
            "dense_identity": config_ids["endpoint0975_dense_identity"],
        },
        endpoint_run_dirs=[
            row["run_dirs"]["endpoint0975_cofitok"],
            row["run_dirs"]["endpoint0975_dense_identity"],
        ],
        benchmark_root=row["benchmark_root"],
    )
    if row["runtime_selection"] != expected_runtime:
        raise ValueError("terminal-SNR embedded runtime selection differs from disk")

    storage_report = _load_bound_object(
        sources["storage_capacity"], "terminal-SNR storage capacity"
    )
    expected_storage = _validate_storage(storage_report, output_root=root)
    if row["storage_capacity"] != expected_storage:
        raise ValueError("terminal-SNR embedded storage evidence differs from disk")

    live_report = _load_bound_object(
        sources["live_snapshot"], "terminal-SNR live snapshot"
    )
    expected_live = validate_terminal_snr_screen_live_snapshot(
        live_report,
        expected_output_root=root,
        expected_execution_checkout=git,
        expected_execution_lock=row["execution_lock"],
    )
    if row["live_snapshot"] != expected_live:
        raise ValueError("terminal-SNR embedded live snapshot differs from disk")
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
    "build_terminal_snr_screen_execution_authorization",
    "build_terminal_snr_screen_launch_receipt",
    "build_terminal_snr_screen_stage_authorization",
    "terminal_snr_screen_benchmark_root",
    "terminal_snr_screen_control_root",
    "terminal_snr_screen_execution_lock_path",
    "validate_terminal_snr_screen_execution_authorization",
    "validate_terminal_snr_screen_launch_receipt_contract",
    "validate_terminal_snr_screen_launch_receipt_physical",
    "validate_terminal_snr_screen_live_snapshot",
    "validate_terminal_snr_screen_runtime_selection",
    "validate_terminal_snr_screen_stage_authorization",
]
