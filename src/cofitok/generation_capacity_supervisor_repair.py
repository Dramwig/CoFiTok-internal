from __future__ import annotations

import copy
import json
import os
import shlex
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.generation_control_process_migration import (
    APPROVAL_ROLE,
    APPROVAL_SCHEMA_VERSION,
    EXECUTION_ROLE,
    EXECUTION_SCHEMA_VERSION,
    PLAN_ROLE,
    PLAN_SCHEMA_VERSION,
    SUPERVISOR_NAMES,
    SUPERVISOR_SPECS,
    ProcessHandle,
    _barriers,
    _git_state_with_porcelain,
    _launch_process,
    _status_snapshot,
    _terminate_handle,
    _validate_barriers,
    _validate_runtime_barrier,
    _wait_for_waiting_status,
    inspect_linux_lifetime_lock,
)
from cofitok.generation_control_process_snapshot import inspect_linux_process
from cofitok.inference_replay import file_identity
from cofitok.reporting import write_json_report, write_text_report


REPAIR_PLAN_SCHEMA_VERSION = 1
REPAIR_APPROVAL_SCHEMA_VERSION = 1
REPAIR_EXECUTION_SCHEMA_VERSION = 1
REPAIR_PLAN_ROLE = "generation_capacity_supervisor_receipt_repair_plan"
REPAIR_APPROVAL_ROLE = "generation_capacity_supervisor_receipt_repair_approval"
REPAIR_EXECUTION_ROLE = "generation_capacity_supervisor_receipt_repair_execution"

TRAINING_NAME = "capacity_full_300k_training"
POSTEVAL_NAME = "capacity_full_300k_posteval"
FINALIZATION_NAME = "capacity_full_300k_finalization"
DOWNSTREAM_NAMES = (POSTEVAL_NAME, FINALIZATION_NAME)

TRAINING_RECEIPT_ROLE = "capacity_full_300k_training_supervisor_deployment"
POSTEVAL_RECEIPT_ROLE = "capacity_full_300k_posteval_supervisor_deployment"
FINALIZATION_RECEIPT_ROLE = (
    "capacity_full_300k_finalization_supervisor_deployment"
)

REPAIR_SCOPE = {
    "byte_preserving_receipt_archive_required": True,
    "canonical_receipt_rotation_allowed": True,
    "selected_missing_control_process_launch_allowed": list(DOWNSTREAM_NAMES),
    "existing_process_signaling_allowed": False,
    "gpu_query_or_allocation_allowed": False,
    "direct_training_sampling_or_evaluation_launch_allowed": False,
    "formal_checkout_modification_allowed": False,
    "active_quality_bridge_or_capacity_probe_signaling_allowed": False,
    "unrelated_process_signaling_allowed": False,
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_identity(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return identity


def _formal_snapshot(project: Path) -> dict[str, Any]:
    state = _git_state_with_porcelain(project)
    return {
        "path": project.resolve().as_posix(),
        "head": state["revision"],
        "porcelain_count": state["porcelain_count"],
        "porcelain_sha256": state["porcelain_sha256"],
    }


def _core_git(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("path", "revision", "tree", "branch", "tracked_dirty")
    }


def _spec(name: str) -> Mapping[str, Any]:
    matches = [row for row in SUPERVISOR_SPECS if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown capacity supervisor: {name}")
    return matches[0]


def _replace_cli_value(argv: Sequence[str], option: str, value: str) -> list[str]:
    result = list(argv)
    matches = [index for index, item in enumerate(result) if item == option]
    if len(matches) != 1 or matches[0] + 1 >= len(result):
        raise ValueError(f"repair argv must contain exactly one {option}")
    result[matches[0] + 1] = value
    return result


def _lock_record(path: Path) -> dict[str, Any]:
    target = path.resolve()
    if not target.is_file():
        raise ValueError(f"repair lifetime lock is missing: {target}")
    stat = target.stat()
    return {
        "path": target.as_posix(),
        "identity": file_identity(target),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "owner_pid": None,
        "owner_descriptors": [],
        "exclusive_nonblocking_probe": "must_be_available_at_launch",
    }


def _target_row(
    source: Mapping[str, Any],
    *,
    target_project: Path,
) -> dict[str, Any]:
    row = copy.deepcopy(dict(source))
    target = target_project.resolve()
    entrypoint = target / str(row["entrypoint"]["argument"])
    row["runtime"]["cwd"] = target.as_posix()
    row["runtime"]["environment"] = {
        **dict(row["runtime"]["environment"]),
        "PYTHONPATH": os.pathsep.join((target.as_posix(), (target / "src").as_posix())),
    }
    row["entrypoint"]["identity"] = file_identity(entrypoint)
    lock_filename = _spec(str(row["name"])).get("lock_filename")
    if lock_filename is None:
        raise ValueError(f"repair target lacks a lifetime lock: {row['name']}")
    status_path = Path(str(row["status_path"]))
    row["lifetime_lock"] = _lock_record(
        status_path.parent.parent / str(lock_filename)
    )
    return row


def _processes_for_entrypoint(
    entrypoint: str,
    *,
    proc_root: Path = Path("/proc"),
) -> list[int]:
    matches = []
    if not proc_root.is_dir():
        return matches
    for child in proc_root.iterdir():
        if not child.name.isdigit():
            continue
        try:
            argv = [
                part.decode("utf-8", errors="replace")
                for part in (child / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except OSError:
            continue
        if entrypoint in argv:
            matches.append(int(child.name))
    return sorted(matches)


def _validate_receipt_sources(report: Mapping[str, Any], *, label: str) -> None:
    sources = report.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError(f"{label} sources are missing")
    for name, source in sources.items():
        if not isinstance(source, Mapping):
            raise ValueError(f"{label} source is invalid: {name}")
        path = Path(str(source.get("path", "")))
        if not path.is_file():
            if name == "incremental_bundle":
                continue
            raise ValueError(f"{label} source is missing: {name}")
        if file_identity(path) != dict(source):
            raise ValueError(f"{label} source changed: {name}")


def build_supervisor_receipt_repair_plan(
    *,
    migration_plan_path: Path,
    expected_migration_plan_sha256: str,
    migration_approval_path: Path,
    expected_migration_approval_sha256: str,
    migration_execution_path: Path,
    expected_migration_execution_sha256: str,
    target_project: Path,
    expected_target_git: Mapping[str, Any],
    formal_project: Path,
    expected_formal_git: Mapping[str, Any],
    live_training_pid: int,
    training_receipt_path: Path,
    expected_training_receipt_sha256: str,
    posteval_receipt_path: Path,
    expected_posteval_receipt_sha256: str,
    finalization_receipt_path: Path,
    expected_finalization_receipt_sha256: str,
    archive_root: Path,
    inspect_process: Callable[[int], Mapping[str, Any]] = inspect_linux_process,
    process_scanner: Callable[[str], Sequence[int]] = _processes_for_entrypoint,
) -> dict[str, Any]:
    if type(live_training_pid) is not int or live_training_pid <= 0:
        raise ValueError("live training supervisor PID is invalid")
    migration_plan_id = _require_identity(
        migration_plan_path,
        expected_migration_plan_sha256,
        label="failed migration plan",
    )
    migration_approval_id = _require_identity(
        migration_approval_path,
        expected_migration_approval_sha256,
        label="failed migration approval",
    )
    migration_execution_id = _require_identity(
        migration_execution_path,
        expected_migration_execution_sha256,
        label="failed migration execution",
    )
    migration_plan = _read_json(migration_plan_path, label="failed migration plan")
    migration_approval = _read_json(
        migration_approval_path,
        label="failed migration approval",
    )
    migration_execution = _read_json(
        migration_execution_path,
        label="failed migration execution",
    )
    if (
        migration_plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or migration_plan.get("role") != PLAN_ROLE
        or migration_plan.get("selected_process_names") != list(SUPERVISOR_NAMES)
        or migration_approval.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or migration_approval.get("role") != APPROVAL_ROLE
        or migration_approval.get("plan") != migration_plan_id
        or migration_execution.get("schema_version") != EXECUTION_SCHEMA_VERSION
        or migration_execution.get("role") != EXECUTION_ROLE
        or migration_execution.get("status") != "failed_restore_incomplete"
        or migration_execution.get("plan") != migration_plan_id
        or migration_execution.get("approval") != migration_approval_id
        or migration_execution.get("restoration_error", {}).get("detail")
        != "migrated supervisor exited before waiting status: "
        "capacity_full_300k_posteval:1"
    ):
        raise ValueError("failed migration lineage differs")

    target_git = _git_state_with_porcelain(target_project.resolve())
    if _core_git(target_git) != dict(expected_target_git):
        raise ValueError("repair target Git identity differs")
    if target_git["porcelain_count"] != 0:
        raise ValueError("repair target checkout must be fully clean")
    formal_git = _git_state_with_porcelain(formal_project.resolve())
    if _core_git(formal_git) != dict(expected_formal_git):
        raise ValueError("formal checkout Git identity differs")

    old_rows = {
        str(row["name"]): row for row in migration_plan.get("old_processes", [])
    }
    new_rows = {
        str(row["name"]): row for row in migration_plan.get("new_processes", [])
    }
    if set(old_rows) != set(SUPERVISOR_NAMES) or set(new_rows) != set(
        SUPERVISOR_NAMES
    ):
        raise ValueError("failed migration process rows differ")

    training_runtime = dict(inspect_process(live_training_pid))
    training_spec = _spec(TRAINING_NAME)
    _validate_runtime_barrier(training_spec, training_runtime)
    training_status = _status_snapshot(training_spec, runtime=training_runtime)
    training_barriers = _barriers(training_spec, runtime=training_runtime)
    if process_scanner(str(_spec(POSTEVAL_NAME)["entrypoint"])):
        raise ValueError("posteval supervisor process already exists")
    if process_scanner(str(_spec(FINALIZATION_NAME)["entrypoint"])):
        raise ValueError("finalization supervisor process already exists")

    receipt_rows = []
    receipt_arguments = (
        (
            "training",
            training_receipt_path,
            expected_training_receipt_sha256,
            TRAINING_RECEIPT_ROLE,
        ),
        (
            "posteval",
            posteval_receipt_path,
            expected_posteval_receipt_sha256,
            POSTEVAL_RECEIPT_ROLE,
        ),
        (
            "finalization",
            finalization_receipt_path,
            expected_finalization_receipt_sha256,
            FINALIZATION_RECEIPT_ROLE,
        ),
    )
    for name, path, expected_sha, role in receipt_arguments:
        identity = _require_identity(path, expected_sha, label=f"{name} receipt")
        payload = _read_json(path, label=f"{name} receipt")
        if (
            payload.get("schema_version") != 1
            or payload.get("role") != role
            or payload.get("status") != "active"
        ):
            raise ValueError(f"{name} receipt contract differs")
        _validate_receipt_sources(payload, label=f"{name} receipt")
        receipt_rows.append(
            {
                "name": name,
                "canonical": identity,
                "archive_path": (
                    archive_root.resolve()
                    / f"{path.stem}.{identity['sha256']}.json"
                ).as_posix(),
                "role": role,
            }
        )

    launch_rows = {
        name: _target_row(new_rows[name], target_project=target_project)
        for name in DOWNSTREAM_NAMES
    }
    _validate_barriers(tuple(launch_rows.values()))
    return {
        "schema_version": REPAIR_PLAN_SCHEMA_VERSION,
        "role": REPAIR_PLAN_ROLE,
        "status": "ready",
        "detail": "failed_migration_receipts_and_missing_downstream_are_repairable",
        "built_at": _timestamp(),
        "migration": {
            "plan": migration_plan_id,
            "approval": migration_approval_id,
            "execution": migration_execution_id,
        },
        "target_git": target_git,
        "formal_git": formal_git,
        "formal_checkout_snapshot": _formal_snapshot(formal_project),
        "live_training": {
            "runtime": training_runtime,
            "status": training_status,
            "barriers": training_barriers,
        },
        "receipts": receipt_rows,
        "launch_rows": launch_rows,
        "archive_root": archive_root.resolve().as_posix(),
        "scope": dict(REPAIR_SCOPE),
        "effects": {
            "receipts_archived": False,
            "canonical_receipts_rotated": False,
            "processes_launched": False,
            "processes_signaled": False,
            "gpu_queried_or_allocated": False,
            "formal_checkout_modified": False,
        },
    }


def build_supervisor_receipt_repair_approval(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    identity = _require_identity(
        plan_path,
        expected_plan_sha256,
        label="supervisor receipt repair plan",
    )
    plan = _read_json(plan_path, label="supervisor receipt repair plan")
    if (
        plan.get("schema_version") != REPAIR_PLAN_SCHEMA_VERSION
        or plan.get("role") != REPAIR_PLAN_ROLE
        or plan.get("status") != "ready"
        or plan.get("scope") != REPAIR_SCOPE
        or set(plan.get("launch_rows", {})) != set(DOWNSTREAM_NAMES)
        or [row.get("name") for row in plan.get("receipts", [])]
        != ["training", "posteval", "finalization"]
    ):
        raise ValueError("supervisor receipt repair plan contract differs")
    return {
        "schema_version": REPAIR_APPROVAL_SCHEMA_VERSION,
        "role": REPAIR_APPROVAL_ROLE,
        "status": "approved",
        "approved_at": _timestamp(),
        "plan": identity,
        "migration": plan["migration"],
        "target_git": plan["target_git"],
        "formal_git": plan["formal_git"],
        "scope": dict(REPAIR_SCOPE),
    }


def _archive_exact(source: Path, destination: Path) -> dict[str, Any]:
    source_identity = file_identity(source)
    if destination.exists():
        archived = file_identity(destination)
        if (
            archived["bytes"] != source_identity["bytes"]
            or archived["sha256"] != source_identity["sha256"]
        ):
            raise ValueError(f"receipt archive collision: {destination}")
        return archived
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            while True:
                block = input_handle.read(1024 * 1024)
                if not block:
                    break
                output_handle.write(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    archived = file_identity(destination)
    if (
        archived["bytes"] != source_identity["bytes"]
        or archived["sha256"] != source_identity["sha256"]
    ):
        raise ValueError(f"receipt archive verification failed: {destination}")
    return archived


def _write_pointer(path: Path, pid: int) -> dict[str, Any]:
    write_text_report(path, f"{pid}\n")
    return file_identity(path)


def _active_sources(
    receipt: Mapping[str, Any],
    *,
    updates: Mapping[str, Mapping[str, Any]],
    control_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = copy.deepcopy(dict(receipt.get("sources", {})))
    missing = {}
    for name in list(sources):
        path = Path(str(sources[name].get("path", "")))
        if not path.is_file():
            missing[name] = sources.pop(name)
    sources.update({key: dict(value) for key, value in updates.items()})
    sources.update({key: dict(value) for key, value in control_sources.items()})
    for name, identity in sources.items():
        if file_identity(identity["path"]) != identity:
            raise ValueError(f"replacement receipt source differs: {name}")
    return sources, missing


def _replacement_receipt(
    receipt: Mapping[str, Any],
    *,
    formal_snapshot: Mapping[str, Any],
    runtime: Mapping[str, Any],
    status_identity: Mapping[str, Any],
    pointer_identity: Mapping[str, Any],
    target_git: Mapping[str, Any],
    target_project: Path,
    archived_prior: Mapping[str, Any],
    prior_canonical: Mapping[str, Any],
    migration: Mapping[str, Any],
    source_updates: Mapping[str, Mapping[str, Any]],
    control_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(receipt))
    sources, missing = _active_sources(
        result,
        updates=source_updates,
        control_sources=control_sources,
    )
    legacy_bundle = result.pop("incremental_bundle", None)
    supervisor = copy.deepcopy(dict(result["supervisor"]))
    supervisor.update(
        {
            "pid": runtime["pid"],
            "process": shlex.join(list(runtime["argv"])),
            "process_argv": list(runtime["argv"]),
            "initial_status_identity": dict(status_identity),
            "pid_pointer": dict(pointer_identity),
        }
    )
    result.update(
        {
            "deployed_at": _timestamp(),
            "hostname": socket.gethostname(),
            "formal_checkout_snapshot": dict(formal_snapshot),
            "sources": sources,
            "supervisor": supervisor,
            "control_checkout": {
                "path": target_project.resolve().as_posix(),
                "git": _core_git(target_git),
            },
            "supersession": {
                "schema_version": 1,
                "role": "capacity_supervisor_deployment_receipt_supersession",
                "reason": "formal_checkout_snapshot_changed_and_failed_migration",
                "superseded_canonical": dict(prior_canonical),
                "byte_preserving_archive": dict(archived_prior),
                "migration": copy.deepcopy(dict(migration)),
                "legacy_missing_sources": missing,
                "legacy_incremental_bundle": legacy_bundle,
                "claim_boundary": {
                    "scientific_checkout_or_protocol_changed": False,
                    "formal_checkout_modified": False,
                    "direct_gpu_work_launched": False,
                    "control_process_continuity_repaired": True,
                },
            },
        }
    )
    return result


def _rotate_receipt(
    canonical: Path,
    payload: Mapping[str, Any],
    *,
    staging_root: Path,
) -> dict[str, Any]:
    staging = staging_root / f".{canonical.name}.{os.getpid()}.replacement"
    write_json_report(staging, dict(payload))
    staged_identity = file_identity(staging)
    os.replace(staging, canonical)
    identity = file_identity(canonical)
    if (
        identity["bytes"] != staged_identity["bytes"]
        or identity["sha256"] != staged_identity["sha256"]
    ):
        raise ValueError(f"canonical receipt rotation failed: {canonical}")
    return identity


def _launch_and_wait(
    row: Mapping[str, Any],
    *,
    timeout_seconds: float,
    launcher: Callable[[Mapping[str, Any]], ProcessHandle],
    status_waiter: Callable[..., Mapping[str, Any]],
    terminator: Callable[..., Mapping[str, Any]],
) -> tuple[ProcessHandle, dict[str, Any]]:
    launched_after_ns = time.time_ns()
    process = launcher(row)
    try:
        evidence = dict(
            status_waiter(
                row,
                process,
                launched_after_ns=launched_after_ns,
                timeout_seconds=timeout_seconds,
            )
        )
    except BaseException:
        terminator(row, process, grace_seconds=10.0)
        raise
    return process, evidence


def _validate_plan_sources(plan: Mapping[str, Any]) -> None:
    if _git_state_with_porcelain(Path(plan["target_git"]["path"])) != plan[
        "target_git"
    ]:
        raise ValueError("repair target checkout changed after planning")
    if _git_state_with_porcelain(Path(plan["formal_git"]["path"])) != plan[
        "formal_git"
    ]:
        raise ValueError("formal checkout changed after repair planning")
    for identity in plan["migration"].values():
        if file_identity(identity["path"]) != identity:
            raise ValueError("failed migration lineage changed after planning")
    for receipt in plan["receipts"]:
        if file_identity(receipt["canonical"]["path"]) != receipt["canonical"]:
            raise ValueError(f"canonical receipt changed after planning: {receipt['name']}")
    for row in plan["launch_rows"].values():
        if file_identity(row["entrypoint"]["identity"]["path"]) != row[
            "entrypoint"
        ]["identity"]:
            raise ValueError(f"repair entrypoint changed: {row['name']}")
        lock = row["lifetime_lock"]
        stat = Path(lock["path"]).stat()
        if (
            file_identity(lock["path"]) != lock["identity"]
            or stat.st_dev != lock["device"]
            or stat.st_ino != lock["inode"]
        ):
            raise ValueError(f"repair lifetime lock changed: {row['name']}")


def execute_supervisor_receipt_repair(
    *,
    approval_path: Path,
    expected_approval_sha256: str,
    publish: Callable[[Mapping[str, Any]], None],
    status_timeout_seconds: float = 30.0,
    inspect_process: Callable[[int], Mapping[str, Any]] = inspect_linux_process,
    process_scanner: Callable[[str], Sequence[int]] = _processes_for_entrypoint,
    launcher: Callable[[Mapping[str, Any]], ProcessHandle] = _launch_process,
    status_waiter: Callable[..., Mapping[str, Any]] = _wait_for_waiting_status,
    terminator: Callable[..., Mapping[str, Any]] = _terminate_handle,
    lock_inspector: Callable[[int, Path], Mapping[str, Any]] = (
        inspect_linux_lifetime_lock
    ),
) -> dict[str, Any]:
    if status_timeout_seconds <= 0:
        raise ValueError("repair status timeout must be positive")
    approval_id = _require_identity(
        approval_path,
        expected_approval_sha256,
        label="supervisor receipt repair approval",
    )
    approval = _read_json(approval_path, label="supervisor receipt repair approval")
    if (
        approval.get("schema_version") != REPAIR_APPROVAL_SCHEMA_VERSION
        or approval.get("role") != REPAIR_APPROVAL_ROLE
        or approval.get("status") != "approved"
        or approval.get("scope") != REPAIR_SCOPE
    ):
        raise ValueError("supervisor receipt repair approval contract differs")
    plan_id = file_identity(approval["plan"]["path"])
    if plan_id != approval["plan"]:
        raise ValueError("supervisor receipt repair plan changed after approval")
    plan = _read_json(Path(plan_id["path"]), label="supervisor receipt repair plan")
    if (
        plan.get("schema_version") != REPAIR_PLAN_SCHEMA_VERSION
        or plan.get("role") != REPAIR_PLAN_ROLE
        or plan.get("status") != "ready"
        or plan.get("scope") != REPAIR_SCOPE
        or approval.get("migration") != plan.get("migration")
        or approval.get("target_git") != plan.get("target_git")
        or approval.get("formal_git") != plan.get("formal_git")
    ):
        raise ValueError("supervisor receipt repair plan contract differs")
    _validate_plan_sources(plan)

    training_runtime = dict(
        inspect_process(int(plan["live_training"]["runtime"]["pid"]))
    )
    if training_runtime != plan["live_training"]["runtime"]:
        raise ValueError("live training supervisor changed after repair planning")
    _validate_runtime_barrier(_spec(TRAINING_NAME), training_runtime)
    _status_snapshot(_spec(TRAINING_NAME), runtime=training_runtime)
    _barriers(_spec(TRAINING_NAME), runtime=training_runtime)
    for name in DOWNSTREAM_NAMES:
        if process_scanner(str(_spec(name)["entrypoint"])):
            raise ValueError(f"repair target process already exists: {name}")
    _validate_barriers(tuple(plan["launch_rows"].values()))

    receipt_by_name = {row["name"]: row for row in plan["receipts"]}
    staging_root = Path(plan["archive_root"])
    archived = {}
    for name, row in receipt_by_name.items():
        archived[name] = _archive_exact(
            Path(row["canonical"]["path"]),
            Path(row["archive_path"]),
        )
    base = {
        "schema_version": REPAIR_EXECUTION_SCHEMA_VERSION,
        "role": REPAIR_EXECUTION_ROLE,
        "started_at": _timestamp(),
        "approval": approval_id,
        "plan": plan_id,
        "migration": plan["migration"],
        "scope": dict(REPAIR_SCOPE),
        "archives": archived,
    }
    publish({**base, "status": "running", "phase": "rotating_training_receipt"})

    control_sources = {
        "failed_migration_execution": plan["migration"]["execution"],
        "hardened_control_entrypoint": file_identity(
            Path(plan["target_git"]["path"])
            / str(_spec(TRAINING_NAME)["entrypoint"])
        ),
    }
    training_row = receipt_by_name["training"]
    training_path = Path(training_row["canonical"]["path"])
    training_old = _read_json(
        Path(training_row["archive_path"]),
        label="archived training receipt",
    )
    training_pointer = _write_pointer(
        Path(training_old["supervisor"]["pid_pointer"]["path"]),
        int(training_runtime["pid"]),
    )
    training_replacement = _replacement_receipt(
        training_old,
        formal_snapshot=plan["formal_checkout_snapshot"],
        runtime=training_runtime,
        status_identity=file_identity(plan["live_training"]["status"]["path"]),
        pointer_identity=training_pointer,
        target_git=plan["target_git"],
        target_project=Path(plan["target_git"]["path"]),
        archived_prior=archived["training"],
        prior_canonical=training_row["canonical"],
        migration=plan["migration"],
        source_updates={},
        control_sources=control_sources,
    )
    training_identity = _rotate_receipt(
        training_path,
        training_replacement,
        staging_root=staging_root,
    )

    launched = {}
    publish({**base, "status": "running", "phase": "launching_posteval"})
    posteval_row = copy.deepcopy(plan["launch_rows"][POSTEVAL_NAME])
    posteval_row["runtime"]["argv"] = _replace_cli_value(
        posteval_row["runtime"]["argv"],
        "--expected-training-supervisor-deployment-sha256",
        training_identity["sha256"],
    )
    posteval_process, posteval_evidence = _launch_and_wait(
        posteval_row,
        timeout_seconds=status_timeout_seconds,
        launcher=launcher,
        status_waiter=status_waiter,
        terminator=terminator,
    )
    launched[POSTEVAL_NAME] = posteval_evidence

    posteval_receipt_row = receipt_by_name["posteval"]
    posteval_path = Path(posteval_receipt_row["canonical"]["path"])
    posteval_old = _read_json(
        Path(posteval_receipt_row["archive_path"]),
        label="archived posteval receipt",
    )
    posteval_pointer = _write_pointer(
        Path(posteval_old["supervisor"]["pid_pointer"]["path"]),
        posteval_process.pid,
    )
    posteval_replacement = _replacement_receipt(
        posteval_old,
        formal_snapshot=plan["formal_checkout_snapshot"],
        runtime=inspect_process(posteval_process.pid),
        status_identity=posteval_evidence["status_identity"],
        pointer_identity=posteval_pointer,
        target_git=plan["target_git"],
        target_project=Path(plan["target_git"]["path"]),
        archived_prior=archived["posteval"],
        prior_canonical=posteval_receipt_row["canonical"],
        migration=plan["migration"],
        source_updates={"training_supervisor_deployment": training_identity},
        control_sources={
            "failed_migration_execution": plan["migration"]["execution"],
            "hardened_control_entrypoint": posteval_row["entrypoint"]["identity"],
        },
    )
    posteval_replacement["training_supervisor_deployment"] = training_identity
    posteval_identity = _rotate_receipt(
        posteval_path,
        posteval_replacement,
        staging_root=staging_root,
    )

    publish({**base, "status": "running", "phase": "launching_finalization"})
    finalization_row = copy.deepcopy(plan["launch_rows"][FINALIZATION_NAME])
    finalization_row["runtime"]["argv"] = _replace_cli_value(
        finalization_row["runtime"]["argv"],
        "--expected-training-supervisor-deployment-sha256",
        training_identity["sha256"],
    )
    finalization_row["runtime"]["argv"] = _replace_cli_value(
        finalization_row["runtime"]["argv"],
        "--expected-posteval-supervisor-deployment-sha256",
        posteval_identity["sha256"],
    )
    finalization_process, finalization_evidence = _launch_and_wait(
        finalization_row,
        timeout_seconds=status_timeout_seconds,
        launcher=launcher,
        status_waiter=status_waiter,
        terminator=terminator,
    )
    launched[FINALIZATION_NAME] = finalization_evidence

    finalization_receipt_row = receipt_by_name["finalization"]
    finalization_path = Path(finalization_receipt_row["canonical"]["path"])
    finalization_old = _read_json(
        Path(finalization_receipt_row["archive_path"]),
        label="archived finalization receipt",
    )
    finalization_pointer = _write_pointer(
        Path(finalization_old["supervisor"]["pid_pointer"]["path"]),
        finalization_process.pid,
    )
    finalization_replacement = _replacement_receipt(
        finalization_old,
        formal_snapshot=plan["formal_checkout_snapshot"],
        runtime=inspect_process(finalization_process.pid),
        status_identity=finalization_evidence["status_identity"],
        pointer_identity=finalization_pointer,
        target_git=plan["target_git"],
        target_project=Path(plan["target_git"]["path"]),
        archived_prior=archived["finalization"],
        prior_canonical=finalization_receipt_row["canonical"],
        migration=plan["migration"],
        source_updates={
            "training_supervisor_deployment": training_identity,
            "posteval_supervisor_deployment": posteval_identity,
        },
        control_sources={
            "failed_migration_execution": plan["migration"]["execution"],
            "hardened_control_entrypoint": finalization_row["entrypoint"][
                "identity"
            ],
        },
    )
    finalization_replacement["training_supervisor_deployment"] = training_identity
    finalization_replacement["posteval_supervisor_deployment"] = posteval_identity
    finalization_identity = _rotate_receipt(
        finalization_path,
        finalization_replacement,
        staging_root=staging_root,
    )

    for name, evidence in launched.items():
        lock = evidence.get("lifetime_lock")
        if not isinstance(lock, Mapping):
            raise ValueError(f"repaired supervisor lacks lifetime lock: {name}")
        lock_inspector(int(evidence["pid"]), Path(lock["path"]))
    report = {
        **base,
        "status": "pass",
        "phase": "completed",
        "completed_at": _timestamp(),
        "canonical_receipts": {
            "training": training_identity,
            "posteval": posteval_identity,
            "finalization": finalization_identity,
        },
        "launched": launched,
        "live_training": {
            "pid": training_runtime["pid"],
            "status": file_identity(plan["live_training"]["status"]["path"]),
        },
        "effects": {
            "receipts_archived": True,
            "canonical_receipts_rotated": True,
            "processes_launched": list(DOWNSTREAM_NAMES),
            "processes_signaled": False,
            "gpu_queried_or_allocated": False,
            "formal_checkout_modified": False,
        },
    }
    publish(report)
    return report
