from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


CONTRACT_SCHEMA_VERSION = 1
CONTRACT_ROLE = "quality_bridge_recovery_supersession_contract"
SUCCESSOR_ROLE = "cofitok_quality_bridge_100k_bounded_recovery_supervisor_v2"
DEPLOYMENT_RECEIPT_ROLE = (
    "cofitok_quality_bridge_ipc_recovery_v2_deployment_receipt"
)
EXECUTION_ROLE = "stability_full_data_quality_bridge_execution"
WATCHDOG_ROLE = "generation_training_watchdog"
CHECKPOINT_AUDIT_ROLE = "generation_checkpoint_physical_integrity_milestone_audit"
EXPECTED_MONITOR_NAME = "generation_stability_full_data_quality_bridge_100k"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("recovery supersession timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _identity(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_json_source(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    payload = source.read_bytes()
    identity = _identity(source, payload)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{source.name} SHA256 differs")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source.name} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{source.name} must contain a JSON object")
    return parsed, identity


def _absolute_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    if not Path(value).is_absolute() and not PurePosixPath(value).is_absolute():
        raise ValueError(f"{label} path is not absolute")
    return value


def _file_reference(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file reference differs")
    path = _absolute_path(value["path"], label=label)
    digest = value["sha256"]
    if not _is_sha256(digest):
        raise ValueError(f"{label} SHA256 is malformed")
    return {"path": path, "sha256": digest}


def load_recovery_supersession_contract(path: str | Path) -> dict[str, Any]:
    payload, identity = _read_json_source(path)
    if set(payload) != {
        "active_chain",
        "exact_resume",
        "expected_training",
        "role",
        "schema_version",
        "successor",
        "superseded_failure",
    }:
        raise ValueError("recovery supersession contract schema differs")
    if (
        payload.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or payload.get("role") != CONTRACT_ROLE
    ):
        raise ValueError("recovery supersession contract identity differs")

    failure = payload["superseded_failure"]
    if not isinstance(failure, dict) or set(failure) != {
        "detail_prefix",
        "role",
        "source",
    }:
        raise ValueError("superseded recovery failure contract differs")
    if not isinstance(failure["role"], str) or not failure["role"]:
        raise ValueError("superseded recovery failure role is missing")
    if not isinstance(failure["detail_prefix"], str) or not failure["detail_prefix"]:
        raise ValueError("superseded recovery failure detail is missing")
    failure["source"] = _file_reference(
        failure["source"], label="superseded recovery failure"
    )

    successor = payload["successor"]
    if not isinstance(successor, dict) or set(successor) != {
        "deployment_receipt",
        "role",
        "status_path",
    }:
        raise ValueError("recovery successor contract differs")
    if successor["role"] != SUCCESSOR_ROLE:
        raise ValueError("recovery successor role differs")
    successor["status_path"] = _absolute_path(
        successor["status_path"], label="recovery successor status"
    )
    successor["deployment_receipt"] = _file_reference(
        successor["deployment_receipt"], label="recovery successor receipt"
    )

    active = payload["active_chain"]
    if not isinstance(active, dict) or set(active) != {
        "execution_status_path",
        "pair_monitor_path",
        "watchdog_statuses",
    }:
        raise ValueError("recovery active-chain contract differs")
    for key in ("execution_status_path", "pair_monitor_path"):
        active[key] = _absolute_path(active[key], label=key)
    watchdog_statuses = active["watchdog_statuses"]
    if not isinstance(watchdog_statuses, list) or not watchdog_statuses:
        raise ValueError("recovery watchdog status contract is missing")
    normalized_watchdogs = []
    previous_target = 0
    for index, status in enumerate(watchdog_statuses):
        if not isinstance(status, dict) or set(status) != {"path", "target_step"}:
            raise ValueError("recovery watchdog status contract differs")
        target_step = status["target_step"]
        if type(target_step) is not int or target_step <= previous_target:
            raise ValueError("recovery watchdog target steps are not increasing")
        normalized_watchdogs.append(
            {
                "path": _absolute_path(
                    status["path"], label=f"watchdog_statuses[{index}]"
                ),
                "target_step": target_step,
            }
        )
        previous_target = target_step
    active["watchdog_statuses"] = normalized_watchdogs

    exact = payload["exact_resume"]
    if not isinstance(exact, dict) or set(exact) != {
        "checkpoint_audit",
        "checkpoint_step",
        "reconciliation",
        "resume_step",
        "run_manifest",
    }:
        raise ValueError("recovery exact-resume contract differs")
    exact["run_manifest"] = _file_reference(
        exact["run_manifest"], label="recovery run manifest"
    )
    exact["reconciliation"] = _file_reference(
        exact["reconciliation"], label="recovery metrics reconciliation"
    )
    exact["checkpoint_audit"] = _file_reference(
        exact["checkpoint_audit"], label="recovery checkpoint audit"
    )
    if (
        type(exact["resume_step"]) is not int
        or type(exact["checkpoint_step"]) is not int
        or exact["resume_step"] < 1
        or exact["checkpoint_step"] <= exact["resume_step"]
    ):
        raise ValueError("recovery exact-resume steps are invalid")

    training = payload["expected_training"]
    if not isinstance(training, dict) or set(training) != {
        "branch",
        "dataset_identity_sha256",
        "effective_batch",
        "output_root",
        "revision",
        "run_dir",
        "runtime_environment_sha256",
        "tree",
    }:
        raise ValueError("recovery expected-training contract differs")
    for key in ("revision", "tree"):
        if not isinstance(training[key], str) or len(training[key]) != 40:
            raise ValueError(f"recovery expected training {key} is malformed")
    if not isinstance(training["branch"], str) or not training["branch"]:
        raise ValueError("recovery expected training branch is missing")
    for key in ("dataset_identity_sha256", "runtime_environment_sha256"):
        if not _is_sha256(training[key]):
            raise ValueError(f"recovery expected training {key} is malformed")
    for key in ("output_root", "run_dir"):
        training[key] = _absolute_path(training[key], label=key)
    if type(training["effective_batch"]) is not int or training["effective_batch"] < 1:
        raise ValueError("recovery expected effective batch is invalid")

    return {"identity": identity, **payload}


def _fresh_active_status(
    payload: Mapping[str, Any],
    *,
    now: datetime,
    stale_seconds: float,
    pid_key: str,
    process_exists: Callable[[int], bool],
    label: str,
) -> tuple[int, str, float]:
    raw_pid = payload.get(pid_key)
    if type(raw_pid) is not int or raw_pid < 1:
        raise ValueError(f"{label} PID is missing")
    if not process_exists(raw_pid):
        raise ValueError(f"{label} process is not alive")
    raw_updated = payload.get("updated_at")
    updated = _parse_timestamp(raw_updated)
    if updated is None:
        raise ValueError(f"{label} timestamp is invalid")
    age = max((_utc(now) - updated).total_seconds(), 0.0)
    if age > stale_seconds:
        raise ValueError(f"{label} status is stale")
    return raw_pid, str(raw_updated), age


def _validate_training_git(
    value: Any,
    *,
    expected: Mapping[str, Any],
    include_tree: bool,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("recovery training Git payload is missing")
    required = {
        "revision": expected["revision"],
        "branch": expected["branch"],
    }
    if include_tree:
        required["tree"] = expected["tree"]
    for key, expected_value in required.items():
        if value.get(key) != expected_value:
            raise ValueError(f"recovery training Git {key} differs")
    dirty = value.get("tracked_dirty", value.get("dirty"))
    if dirty is not False:
        raise ValueError("recovery training Git is dirty")


def _validate_scope(scope: Any, *, exact_resume: bool = False) -> None:
    if not isinstance(scope, dict):
        raise ValueError("recovery scope is missing")
    for key in (
        "full_300k_launch_allowed",
        "full_training_launch_allowed",
        "release_authorization_allowed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"recovery scope {key} differs")
    if exact_resume and scope.get("quality_bridge_exact_resume_allowed") is not True:
        raise ValueError("recovery exact-resume scope differs")


def inspect_recovery_supersession(
    failed_stage: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    now: datetime,
    stale_seconds: float,
    process_exists: Callable[[int], bool],
) -> dict[str, Any]:
    if stale_seconds <= 0:
        raise ValueError("recovery supersession stale threshold must be positive")
    observed_at = _utc(now)
    failure_contract = contract["superseded_failure"]
    failure, failure_identity = _read_json_source(
        failure_contract["source"]["path"],
        expected_sha256=failure_contract["source"]["sha256"],
    )
    if (
        failed_stage.get("classification") != "failure"
        or failed_stage.get("health") != "failed"
        or failed_stage.get("identity", {}).get("sha256")
        != failure_contract["source"]["sha256"]
        or failure.get("role") != failure_contract["role"]
        or failure.get("status") != "failed"
        or not str(failure.get("detail", "")).startswith(
            failure_contract["detail_prefix"]
        )
    ):
        raise ValueError("superseded recovery failure identity differs")

    training = contract["expected_training"]
    successor_contract = contract["successor"]
    receipt, receipt_identity = _read_json_source(
        successor_contract["deployment_receipt"]["path"],
        expected_sha256=successor_contract["deployment_receipt"]["sha256"],
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("role") != DEPLOYMENT_RECEIPT_ROLE
        or receipt.get("status") != "pass"
    ):
        raise ValueError("recovery successor deployment receipt differs")
    _validate_scope(receipt.get("scope"), exact_resume=True)
    receipt_training = receipt.get("training")
    if not isinstance(receipt_training, dict) or any(
        receipt_training.get(key) != training[key]
        for key in ("revision", "tree", "branch", "output_root")
    ):
        raise ValueError("recovery successor receipt training identity differs")

    successor, successor_identity = _read_json_source(
        successor_contract["status_path"]
    )
    if successor.get("schema_version") != 1 or successor.get("role") != SUCCESSOR_ROLE:
        raise ValueError("recovery successor status identity differs")
    successor_status = str(successor.get("status", "")).lower()
    active_successor = successor_status in {"active", "observing", "running", "waiting"}
    complete_successor = successor_status in {
        "complete",
        "completed",
        "pass",
        "passed",
        "success",
        "succeeded",
    }
    if not active_successor and not complete_successor:
        raise ValueError("recovery successor is not active or complete")
    successor_pid = successor.get("supervisor_pid")
    successor_age: float | None = None
    successor_updated = successor.get("updated_at")
    if active_successor:
        successor_pid, successor_updated, successor_age = _fresh_active_status(
            successor,
            now=observed_at,
            stale_seconds=stale_seconds,
            pid_key="supervisor_pid",
            process_exists=process_exists,
            label="recovery successor",
        )
    _validate_training_git(successor.get("git"), expected=training, include_tree=True)
    _validate_scope(successor.get("scope"))
    deployed = receipt.get("deployment")
    successor_source = successor.get("identities", {}).get("recovery_supervisor")
    if (
        not isinstance(deployed, dict)
        or not isinstance(successor_source, dict)
        or successor_source.get("path") != deployed.get("deployed_source_path")
        or successor_source.get("bytes") != deployed.get("deployed_source_bytes")
        or successor_source.get("sha256") != deployed.get("deployed_source_sha256")
    ):
        raise ValueError("recovery successor deployed source binding differs")

    active = contract["active_chain"]
    execution, execution_identity = _read_json_source(active["execution_status_path"])
    if execution.get("schema_version") != 1 or execution.get("role") != EXECUTION_ROLE:
        raise ValueError("quality-bridge execution status identity differs")
    execution_status = str(execution.get("status", "")).lower()
    if execution_status not in {
        "complete",
        "completed",
        "pass",
        "passed",
        "running",
        "success",
        "succeeded",
    }:
        raise ValueError("quality-bridge execution status differs")
    if execution_status == "running":
        execution_pid = execution.get("pid")
        if type(execution_pid) is not int or not process_exists(execution_pid):
            raise ValueError("quality-bridge execution process is not alive")
    _validate_training_git(execution.get("git"), expected=training, include_tree=False)
    if (
        execution.get("quality_bridge_only") is not True
        or execution.get("full_300k_launch_allowed") is not False
        or execution.get("full_training_launch_allowed") is not False
        or execution.get("report_is_promotion_gate") is not False
    ):
        raise ValueError("quality-bridge execution scope differs")

    pair, pair_identity = _read_json_source(active["pair_monitor_path"])
    pair_status = str(pair.get("status", "")).lower()
    if (
        pair.get("schema_version") not in {1, 2}
        or pair.get("monitor") != EXPECTED_MONITOR_NAME
        or pair_status not in {"complete", "completed", "pass", "passed", "running"}
        or pair.get("issues") != []
    ):
        raise ValueError("quality-bridge pair monitor is unhealthy")
    pair_updated = _parse_timestamp(pair.get("updated_at"))
    if pair_updated is None:
        raise ValueError("quality-bridge pair monitor timestamp is invalid")
    pair_age = max((observed_at - pair_updated).total_seconds(), 0.0)
    if pair_status == "running" and pair_age > stale_seconds:
        raise ValueError("quality-bridge pair monitor is stale")
    _validate_training_git(pair.get("git"), expected=training, include_tree=False)
    runs = pair.get("runs")
    cofitok = runs.get("cofitok") if isinstance(runs, dict) else None
    if not isinstance(cofitok, dict) or cofitok.get("health_issues") != []:
        raise ValueError("quality-bridge CoFiTok run is unhealthy")

    exact = contract["exact_resume"]
    run_manifest, run_manifest_identity = _read_json_source(
        exact["run_manifest"]["path"],
        expected_sha256=exact["run_manifest"]["sha256"],
    )
    manifest_git = run_manifest.get("git")
    _validate_training_git(manifest_git, expected=training, include_tree=False)
    if (
        run_manifest.get("output_dir") != training["run_dir"]
        or run_manifest.get("resume")
        != f"{training['run_dir']}/checkpoint_step_{exact['resume_step']:08d}.pt"
        or run_manifest.get("dataset_provenance", {}).get("identity_sha256")
        != training["dataset_identity_sha256"]
        or run_manifest.get("runtime_environment_sha256")
        != training["runtime_environment_sha256"]
    ):
        raise ValueError("quality-bridge exact-resume manifest differs")
    manifest_reconciliation = run_manifest.get("metrics_resume_reconciliation")
    if (
        not isinstance(manifest_reconciliation, dict)
        or manifest_reconciliation.get("status") != "reconciled"
        or manifest_reconciliation.get("resume_step") != exact["resume_step"]
        or manifest_reconciliation.get("report")
        != exact["reconciliation"]["path"]
    ):
        raise ValueError("quality-bridge manifest reconciliation differs")

    reconciliation, reconciliation_identity = _read_json_source(
        exact["reconciliation"]["path"],
        expected_sha256=exact["reconciliation"]["sha256"],
    )
    if (
        reconciliation.get("schema_version") != 1
        or reconciliation.get("status") != "reconciled"
        or reconciliation.get("resume_step") != exact["resume_step"]
        or type(reconciliation.get("orphaned_rows")) is not int
        or reconciliation.get("orphaned_rows", 0) < 1
        or reconciliation.get("orphan_sha256")
        != manifest_reconciliation.get("orphan_sha256")
    ):
        raise ValueError("quality-bridge metrics reconciliation differs")

    checkpoint, checkpoint_identity = _read_json_source(
        exact["checkpoint_audit"]["path"],
        expected_sha256=exact["checkpoint_audit"]["sha256"],
    )
    checkpoint_payload = checkpoint.get("checkpoint")
    integrity = (
        checkpoint_payload.get("integrity")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("role") != CHECKPOINT_AUDIT_ROLE
        or checkpoint.get("status") != "pass"
        or checkpoint.get("run_dir") != training["run_dir"]
        or not isinstance(checkpoint_payload, dict)
        or checkpoint_payload.get("step") != exact["checkpoint_step"]
        or checkpoint_payload.get("physical_sha256_verified") is not True
        or not isinstance(integrity, dict)
        or integrity.get("step") != exact["checkpoint_step"]
        or integrity.get("git_revision") != training["revision"]
        or integrity.get("git_branch") != training["branch"]
        or integrity.get("git_dirty") is not False
        or integrity.get("dataset_identity_sha256")
        != training["dataset_identity_sha256"]
        or integrity.get("runtime_environment_sha256")
        != training["runtime_environment_sha256"]
        or checkpoint.get("latest_pointer", {}).get("exact_target_binding") is not True
        or checkpoint.get("metrics", {}).get("strictly_increasing") is not True
        or checkpoint.get("metrics", {}).get("samples_seen_binding_verified")
        is not True
        or checkpoint.get("metrics", {}).get("target_row", {}).get("samples_seen")
        != exact["checkpoint_step"] * training["effective_batch"]
    ):
        raise ValueError("quality-bridge trusted checkpoint audit differs")
    _validate_training_git(
        checkpoint.get("training_checkout"), expected=training, include_tree=True
    )

    last_step = cofitok.get("last_step")
    if type(last_step) is not int or last_step < exact["checkpoint_step"]:
        raise ValueError("quality-bridge pair monitor predates trusted checkpoint")
    run_manifest_status = cofitok.get("run_manifest")
    if (
        not isinstance(run_manifest_status, dict)
        or run_manifest_status.get("status") != "verified"
        or run_manifest_status.get("git_revision") != training["revision"]
        or run_manifest_status.get("dataset_identity_sha256")
        != training["dataset_identity_sha256"]
        or run_manifest_status.get("runtime_environment_sha256")
        != training["runtime_environment_sha256"]
    ):
        raise ValueError("quality-bridge live run-manifest binding differs")

    existing_watchdogs: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    missing_watchdog_seen = False
    for watchdog_contract in active["watchdog_statuses"]:
        watchdog_path = Path(watchdog_contract["path"])
        if not watchdog_path.is_file():
            missing_watchdog_seen = True
            continue
        if missing_watchdog_seen:
            raise ValueError("quality-bridge watchdog status sequence has a gap")
        watchdog_payload, watchdog_source = _read_json_source(watchdog_path)
        existing_watchdogs.append(
            (watchdog_contract, watchdog_payload, watchdog_source)
        )
    if not existing_watchdogs:
        raise ValueError("quality-bridge watchdog status is missing")
    watchdog_contract, watchdog, watchdog_identity = existing_watchdogs[-1]
    watchdog_target_step = watchdog_contract["target_step"]
    if last_step > watchdog_target_step:
        raise ValueError("quality-bridge next watchdog status is missing")
    if watchdog.get("schema_version") != 1 or watchdog.get("role") != WATCHDOG_ROLE:
        raise ValueError("quality-bridge watchdog identity differs")
    cofitok_complete = cofitok.get("complete") is True
    watchdog_status = str(watchdog.get("status", "")).lower()
    active_watchdog = watchdog_status == "running"
    complete_watchdog = watchdog_status in {
        "complete",
        "completed",
        "pass",
        "passed",
        "success",
        "succeeded",
    }
    if last_step < watchdog_target_step:
        if not active_watchdog:
            raise ValueError("quality-bridge watchdog is not running")
    elif not active_watchdog and not complete_watchdog:
        raise ValueError("quality-bridge milestone watchdog differs")

    if active_watchdog:
        watchdog_pid, _, _ = _fresh_active_status(
            watchdog,
            now=observed_at,
            stale_seconds=stale_seconds,
            pid_key="pid",
            process_exists=process_exists,
            label="quality-bridge watchdog",
        )
        child_pid = watchdog.get("child_pid")
        monitor_pid = watchdog.get("monitor_pid")
        if (
            type(child_pid) is not int
            or type(monitor_pid) is not int
            or not process_exists(child_pid)
            or not process_exists(monitor_pid)
            or watchdog.get("monitor_process_alive") is not True
            or watchdog.get("child_exit_code") is not None
            or watchdog.get("watchdog_exit_code") is not None
            or watchdog.get("reason") != "child_and_monitor_active"
            or watchdog.get("monitor_report_path") != active["pair_monitor_path"]
            or watchdog.get("monitor", {}).get("status") != "running"
        ):
            raise ValueError("quality-bridge trainer/watchdog binding differs")
        command = watchdog.get("command")
        if (
            not isinstance(command, list)
            or "--resume" not in command
            or "auto" not in command
            or training["run_dir"] not in command
        ):
            raise ValueError("quality-bridge exact-resume command differs")
    else:
        watchdog_pid = watchdog.get("pid")
        if watchdog.get("child_exit_code") not in {0, None}:
            raise ValueError("completed quality-bridge watchdog differs")
    if cofitok_complete and (
        last_step != active["watchdog_statuses"][-1]["target_step"]
        or watchdog_target_step != active["watchdog_statuses"][-1]["target_step"]
        or not complete_watchdog
    ):
        raise ValueError("completed quality-bridge watchdog lineage differs")

    synthetic_status = "pass" if complete_successor else "running"
    return {
        **dict(failed_stage),
        "identity": successor_identity,
        "role": successor.get("role"),
        "status": synthetic_status,
        "detail": (
            "superseded_by_source_bound_exact_resume:"
            f"{successor.get('detail')}"
        ),
        "decision": successor.get("decision"),
        "attempt": successor.get("attempt"),
        "classification": "success" if complete_successor else "active",
        "health": "pass" if complete_successor else "running",
        "pid": successor_pid,
        "process_alive": (
            True
            if active_successor
            else bool(successor_pid and process_exists(successor_pid))
        ),
        "updated_at": successor_updated,
        "age_seconds": successor_age,
        "issues": [],
        "supersession": {
            "status": "pass",
            "verified_at": observed_at.isoformat(),
            "contract": dict(contract["identity"]),
            "superseded_failure": failure_identity,
            "successor_status": successor_identity,
            "successor_deployment_receipt": receipt_identity,
            "execution_status": execution_identity,
            "pair_monitor": pair_identity,
            "watchdog_status": watchdog_identity,
            "run_manifest": run_manifest_identity,
            "metrics_reconciliation": reconciliation_identity,
            "trusted_checkpoint_audit": checkpoint_identity,
            "resume_step": exact["resume_step"],
            "trusted_checkpoint_step": exact["checkpoint_step"],
            "live_cofitok_step": last_step,
            "watchdog_target_step": watchdog_target_step,
            "watchdog_pid": watchdog_pid,
            "claim_boundary": {
                "training_launch_allowed": False,
                "sampling_or_evaluation_launch_allowed": False,
                "promotion_or_release_allowed": False,
                "process_signaling_allowed": False,
                "gpu_allocation_allowed": False,
                "read_only_observation_only": True,
            },
        },
    }
