from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from cofitok.generation_control_continuity import git_state
from cofitok.generation_control_process_snapshot import (
    AUTHORIZATION_BOUNDARY as SNAPSHOT_AUTHORIZATION_BOUNDARY,
    verify_process_relaunch_manifest,
)
from cofitok.inference_replay import file_identity


READINESS_SCHEMA_VERSION = 1
APPROVAL_SCHEMA_VERSION = 1
EXECUTION_SCHEMA_VERSION = 1
READINESS_ROLE = "generation_control_plane_process_relaunch_readiness"
APPROVAL_ROLE = "generation_control_plane_process_relaunch_approval"
EXECUTION_ROLE = "generation_control_plane_process_relaunch_execution"
STANDING_AUTHORIZATION_ROLE = "cofitok_standing_experiment_authorization_record"
STANDING_AUTHORIZATION_TEXT = "之后不要我授权你直接运行需要的实验"
SUCCESS_STATUSES = frozenset(
    {"pass", "passed", "complete", "completed", "ready", "success", "succeeded"}
)
ACTIVE_STATUSES = frozenset(
    {"active", "observing", "retrying", "running", "starting", "waiting"}
)
HOLD_STATUSES = frozenset({"blocked", "hold", "not_selected", "rejected"})
FAILURE_STATUSES = frozenset({"fail", "failed", "invalid", "stalled"})
READINESS_AUTHORIZATION_BOUNDARY = {
    "read_process_table_allowed": True,
    "read_control_status_allowed": True,
    "verify_static_sources_allowed": True,
    "issue_cpu_control_relaunch_approval_when_ready": True,
    "process_launch_allowed": False,
    "process_signaling_allowed": False,
    "gpu_query_or_allocation_allowed": False,
    "direct_training_sampling_or_evaluation_launch_allowed": False,
    "formal_checkout_modification_allowed": False,
}
APPROVAL_SCOPE = {
    "cpu_control_process_relaunch_allowed": True,
    "direct_training_sampling_or_evaluation_launch_allowed": False,
    "existing_stage_authorizations_remain_required": True,
    "unrelated_process_modification_allowed": False,
    "owned_partial_relaunch_rollback_allowed": True,
    "gpu_query_or_allocation_allowed": False,
    "formal_checkout_modification_allowed": False,
}


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...


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


def _classification(value: Any) -> str:
    if not isinstance(value, str):
        return "invalid"
    normalized = value.strip().lower()
    if normalized in SUCCESS_STATUSES:
        return "success"
    if normalized in ACTIVE_STATUSES:
        return "active"
    if normalized in HOLD_STATUSES:
        return "hold"
    if normalized in FAILURE_STATUSES:
        return "failure"
    return "invalid"


def _status_pid(payload: Mapping[str, Any], *, observer: bool) -> int | None:
    if observer:
        value = payload.get("observer")
        raw = value.get("pid") if isinstance(value, Mapping) else None
    else:
        raw = payload.get("pid", payload.get("supervisor_pid"))
    return raw if type(raw) is int and raw > 0 else None


def _validate_standing_authorization(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("standing authorization SHA256 differs")
    payload = _read_json(path, label="standing experiment authorization")
    instruction = payload.get("instruction")
    boundaries = payload.get("preserved_safety_boundaries")
    required_boundaries = {
        "unrelated_project_processes_must_not_be_modified",
        "formal_remote_checkout_must_not_be_modified",
        "locked_evidence_must_not_be_overwritten",
        "independent_clean_checkout_required",
        "exact_revision_stage_and_output_binding_required",
        "stage_must_remain_non_authorizing_when_protocol_declares_non_authorizing",
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("role") != STANDING_AUTHORIZATION_ROLE
        or payload.get("status") != "active"
        or not isinstance(instruction, Mapping)
        or instruction.get("exact_text") != STANDING_AUTHORIZATION_TEXT
        or not isinstance(boundaries, Mapping)
        or any(boundaries.get(key) is not True for key in required_boundaries)
    ):
        raise ValueError("standing experiment authorization contract differs")
    return identity


def _verify_git_state(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    core_keys = {"path", "revision", "tree", "branch", "tracked_dirty"}
    optional_keys = {"porcelain_count", "porcelain_sha256"}
    if not core_keys.issubset(expected) or not set(expected).issubset(
        core_keys | optional_keys
    ):
        raise ValueError(f"{label} expected Git schema differs")
    project = Path(str(expected["path"]))
    observed: dict[str, Any] = git_state(project)
    if optional_keys & set(expected):
        if not optional_keys.issubset(expected):
            raise ValueError(f"{label} porcelain identity is incomplete")
        porcelain = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=project,
            check=True,
            capture_output=True,
        ).stdout
        observed.update(
            {
                "porcelain_count": len(porcelain.splitlines()),
                "porcelain_sha256": hashlib.sha256(porcelain).hexdigest(),
            }
        )
    if observed != dict(expected):
        raise ValueError(f"{label} Git state differs")
    return observed


def _entrypoint_path(row: Mapping[str, Any], *, cwd: str | None = None) -> str:
    runtime = row["runtime"]
    argument = row["entrypoint"]["argument"]
    source = Path(argument)
    base = Path(cwd or runtime["cwd"])
    if not source.is_absolute():
        source = base / source
    return source.resolve().as_posix()


def _start_ticks(proc: Path) -> int | None:
    try:
        raw = (proc / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    end = raw.rfind(") ")
    fields = raw[end + 2 :].split() if end >= 0 else []
    return int(fields[19]) if len(fields) >= 20 else None


def scan_matching_processes(
    rows: Sequence[Mapping[str, Any]], *, proc_root: Path = Path("/proc")
) -> list[dict[str, Any]]:
    signatures: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (
            Path(row["runtime"]["cwd"]).resolve().as_posix(),
            _entrypoint_path(row),
        )
        signatures.setdefault(key, []).append(str(row["name"]))
    matches = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cwd = Path(os.readlink(proc / "cwd")).resolve().as_posix()
            argv = [
                part.decode("utf-8", errors="surrogateescape")
                for part in (proc / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except OSError:
            continue
        for row in rows:
            index = row["entrypoint"]["argument_index"]
            if index >= len(argv):
                continue
            raw = Path(argv[index])
            source = raw if raw.is_absolute() else Path(cwd) / raw
            key = (cwd, source.resolve().as_posix())
            if row["name"] not in signatures.get(key, []):
                continue
            matches.append(
                {
                    "name": row["name"],
                    "role": row["role"],
                    "pid": int(proc.name),
                    "start_ticks": _start_ticks(proc),
                    "cwd": cwd,
                    "entrypoint": key[1],
                }
            )
    return sorted(matches, key=lambda value: (value["name"], value["pid"]))


def _observe_status(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(row["status_path"])
    if not path.is_file():
        return {
            "name": row["name"],
            "role": row["role"],
            "blocking": row["blocking"],
            "path": path.resolve().as_posix(),
            "present": False,
            "status": None,
            "classification": "missing",
            "pid": None,
            "detail": "status_file_missing",
        }
    payload = _read_json(path, label=f"status for {row['name']}")
    observer = row["name"] == "capacity_pipeline_lineage_observer"
    if payload.get("role") != row["role"]:
        raise ValueError(f"status role differs for {row['name']}")
    return {
        "name": row["name"],
        "role": row["role"],
        "blocking": row["blocking"],
        "path": path.resolve().as_posix(),
        "present": True,
        "status": payload.get("status"),
        "classification": _classification(payload.get("status")),
        "pid": _status_pid(payload, observer=observer),
        "detail": payload.get("detail"),
    }


def assess_process_relaunch_readiness(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    standing_authorization_path: Path,
    expected_standing_authorization_sha256: str,
    formal_git: Mapping[str, Any],
    implementation_git: Mapping[str, Any],
    expected_process_count: int,
    source_verifier: Callable[..., Mapping[str, Any]] = verify_process_relaunch_manifest,
    process_scanner: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]]
    | None = None,
) -> dict[str, Any]:
    manifest_identity = file_identity(manifest_path)
    if manifest_identity["sha256"] != expected_manifest_sha256:
        raise ValueError("process relaunch manifest SHA256 differs")
    manifest = _read_json(manifest_path, label="process relaunch manifest")
    rows = manifest.get("processes")
    if not isinstance(rows, list) or len(rows) != expected_process_count:
        raise ValueError("process relaunch manifest process count differs")
    source_verification = dict(
        source_verifier(
            manifest_path=manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_process_count=expected_process_count,
            require_live=False,
        )
    )
    if source_verification.get("status") != "pass":
        raise ValueError("process relaunch source verification did not pass")
    standing_identity = _validate_standing_authorization(
        standing_authorization_path,
        expected_sha256=expected_standing_authorization_sha256,
    )
    _verify_git_state(formal_git, label="formal checkout")
    _verify_git_state(implementation_git, label="process relaunch implementation")

    scanner = process_scanner or (lambda values: scan_matching_processes(values))
    live_matches = [dict(value) for value in scanner(rows)]
    statuses = [_observe_status(row) for row in rows]
    invalid = [
        value
        for value in statuses
        if value["classification"] == "invalid"
    ]
    terminal = [
        value
        for value in statuses
        if value["classification"] in {"hold", "failure"}
    ]
    selected = [
        value["name"]
        for value in statuses
        if value["classification"] in {"active", "missing"}
    ]
    skipped = [
        value["name"]
        for value in statuses
        if value["classification"] == "success"
    ]
    issues = []
    if live_matches:
        issues.append("matching_control_processes_are_still_alive")
    if invalid:
        issues.extend(f"invalid_status:{value['name']}" for value in invalid)
    if terminal:
        issues.extend(
            f"terminal_{value['classification']}:{value['name']}" for value in terminal
        )
    if terminal:
        status = "terminal"
        detail = "control_chain_has_hold_or_failure_status"
    elif invalid:
        status = "invalid"
        detail = "control_chain_status_is_invalid"
    elif live_matches:
        status = "not_ready"
        detail = "matching_control_processes_are_still_alive"
    elif selected:
        status = "ready"
        detail = "all_selected_control_processes_are_absent"
    else:
        status = "complete"
        detail = "all_control_processes_have_terminal_success_status"
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "role": READINESS_ROLE,
        "status": status,
        "detail": detail,
        "assessed_at": _timestamp(),
        "approval_allowed": status == "ready",
        "manifest": manifest_identity,
        "standing_authorization": standing_identity,
        "formal_git": dict(formal_git),
        "implementation_git": dict(implementation_git),
        "expected_process_count": expected_process_count,
        "source_verification": source_verification,
        "live_matches": live_matches,
        "statuses": statuses,
        "selected_process_names": selected,
        "successful_process_names": skipped,
        "issues": issues,
        "authorization_boundary": dict(READINESS_AUTHORIZATION_BOUNDARY),
        "snapshot_authorization_boundary": dict(SNAPSHOT_AUTHORIZATION_BOUNDARY),
        "effects": {
            "processes_launched": False,
            "processes_signaled": False,
            "gpu_queried_or_allocated": False,
            "formal_checkout_modified": False,
        },
    }


def build_process_relaunch_approval(
    *, readiness_path: Path, expected_readiness_sha256: str
) -> dict[str, Any]:
    identity = file_identity(readiness_path)
    if identity["sha256"] != expected_readiness_sha256:
        raise ValueError("process relaunch readiness SHA256 differs")
    readiness = _read_json(readiness_path, label="process relaunch readiness")
    if (
        readiness.get("schema_version") != READINESS_SCHEMA_VERSION
        or readiness.get("role") != READINESS_ROLE
        or readiness.get("status") != "ready"
        or readiness.get("approval_allowed") is not True
        or readiness.get("issues") != []
        or readiness.get("live_matches") != []
        or not readiness.get("selected_process_names")
        or readiness.get("authorization_boundary")
        != READINESS_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("process relaunch readiness does not authorize execution")
    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "role": APPROVAL_ROLE,
        "status": "approved",
        "approved_at": _timestamp(),
        "readiness": identity,
        "manifest": readiness["manifest"],
        "standing_authorization": readiness["standing_authorization"],
        "formal_git": readiness["formal_git"],
        "implementation_git": readiness["implementation_git"],
        "expected_process_count": readiness["expected_process_count"],
        "selected_process_names": readiness["selected_process_names"],
        "successful_process_names": readiness["successful_process_names"],
        "scope": dict(APPROVAL_SCOPE),
    }


def _validate_approval(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("process relaunch approval SHA256 differs")
    approval = _read_json(path, label="process relaunch approval")
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or approval.get("role") != APPROVAL_ROLE
        or approval.get("status") != "approved"
        or approval.get("scope") != APPROVAL_SCOPE
        or not isinstance(approval.get("selected_process_names"), list)
        or not approval["selected_process_names"]
    ):
        raise ValueError("process relaunch approval contract differs")
    return approval, identity


def _launch_process(row: Mapping[str, Any]) -> subprocess.Popen[bytes]:
    runtime = row["runtime"]
    io = row["relaunch_io"]
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    target_nice = int(runtime["nice"])
    if target_nice < current_nice:
        raise ValueError(f"executor cannot lower niceness for {row['name']}")
    stdout_path = Path(io["stdout_target"])
    stderr_path = Path(io["stderr_target"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("ab", buffering=0)
    stderr_handle = None
    stderr_value: int | Any = subprocess.STDOUT
    if stderr_path.resolve() != stdout_path.resolve():
        stderr_handle = stderr_path.open("ab", buffering=0)
        stderr_value = stderr_handle
    environment = dict(os.environ)
    environment.update(runtime["environment"])
    nice_delta = target_nice - current_nice

    def prepare() -> None:
        if nice_delta:
            os.nice(nice_delta)

    try:
        process = subprocess.Popen(
            runtime["argv"],
            cwd=runtime["cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_value,
            start_new_session=True,
            preexec_fn=prepare,
        )
    finally:
        stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
    return process


def _wait_for_status(
    row: Mapping[str, Any],
    process: ProcessHandle,
    *,
    launched_after_ns: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    path = Path(row["status_path"])
    observer = row["name"] == "capacity_pipeline_lineage_observer"
    while time.monotonic() < deadline:
        code = process.poll()
        if path.is_file() and path.stat().st_mtime_ns >= launched_after_ns:
            payload = _read_json(path, label=f"relaunched status for {row['name']}")
            classification = _classification(payload.get("status"))
            if (
                payload.get("role") == row["role"]
                and _status_pid(payload, observer=observer) == process.pid
            ):
                if classification in {"hold", "failure", "invalid"}:
                    raise RuntimeError(f"relaunched process reported {classification}: {row['name']}")
                if code is None or (code == 0 and classification == "success"):
                    return {
                        "name": row["name"],
                        "role": row["role"],
                        "pid": process.pid,
                        "status": payload.get("status"),
                        "classification": classification,
                        "process_alive": code is None,
                        "exit_code": code,
                        "status_identity": file_identity(path),
                    }
        if code is not None:
            raise RuntimeError(
                f"relaunched process exited before a valid status: {row['name']}:{code}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"relaunched process status timed out: {row['name']}")


def _rollback_owned_processes(
    launched: Sequence[tuple[Mapping[str, Any], ProcessHandle]], *, grace_seconds: float
) -> list[dict[str, Any]]:
    rows = []
    for row, process in reversed(launched):
        if process.poll() is not None:
            rows.append({"name": row["name"], "pid": process.pid, "signaled": False})
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            rows.append({"name": row["name"], "pid": process.pid, "signaled": False})
            continue
        deadline = time.monotonic() + grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        forced = process.poll() is None
        if forced:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                forced = False
        rows.append(
            {
                "name": row["name"],
                "pid": process.pid,
                "signaled": True,
                "forced": forced,
            }
        )
    return rows


def execute_process_relaunch(
    *,
    approval_path: Path,
    expected_approval_sha256: str,
    publish: Callable[[Mapping[str, Any]], None],
    status_timeout_seconds: float = 30.0,
    rollback_grace_seconds: float = 10.0,
    source_verifier: Callable[..., Mapping[str, Any]] = verify_process_relaunch_manifest,
    process_scanner: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]]
    | None = None,
    launcher: Callable[[Mapping[str, Any]], ProcessHandle] = _launch_process,
    status_waiter: Callable[..., Mapping[str, Any]] = _wait_for_status,
    rollback: Callable[..., Sequence[Mapping[str, Any]]] = _rollback_owned_processes,
) -> dict[str, Any]:
    approval, approval_identity = _validate_approval(
        approval_path, expected_approval_sha256
    )
    readiness = assess_process_relaunch_readiness(
        manifest_path=Path(approval["manifest"]["path"]),
        expected_manifest_sha256=approval["manifest"]["sha256"],
        standing_authorization_path=Path(approval["standing_authorization"]["path"]),
        expected_standing_authorization_sha256=approval["standing_authorization"]["sha256"],
        formal_git=approval["formal_git"],
        implementation_git=approval["implementation_git"],
        expected_process_count=approval["expected_process_count"],
        source_verifier=source_verifier,
        process_scanner=process_scanner,
    )
    if (
        readiness["status"] != "ready"
        or readiness["selected_process_names"] != approval["selected_process_names"]
        or readiness["successful_process_names"] != approval["successful_process_names"]
    ):
        raise ValueError("process relaunch readiness changed after approval")
    manifest = _read_json(Path(approval["manifest"]["path"]), label="process relaunch manifest")
    by_name = {row["name"]: row for row in manifest["processes"]}
    selected = [by_name[name] for name in approval["selected_process_names"]]
    base = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "role": EXECUTION_ROLE,
        "started_at": _timestamp(),
        "approval": approval_identity,
        "manifest": approval["manifest"],
        "selected_process_names": approval["selected_process_names"],
        "successful_process_names": approval["successful_process_names"],
        "scope": dict(APPROVAL_SCOPE),
    }
    launched: list[tuple[Mapping[str, Any], ProcessHandle]] = []
    evidence = []
    publish({**base, "status": "launching", "launched": list(evidence), "rollback": []})
    try:
        for row in selected:
            launched_after_ns = time.time_ns()
            process = launcher(row)
            launched.append((row, process))
            observed = dict(
                status_waiter(
                    row,
                    process,
                    launched_after_ns=launched_after_ns,
                    timeout_seconds=status_timeout_seconds,
                )
            )
            evidence.append(observed)
            publish(
                {
                    **base,
                    "status": "launching",
                    "launched": list(evidence),
                    "rollback": [],
                }
            )
    except Exception as error:
        rollback_rows = [
            dict(value)
            for value in rollback(launched, grace_seconds=rollback_grace_seconds)
        ]
        report = {
            **base,
            "status": "failed",
            "failed_at": _timestamp(),
            "error_type": type(error).__name__,
            "detail": str(error),
            "launched": list(evidence),
            "rollback": rollback_rows,
        }
        publish(report)
        return report
    report = {
        **base,
        "status": "pass",
        "completed_at": _timestamp(),
        "launched": list(evidence),
        "rollback": [],
    }
    publish(report)
    return report
