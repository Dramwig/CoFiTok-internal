from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation_control_continuity import git_state
from cofitok.inference_replay import file_identity


MANIFEST_SCHEMA_VERSION = 1
VERIFICATION_SCHEMA_VERSION = 1
MANIFEST_ROLE = "generation_control_plane_process_relaunch_manifest"
VERIFICATION_ROLE = "generation_control_plane_process_relaunch_verification"
LINEAGE_ROLE = "capacity_generation_pipeline_lineage_observer"
SAFE_ENVIRONMENT_KEYS = (
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PATH",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
)
AUTHORIZATION_BOUNDARY = {
    "capture_live_process_metadata_allowed": True,
    "persist_data_only_relaunch_recipe_allowed": True,
    "verify_static_relaunch_sources_allowed": True,
    "process_launch_allowed": False,
    "process_signaling_allowed": False,
    "gpu_query_or_allocation_allowed": False,
    "experiment_output_modification_allowed": False,
    "formal_checkout_modification_allowed": False,
    "relaunch_authorization_created": False,
}
RELAUNCH_POLICY = {
    "commands_are_inert_json_data": True,
    "automatic_relaunch_allowed_by_manifest": False,
    "all_original_processes_must_be_absent_before_future_relaunch": True,
    "fresh_static_source_verification_required_before_future_relaunch": True,
    "separate_execution_receipt_required_before_future_relaunch": True,
    "future_stdout_and_stderr_must_append_to_recorded_targets": True,
}
EFFECTS = {
    "processes_launched": False,
    "processes_signaled": False,
    "gpu_queried_or_allocated": False,
    "experiment_outputs_modified": False,
    "formal_checkout_modified": False,
    "control_metadata_capture_only": True,
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUNTIME_KEYS = {
    "pid",
    "ppid",
    "process_group_id",
    "session_id",
    "nice",
    "start_ticks",
    "umask",
    "cwd",
    "argv",
    "executable_target",
    "stdin_target",
    "stdout_target",
    "stderr_target",
    "stdout_fd_flags",
    "stderr_fd_flags",
    "environment",
}
_PROCESS_KEYS = {
    "index",
    "name",
    "role",
    "blocking",
    "status_path",
    "status_snapshot",
    "runtime",
    "relaunch_io",
    "entrypoint",
    "checkout_git",
}
_MANIFEST_KEYS = {
    "schema_version",
    "role",
    "status",
    "complete",
    "captured_at",
    "hostname",
    "builder_git",
    "static_continuity_manifest",
    "lineage_snapshot",
    "process_count",
    "processes",
    "relaunch_policy",
    "authorization_boundary",
    "effects",
}


def _timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("process snapshot timestamp must be timezone-aware")
    return current.astimezone(timezone.utc).isoformat()


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute()


def _json_snapshot(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = path.resolve()
    try:
        payload_bytes = source.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, {
        "path": source.as_posix(),
        "bytes": len(payload_bytes),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }


def _run_git(cwd: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise ValueError(f"Git source inspection failed at {cwd}: {detail}") from error
    return result.stdout.strip()


def _checkout_state(cwd: Path) -> dict[str, Any]:
    top = Path(_run_git(cwd, "rev-parse", "--show-toplevel")).resolve()
    state = git_state(top)
    if state["tracked_dirty"] is not False:
        raise ValueError(f"process checkout has tracked changes: {top}")
    return state


def _fd_flags(proc: Path, descriptor: int) -> int | None:
    try:
        lines = (proc / f"fdinfo/{descriptor}").read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, value = line.partition(":")
        if key == "flags" and separator:
            return int(value.strip(), 8)
    return None


def _umask(proc: Path) -> str | None:
    try:
        lines = (proc / "status").read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, value = line.partition(":")
        if key == "Umask" and separator:
            return value.strip()
    return None


def inspect_linux_process(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    if type(pid) is not int or pid <= 0:
        raise ValueError("process PID must be a positive integer")
    proc = proc_root / str(pid)
    if not proc.is_dir():
        raise ProcessLookupError(f"process does not exist: {pid}")
    try:
        argv = [
            value.decode("utf-8", errors="surrogateescape")
            for value in (proc / "cmdline").read_bytes().split(b"\0")
            if value
        ]
        environment_values = {}
        for value in (proc / "environ").read_bytes().split(b"\0"):
            if not value or b"=" not in value:
                continue
            raw_key, raw_value = value.split(b"=", 1)
            key = raw_key.decode("utf-8", errors="surrogateescape")
            if key in SAFE_ENVIRONMENT_KEYS:
                environment_values[key] = raw_value.decode(
                    "utf-8", errors="surrogateescape"
                )
        raw_stat = (proc / "stat").read_text(encoding="utf-8", errors="strict")
        end = raw_stat.rfind(") ")
        if end < 0:
            raise ValueError("Linux process stat record is malformed")
        fields = raw_stat[end + 2 :].split()
        if len(fields) < 20:
            raise ValueError("Linux process stat record is truncated")
        runtime = {
            "pid": pid,
            "ppid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "session_id": int(fields[3]),
            "nice": int(fields[16]),
            "start_ticks": int(fields[19]),
            "umask": _umask(proc),
            "cwd": os.readlink(proc / "cwd"),
            "argv": argv,
            "executable_target": os.readlink(proc / "exe"),
            "stdin_target": os.readlink(proc / "fd/0"),
            "stdout_target": os.readlink(proc / "fd/1"),
            "stderr_target": os.readlink(proc / "fd/2"),
            "stdout_fd_flags": _fd_flags(proc, 1),
            "stderr_fd_flags": _fd_flags(proc, 2),
            "environment": dict(sorted(environment_values.items())),
        }
    except OSError as error:
        raise ProcessLookupError(f"process changed during inspection: {pid}") from error
    return _validate_runtime(runtime, expected_pid=pid)


def _validate_runtime(value: Mapping[str, Any], *, expected_pid: int) -> dict[str, Any]:
    if set(value) != _RUNTIME_KEYS:
        raise ValueError(f"process runtime schema differs for PID {expected_pid}")
    runtime = dict(value)
    integer_keys = (
        "pid",
        "ppid",
        "process_group_id",
        "session_id",
        "nice",
        "start_ticks",
    )
    if any(type(runtime[key]) is not int for key in integer_keys):
        raise ValueError(f"process runtime integers are malformed for PID {expected_pid}")
    if runtime["pid"] != expected_pid or runtime["start_ticks"] <= 0:
        raise ValueError(f"process runtime identity differs for PID {expected_pid}")
    if not isinstance(runtime["argv"], list) or len(runtime["argv"]) < 2:
        raise ValueError(f"process argv is missing for PID {expected_pid}")
    if any(not isinstance(item, str) or not item for item in runtime["argv"]):
        raise ValueError(f"process argv is malformed for PID {expected_pid}")
    for key in (
        "cwd",
        "executable_target",
        "stdin_target",
        "stdout_target",
        "stderr_target",
    ):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f"process {key} is malformed for PID {expected_pid}")
    if not _is_absolute(runtime["cwd"]):
        raise ValueError(f"process CWD is not absolute for PID {expected_pid}")
    for key in ("stdout_fd_flags", "stderr_fd_flags"):
        if runtime[key] is not None and type(runtime[key]) is not int:
            raise ValueError(f"process FD flags are malformed for PID {expected_pid}")
    if runtime["umask"] is not None and not isinstance(runtime["umask"], str):
        raise ValueError(f"process umask is malformed for PID {expected_pid}")
    environment = runtime["environment"]
    if not isinstance(environment, dict) or any(
        key not in SAFE_ENVIRONMENT_KEYS
        or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError(f"process safe environment differs for PID {expected_pid}")
    return runtime


def _status_pid(payload: Mapping[str, Any], *, observer: bool) -> int | None:
    if observer:
        value = payload.get("observer")
        return value.get("pid") if isinstance(value, Mapping) else None
    raw = payload.get("pid", payload.get("supervisor_pid"))
    return raw if type(raw) is int else None


def _entrypoint(runtime: Mapping[str, Any]) -> dict[str, Any]:
    argv = runtime["argv"]
    matches = [
        (index, value)
        for index, value in enumerate(argv[1:], start=1)
        if value.endswith((".py", ".sh"))
    ]
    if not matches:
        raise ValueError(f"process entrypoint is missing for PID {runtime['pid']}")
    index, argument = matches[0]
    raw = Path(argument)
    path = raw if raw.is_absolute() else Path(runtime["cwd"]) / raw
    return {
        "argument_index": index,
        "argument": argument,
        "identity": file_identity(path),
    }


def _process_row(
    *,
    spec: Mapping[str, Any],
    status_payload: Mapping[str, Any],
    status_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    observer: bool,
) -> dict[str, Any]:
    pid = int(spec["pid"])
    observed = _validate_runtime(runtime, expected_pid=pid)
    if observed["stdin_target"] != "/dev/null":
        raise ValueError(f"process stdin is not detached for {spec['name']}")
    for key in ("stdout_target", "stderr_target"):
        if not _is_absolute(observed[key]):
            raise ValueError(f"process log target is not absolute for {spec['name']}")
    expected_role = str(spec["role"])
    if status_payload.get("role") != expected_role:
        raise ValueError(f"status role differs for process {spec['name']}")
    if _status_pid(status_payload, observer=observer) != pid:
        raise ValueError(f"status PID differs for process {spec['name']}")
    cwd = Path(observed["cwd"])
    return {
        "index": int(spec["index"]),
        "name": str(spec["name"]),
        "role": expected_role,
        "blocking": bool(spec["blocking"]),
        "status_path": Path(spec["status_path"]).resolve().as_posix(),
        "status_snapshot": {
            "identity": dict(status_identity),
            "status": status_payload.get("status"),
            "detail": status_payload.get("detail"),
            "updated_at": status_payload.get(
                "updated_at", status_payload.get("observed_at")
            ),
            "pid": pid,
        },
        "runtime": observed,
        "relaunch_io": {
            "stdin_target": "/dev/null",
            "stdout_target": observed["stdout_target"],
            "stderr_target": observed["stderr_target"],
            "stdout_open_mode": "append",
            "stderr_open_mode": "append",
        },
        "entrypoint": _entrypoint(observed),
        "checkout_git": _checkout_state(cwd),
    }


def build_process_relaunch_manifest(
    *,
    lineage_report_path: Path,
    static_continuity_manifest_path: Path,
    expected_static_continuity_sha256: str,
    builder_git: Mapping[str, Any],
    expected_process_count: int,
    inspect_process: Callable[[int], Mapping[str, Any]] = inspect_linux_process,
    captured_at: datetime | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    if type(expected_process_count) is not int or expected_process_count <= 0:
        raise ValueError("expected process count must be positive")
    if not _SHA256.fullmatch(expected_static_continuity_sha256):
        raise ValueError("static continuity manifest SHA256 is malformed")
    static_identity = file_identity(static_continuity_manifest_path)
    if static_identity["sha256"] != expected_static_continuity_sha256:
        raise ValueError("static continuity manifest SHA256 differs")
    builder = dict(builder_git)
    if set(builder) != {"path", "revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError("process snapshot builder Git schema differs")
    if builder["tracked_dirty"] is not False:
        raise ValueError("process snapshot builder checkout has tracked changes")

    lineage, lineage_identity = _json_snapshot(
        lineage_report_path, label="capacity pipeline lineage report"
    )
    stages = lineage.get("stages")
    observer = lineage.get("observer")
    if (
        lineage.get("role") != LINEAGE_ROLE
        or lineage.get("issues") != []
        or not isinstance(stages, list)
        or not isinstance(observer, Mapping)
        or type(observer.get("pid")) is not int
    ):
        raise ValueError("capacity pipeline lineage report is not healthy")
    if len(stages) + 1 != expected_process_count:
        raise ValueError("capacity pipeline lineage process count differs")

    processes = []
    seen_pids: set[int] = set()
    seen_names: set[str] = set()
    for position, stage in enumerate(stages):
        if (
            not isinstance(stage, Mapping)
            or stage.get("index") != position
            or stage.get("issues") != []
            or stage.get("process_alive") is not True
            or type(stage.get("pid")) is not int
            or stage.get("role") != stage.get("expected_role")
        ):
            raise ValueError(f"capacity pipeline stage {position} is not healthy")
        pid = int(stage["pid"])
        name = stage.get("name")
        if not isinstance(name, str) or not name or pid in seen_pids or name in seen_names:
            raise ValueError(f"capacity pipeline stage {position} identity is duplicated")
        seen_pids.add(pid)
        seen_names.add(name)
        status_path = Path(str(stage["status_path"]))
        status_payload, status_identity = _json_snapshot(
            status_path, label=f"status for {name}"
        )
        processes.append(
            _process_row(
                spec={
                    "index": position,
                    "name": name,
                    "role": stage["expected_role"],
                    "blocking": stage["blocking"],
                    "status_path": status_path,
                    "pid": pid,
                },
                status_payload=status_payload,
                status_identity=status_identity,
                runtime=inspect_process(pid),
                observer=False,
            )
        )

    observer_pid = int(observer["pid"])
    if observer_pid in seen_pids:
        raise ValueError("lineage observer PID duplicates a stage PID")
    processes.append(
        _process_row(
            spec={
                "index": len(stages),
                "name": "capacity_pipeline_lineage_observer",
                "role": LINEAGE_ROLE,
                "blocking": False,
                "status_path": lineage_report_path,
                "pid": observer_pid,
            },
            status_payload=lineage,
            status_identity=lineage_identity,
            runtime=inspect_process(observer_pid),
            observer=True,
        )
    )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "role": MANIFEST_ROLE,
        "status": "pass",
        "complete": True,
        "captured_at": _timestamp(captured_at),
        "hostname": hostname or socket.gethostname(),
        "builder_git": builder,
        "static_continuity_manifest": static_identity,
        "lineage_snapshot": {
            "identity": lineage_identity,
            "status": lineage.get("status"),
            "detail": lineage.get("detail"),
            "observed_at": lineage.get("observed_at"),
            "issues": lineage.get("issues"),
        },
        "process_count": len(processes),
        "processes": processes,
        "relaunch_policy": dict(RELAUNCH_POLICY),
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
        "effects": dict(EFFECTS),
    }


def _validate_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(payload) != _MANIFEST_KEYS:
        raise ValueError("process relaunch manifest schema differs")
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("role") != MANIFEST_ROLE
        or payload.get("status") != "pass"
        or payload.get("complete") is not True
        or payload.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or payload.get("relaunch_policy") != RELAUNCH_POLICY
        or payload.get("effects") != EFFECTS
    ):
        raise ValueError("process relaunch manifest identity differs")
    rows = payload.get("processes")
    count = payload.get("process_count")
    if not isinstance(rows, list) or type(count) is not int or len(rows) != count:
        raise ValueError("process relaunch manifest process count differs")
    names: set[str] = set()
    pids: set[int] = set()
    normalized = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _PROCESS_KEYS:
            raise ValueError(f"process relaunch row {index} schema differs")
        row = dict(raw)
        runtime = _validate_runtime(row["runtime"], expected_pid=row["runtime"]["pid"])
        expected_io = {
            "stdin_target": "/dev/null",
            "stdout_target": runtime["stdout_target"],
            "stderr_target": runtime["stderr_target"],
            "stdout_open_mode": "append",
            "stderr_open_mode": "append",
        }
        if row["relaunch_io"] != expected_io:
            raise ValueError(f"process relaunch I/O differs for row {index}")
        if row["index"] != index or row["name"] in names or runtime["pid"] in pids:
            raise ValueError(f"process relaunch row {index} identity differs")
        if not isinstance(row["role"], str) or not row["role"]:
            raise ValueError(f"process relaunch row {index} role differs")
        if type(row["blocking"]) is not bool:
            raise ValueError(f"process relaunch row {index} blocking flag differs")
        if not isinstance(row["status_path"], str) or not _is_absolute(row["status_path"]):
            raise ValueError(f"process relaunch row {index} status path differs")
        names.add(row["name"])
        pids.add(runtime["pid"])
        normalized.append(row)
    return normalized


def verify_process_relaunch_manifest(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_process_count: int,
    require_live: bool,
    inspect_process: Callable[[int], Mapping[str, Any]] = inspect_linux_process,
) -> dict[str, Any]:
    if not _SHA256.fullmatch(expected_manifest_sha256):
        raise ValueError("process relaunch manifest SHA256 is malformed")
    manifest, manifest_identity = _json_snapshot(
        manifest_path, label="process relaunch manifest"
    )
    if manifest_identity["sha256"] != expected_manifest_sha256:
        raise ValueError("process relaunch manifest SHA256 differs")
    rows = _validate_manifest(manifest)
    if len(rows) != expected_process_count:
        raise ValueError("expected process relaunch count differs")

    static_identity = file_identity(manifest["static_continuity_manifest"]["path"])
    if static_identity != manifest["static_continuity_manifest"]:
        raise ValueError("static continuity manifest identity differs")
    builder_state = git_state(Path(manifest["builder_git"]["path"]))
    if builder_state != manifest["builder_git"]:
        raise ValueError("process snapshot builder Git state differs")

    live_verified = 0
    for row in rows:
        runtime = row["runtime"]
        entrypoint = row["entrypoint"]
        if not isinstance(entrypoint, Mapping) or set(entrypoint) != {
            "argument_index",
            "argument",
            "identity",
        }:
            raise ValueError(f"entrypoint schema differs for {row['name']}")
        index = entrypoint["argument_index"]
        if (
            type(index) is not int
            or index <= 0
            or index >= len(runtime["argv"])
            or runtime["argv"][index] != entrypoint["argument"]
        ):
            raise ValueError(f"entrypoint argv binding differs for {row['name']}")
        raw = Path(entrypoint["argument"])
        source = raw if raw.is_absolute() else Path(runtime["cwd"]) / raw
        if file_identity(source) != entrypoint["identity"]:
            raise ValueError(f"entrypoint identity differs for {row['name']}")
        checkout = git_state(Path(row["checkout_git"]["path"]))
        if checkout != row["checkout_git"]:
            raise ValueError(f"checkout Git state differs for {row['name']}")

        if require_live:
            current = _validate_runtime(
                inspect_process(runtime["pid"]), expected_pid=runtime["pid"]
            )
            if current != runtime:
                raise ValueError(f"live process runtime differs for {row['name']}")
            status, _ = _json_snapshot(
                Path(row["status_path"]), label=f"live status for {row['name']}"
            )
            observer = row["name"] == "capacity_pipeline_lineage_observer"
            if (
                status.get("role") != row["role"]
                or _status_pid(status, observer=observer) != runtime["pid"]
            ):
                raise ValueError(f"live status identity differs for {row['name']}")
            live_verified += 1

    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "role": VERIFICATION_ROLE,
        "status": "pass",
        "complete": True,
        "verified_at": _timestamp(),
        "manifest": manifest_identity,
        "expected_process_count": expected_process_count,
        "source_verified_process_count": len(rows),
        "live_verification_required": require_live,
        "live_verified_process_count": live_verified,
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
        "effects": dict(EFFECTS),
    }
