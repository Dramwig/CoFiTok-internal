from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]

from cofitok.generation.capacity_probe_execution import (
    validate_standing_experiment_authorization,
)
from cofitok.generation_control_continuity import git_state
from cofitok.generation_control_process_snapshot import inspect_linux_process
from cofitok.inference_replay import file_identity


PLAN_SCHEMA_VERSION = 1
APPROVAL_SCHEMA_VERSION = 1
EXECUTION_SCHEMA_VERSION = 1
PLAN_ROLE = "generation_capacity_supervisor_migration_plan"
APPROVAL_ROLE = "generation_capacity_supervisor_migration_approval"
EXECUTION_ROLE = "generation_capacity_supervisor_migration_execution"

SUPERVISOR_SPECS = (
    {
        "name": "capacity_scaling_50k_execution",
        "role": "generation_capacity_scaling_50k_supervisor",
        "entrypoint": "scripts/run_generation_capacity_scaling_50k_supervisor.py",
        "waiting_detail": "waiting_for_source_replayed_capacity_scaling_decision",
        "barrier_options": ("--decision",),
        "lock_filename": None,
    },
    {
        "name": "capacity_completion_100k_execution",
        "role": "generation_capacity_completion_100k_supervisor",
        "entrypoint": "scripts/run_generation_capacity_completion_100k_supervisor.py",
        "waiting_detail": "waiting_for_source_replayed_capacity_completion_decision",
        "barrier_options": ("--decision",),
        "lock_filename": None,
    },
    {
        "name": "capacity_full_300k_readiness",
        "role": "capacity_full_300k_readiness_supervisor",
        "entrypoint": "scripts/run_generation_capacity_full_300k_readiness_supervisor.py",
        "waiting_detail": "waiting_for_source_bound_capacity_full_readiness_decision",
        "barrier_options": ("--decision",),
        "lock_filename": "capacity_full_300k_readiness_supervisor.lock",
    },
    {
        "name": "capacity_full_300k_training",
        "role": "capacity_full_300k_training_supervisor",
        "entrypoint": "scripts/run_generation_capacity_full_300k_training_supervisor.py",
        "waiting_detail": "waiting_for_passed_capacity_full_readiness",
        "barrier_options": ("--readiness",),
        "lock_filename": "capacity_full_300k_training_supervisor.lock",
    },
    {
        "name": "capacity_full_300k_posteval",
        "role": "capacity_full_300k_posteval_supervisor",
        "entrypoint": "scripts/run_generation_capacity_full_300k_posteval_supervisor.py",
        "waiting_detail": "waiting_for_passed_capacity_full_training",
        "barrier_options": ("--cofitok-training", "--dense-training"),
        "lock_filename": "capacity_full_300k_posteval_supervisor.lock",
    },
    {
        "name": "capacity_full_300k_finalization",
        "role": "capacity_full_300k_finalization_supervisor",
        "entrypoint": "scripts/run_generation_capacity_full_300k_finalization_supervisor.py",
        "waiting_detail": "waiting_for_capacity_full_final_gate",
        "barrier_options": (
            "--final-gate",
            "--completion-audit",
            "--release-receipt",
        ),
        "lock_filename": "capacity_full_300k_finalization_supervisor.lock",
    },
)
SUPERVISOR_NAMES = tuple(str(value["name"]) for value in SUPERVISOR_SPECS)
_SPEC_BY_NAME = {str(value["name"]): value for value in SUPERVISOR_SPECS}

PLAN_SCOPE = {
    "selected_owned_cpu_control_processes_only": True,
    "selected_process_names": list(SUPERVISOR_NAMES),
    "process_signals_sent": False,
    "processes_launched": False,
    "gpu_query_or_allocation_allowed": False,
    "direct_training_sampling_or_evaluation_launch_allowed": False,
    "formal_checkout_modification_allowed": False,
    "active_quality_bridge_or_capacity_probe_signaling_allowed": False,
    "unrelated_process_signaling_allowed": False,
}
APPROVAL_SCOPE = {
    "selected_owned_cpu_control_process_signaling_allowed": True,
    "selected_replacement_control_process_launch_allowed": True,
    "old_control_process_restore_on_failure_required": True,
    "new_partial_launch_rollback_required": True,
    "selected_process_names": list(SUPERVISOR_NAMES),
    "gpu_query_or_allocation_allowed": False,
    "direct_training_sampling_or_evaluation_launch_allowed": False,
    "formal_checkout_modification_allowed": False,
    "active_quality_bridge_or_capacity_probe_signaling_allowed": False,
    "unrelated_process_signaling_allowed": False,
}


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


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


def _git_state_with_porcelain(project: Path) -> dict[str, Any]:
    state = git_state(project.resolve())
    porcelain = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=project,
        check=True,
        capture_output=True,
    ).stdout
    return {
        **state,
        "porcelain_count": len(porcelain.splitlines()),
        "porcelain_sha256": hashlib.sha256(porcelain).hexdigest(),
    }


def _validate_expected_git(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    core = {
        "path": observed.get("path"),
        "revision": observed.get("revision"),
        "tree": observed.get("tree"),
        "branch": observed.get("branch"),
        "tracked_dirty": observed.get("tracked_dirty"),
    }
    if core != dict(expected):
        raise ValueError(f"{label} Git identity differs")
    if core["tracked_dirty"] is not False:
        raise ValueError(f"{label} has tracked changes")
    return dict(observed)


def _cli_value(argv: Sequence[str], option: str) -> str:
    matches = [index for index, value in enumerate(argv) if value == option]
    if len(matches) != 1 or matches[0] + 1 >= len(argv):
        raise ValueError(f"supervisor argv must contain exactly one {option}")
    value = argv[matches[0] + 1]
    if not value or value.startswith("--"):
        raise ValueError(f"supervisor argv value is missing for {option}")
    return value


def _entrypoint(runtime: Mapping[str, Any], expected: str) -> dict[str, Any]:
    argv = runtime["argv"]
    matches = [index for index, value in enumerate(argv) if value == expected]
    if len(matches) != 1:
        raise ValueError(f"supervisor entrypoint differs: {expected}")
    index = matches[0]
    raw = Path(expected)
    path = raw if raw.is_absolute() else Path(runtime["cwd"]) / raw
    return {
        "argument_index": index,
        "argument": expected,
        "identity": file_identity(path),
    }


def _status_snapshot(
    row: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(_cli_value(runtime["argv"], "--status-output")).resolve()
    payload = _read_json(path, label=f"status for {row['name']}")
    if (
        payload.get("role") != row["role"]
        or payload.get("status") != "waiting"
        or payload.get("detail") != row["waiting_detail"]
        or payload.get("pid") != runtime["pid"]
        or payload.get("child_pid") is not None
    ):
        raise ValueError(f"supervisor is not at the migration barrier: {row['name']}")
    return {
        "path": path.as_posix(),
        "identity": file_identity(path),
        "status": "waiting",
        "detail": row["waiting_detail"],
        "pid": runtime["pid"],
        "child_pid": None,
    }


def _barriers(
    row: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values = []
    for option in row["barrier_options"]:
        path = Path(_cli_value(runtime["argv"], str(option))).resolve()
        if path.exists():
            raise ValueError(
                f"supervisor trigger barrier already exists: {row['name']}:{path}"
            )
        values.append({"option": option, "path": path.as_posix(), "exists": False})
    return values


def _lock_path(row: Mapping[str, Any], *, status_path: Path) -> Path | None:
    filename = row.get("lock_filename")
    if filename is None:
        return None
    if not isinstance(filename, str) or not filename.endswith(".lock"):
        raise ValueError(f"invalid lifetime lock filename: {row['name']}")
    return (status_path.parent.parent / filename).resolve()


def inspect_linux_lifetime_lock(
    pid: int,
    path: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    if fcntl is None:
        raise RuntimeError("lifetime-lock inspection requires POSIX fcntl")
    target = path.resolve()
    if not target.is_file():
        raise ValueError(f"supervisor lifetime lock is missing: {target}")
    descriptors = []
    fd_root = proc_root / str(pid) / "fd"
    try:
        entries = list(fd_root.iterdir())
    except OSError as error:
        raise ProcessLookupError(pid) from error
    for entry in entries:
        try:
            if os.path.samefile(entry, target):
                descriptors.append(int(entry.name))
        except (FileNotFoundError, OSError, ValueError):
            continue
    if not descriptors:
        raise ValueError(f"supervisor does not hold lifetime-lock descriptor: {pid}")

    descriptor = os.open(target, os.O_WRONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise ValueError(f"supervisor lifetime lock is not contended: {pid}")
    finally:
        os.close(descriptor)
    stat = target.stat()
    return {
        "path": target.as_posix(),
        "identity": file_identity(target),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "owner_pid": pid,
        "owner_descriptors": sorted(descriptors),
        "exclusive_nonblocking_probe": "contended",
    }


def _validate_lock_identity(lock: Mapping[str, Any] | None) -> None:
    if lock is None:
        return
    path = Path(str(lock.get("path", ""))).resolve()
    if file_identity(path) != lock.get("identity"):
        raise ValueError(f"supervisor lifetime lock changed: {path}")
    stat = path.stat()
    if stat.st_dev != lock.get("device") or stat.st_ino != lock.get("inode"):
        raise ValueError(f"supervisor lifetime lock inode changed: {path}")


def scan_selected_session_members(
    selected_pids: Sequence[int],
    *,
    proc_root: Path = Path("/proc"),
) -> list[dict[str, int]]:
    selected = set(selected_pids)
    members = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid in selected:
            continue
        try:
            raw = (proc / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        end = raw.rfind(") ")
        fields = raw[end + 2 :].split() if end >= 0 else []
        if len(fields) < 4:
            continue
        ppid = int(fields[1])
        process_group_id = int(fields[2])
        session_id = int(fields[3])
        owners = sorted(
            selected.intersection({ppid, process_group_id, session_id})
        )
        if owners:
            members.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "process_group_id": process_group_id,
                    "session_id": session_id,
                    "selected_owner_pid": owners[0],
                }
            )
    return sorted(members, key=lambda value: value["pid"])


def _validate_runtime_barrier(
    row: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    pid = int(runtime["pid"])
    if (
        runtime["ppid"] != 1
        or runtime["process_group_id"] != pid
        or runtime["session_id"] != pid
        or runtime["stdin_target"] != "/dev/null"
    ):
        raise ValueError(f"supervisor is not an isolated detached process: {row['name']}")
    if not Path(runtime["stdout_target"]).is_absolute() or not Path(
        runtime["stderr_target"]
    ).is_absolute():
        raise ValueError(f"supervisor log target is not absolute: {row['name']}")


def _standing_authorization(path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    validate_standing_experiment_authorization(
        _read_json(path, label="standing experiment authorization")
    )
    return identity


def build_supervisor_migration_plan(
    *,
    selected_pids: Mapping[str, int],
    target_project: Path,
    expected_target_git: Mapping[str, Any],
    formal_project: Path,
    expected_formal_git: Mapping[str, Any],
    standing_authorization_path: Path,
    expected_standing_authorization_sha256: str,
    inspect_process: Callable[[int], Mapping[str, Any]] = inspect_linux_process,
    session_scanner: Callable[[Sequence[int]], Sequence[Mapping[str, Any]]]
    | None = None,
    git_inspector: Callable[[Path], Mapping[str, Any]] = _git_state_with_porcelain,
    lock_inspector: Callable[[int, Path], Mapping[str, Any]] = (
        inspect_linux_lifetime_lock
    ),
) -> dict[str, Any]:
    if set(selected_pids) != set(SUPERVISOR_NAMES):
        raise ValueError("migration must select exactly the six capacity supervisors")
    if any(type(pid) is not int or pid <= 0 for pid in selected_pids.values()):
        raise ValueError("migration PIDs must be positive integers")
    if len(set(selected_pids.values())) != len(SUPERVISOR_NAMES):
        raise ValueError("migration PIDs must be unique")

    target = target_project.resolve()
    formal = formal_project.resolve()
    target_git = _validate_expected_git(
        git_inspector(target), expected_target_git, label="migration target checkout"
    )
    if target_git["porcelain_count"] != 0:
        raise ValueError("migration target checkout must be fully clean")
    formal_git = _validate_expected_git(
        git_inspector(formal), expected_formal_git, label="formal checkout"
    )
    standing = _standing_authorization(
        standing_authorization_path,
        expected_standing_authorization_sha256,
    )

    old_rows = []
    new_rows = []
    for spec in SUPERVISOR_SPECS:
        name = str(spec["name"])
        runtime = dict(inspect_process(selected_pids[name]))
        _validate_runtime_barrier(spec, runtime)
        entrypoint = _entrypoint(runtime, str(spec["entrypoint"]))
        status = _status_snapshot(spec, runtime=runtime)
        barriers = _barriers(spec, runtime=runtime)
        lifetime_lock_path = _lock_path(
            spec,
            status_path=Path(status["path"]),
        )
        lifetime_lock = (
            dict(lock_inspector(int(runtime["pid"]), lifetime_lock_path))
            if lifetime_lock_path is not None
            else None
        )
        old_project = Path(runtime["cwd"]).resolve()
        old_git = dict(git_inspector(old_project))
        if old_git["tracked_dirty"] is not False:
            raise ValueError(f"old supervisor checkout has tracked changes: {name}")
        old_rows.append(
            {
                "name": name,
                "role": spec["role"],
                "waiting_detail": spec["waiting_detail"],
                "status_path": status["path"],
                "status": status,
                "barriers": barriers,
                "runtime": runtime,
                "relaunch_io": {
                    "stdin_target": "/dev/null",
                    "stdout_target": runtime["stdout_target"],
                    "stderr_target": runtime["stderr_target"],
                    "stdout_open_mode": "append",
                    "stderr_open_mode": "append",
                },
                "entrypoint": entrypoint,
                "lifetime_lock": lifetime_lock,
                "checkout_git": old_git,
            }
        )

        target_source = target / str(spec["entrypoint"])
        source_text = target_source.read_text(encoding="utf-8")
        if "publish_child_heartbeat" not in source_text:
            raise ValueError(f"target supervisor lacks heartbeat hardening: {name}")
        environment = dict(runtime["environment"])
        environment["PYTHONPATH"] = os.pathsep.join(
            (target.as_posix(), (target / "src").as_posix())
        )
        new_rows.append(
            {
                "name": name,
                "role": spec["role"],
                "waiting_detail": spec["waiting_detail"],
                "status_path": status["path"],
                "barriers": barriers,
                "runtime": {
                    "argv": list(runtime["argv"]),
                    "cwd": target.as_posix(),
                    "environment": environment,
                    "nice": runtime["nice"],
                    "umask": runtime["umask"],
                },
                "relaunch_io": {
                    "stdin_target": "/dev/null",
                    "stdout_target": runtime["stdout_target"],
                    "stderr_target": runtime["stderr_target"],
                    "stdout_open_mode": "append",
                    "stderr_open_mode": "append",
                },
                "entrypoint": {
                    "argument_index": entrypoint["argument_index"],
                    "argument": spec["entrypoint"],
                    "identity": file_identity(target_source),
                },
                "lifetime_lock": lifetime_lock,
            }
        )

    scanner = session_scanner or scan_selected_session_members
    session_members = [
        dict(value) for value in scanner(list(selected_pids.values()))
    ]
    if session_members:
        raise ValueError("selected supervisors have child or session-member processes")
    helper = target / "src/cofitok/process_monitoring.py"
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "role": PLAN_ROLE,
        "status": "ready",
        "detail": "six_waiting_supervisors_are_safe_to_migrate",
        "built_at": _timestamp(),
        "selected_process_names": list(SUPERVISOR_NAMES),
        "standing_authorization": standing,
        "target_git": target_git,
        "formal_git": formal_git,
        "heartbeat_helper": file_identity(helper),
        "old_processes": old_rows,
        "new_processes": new_rows,
        "session_members": [],
        "scope": dict(PLAN_SCOPE),
        "effects": {
            "processes_signaled": False,
            "processes_launched": False,
            "gpu_queried_or_allocated": False,
            "formal_checkout_modified": False,
        },
    }


def build_supervisor_migration_approval(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    identity = file_identity(plan_path)
    if identity["sha256"] != expected_plan_sha256:
        raise ValueError("supervisor migration plan SHA256 differs")
    plan = _read_json(plan_path, label="supervisor migration plan")
    old_names = [row.get("name") for row in plan.get("old_processes", [])]
    new_names = [row.get("name") for row in plan.get("new_processes", [])]
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("role") != PLAN_ROLE
        or plan.get("status") != "ready"
        or plan.get("selected_process_names") != list(SUPERVISOR_NAMES)
        or old_names != list(SUPERVISOR_NAMES)
        or new_names != list(SUPERVISOR_NAMES)
        or plan.get("session_members") != []
        or plan.get("scope") != PLAN_SCOPE
        or plan.get("effects")
        != {
            "processes_signaled": False,
            "processes_launched": False,
            "gpu_queried_or_allocated": False,
            "formal_checkout_modified": False,
        }
    ):
        raise ValueError("supervisor migration plan does not authorize execution")
    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "role": APPROVAL_ROLE,
        "status": "approved",
        "approved_at": _timestamp(),
        "plan": identity,
        "selected_process_names": list(SUPERVISOR_NAMES),
        "standing_authorization": plan["standing_authorization"],
        "target_git": plan["target_git"],
        "formal_git": plan["formal_git"],
        "scope": dict(APPROVAL_SCOPE),
    }


def _validate_plan_sources(plan: Mapping[str, Any]) -> None:
    if _git_state_with_porcelain(Path(plan["target_git"]["path"])) != plan["target_git"]:
        raise ValueError("migration target checkout changed after planning")
    if _git_state_with_porcelain(Path(plan["formal_git"]["path"])) != plan["formal_git"]:
        raise ValueError("formal checkout changed after planning")
    standing = plan["standing_authorization"]
    _standing_authorization(Path(standing["path"]), standing["sha256"])
    if file_identity(plan["heartbeat_helper"]["path"]) != plan["heartbeat_helper"]:
        raise ValueError("heartbeat helper changed after planning")
    for row in (*plan["old_processes"], *plan["new_processes"]):
        if file_identity(row["entrypoint"]["identity"]["path"]) != row["entrypoint"][
            "identity"
        ]:
            raise ValueError(f"supervisor entrypoint changed after planning: {row['name']}")
        _validate_lock_identity(row.get("lifetime_lock"))
    for row in plan["old_processes"]:
        if _git_state_with_porcelain(Path(row["checkout_git"]["path"])) != row[
            "checkout_git"
        ]:
            raise ValueError(f"old supervisor checkout changed after planning: {row['name']}")


def _validate_barriers(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        for barrier in row["barriers"]:
            if Path(barrier["path"]).exists():
                raise ValueError(
                    f"supervisor trigger appeared during migration: {row['name']}"
                )


def _validate_old_processes(
    plan: Mapping[str, Any],
    *,
    inspect_process: Callable[[int], Mapping[str, Any]],
    session_scanner: Callable[[Sequence[int]], Sequence[Mapping[str, Any]]],
    lock_inspector: Callable[[int, Path], Mapping[str, Any]],
) -> None:
    pids = []
    for row in plan["old_processes"]:
        expected = row["runtime"]
        pid = int(expected["pid"])
        observed = dict(inspect_process(pid))
        if observed != expected:
            raise ValueError(f"old supervisor process identity changed: {row['name']}")
        spec = _SPEC_BY_NAME[row["name"]]
        _status_snapshot(spec, runtime=observed)
        lock = row.get("lifetime_lock")
        if lock is not None:
            if dict(lock_inspector(pid, Path(lock["path"]))) != lock:
                raise ValueError(f"old supervisor lifetime lock changed: {row['name']}")
        pids.append(pid)
    _validate_barriers(plan["old_processes"])
    if list(session_scanner(pids)):
        raise ValueError("selected supervisors gained child or session-member processes")


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
    target_umask = runtime.get("umask")
    lifetime_lock = row.get("lifetime_lock")
    lock_descriptor: int | None = None
    if lifetime_lock is not None:
        if fcntl is None:
            raise RuntimeError("lifetime-lock launch requires POSIX fcntl")
        _validate_lock_identity(lifetime_lock)
        lock_descriptor = os.open(Path(lifetime_lock["path"]), os.O_WRONLY)
        try:
            fcntl.flock(
                lock_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BaseException:
            os.close(lock_descriptor)
            raise

    def prepare() -> None:
        if target_umask is not None:
            os.umask(int(str(target_umask), 8))
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
            pass_fds=(() if lock_descriptor is None else (lock_descriptor,)),
        )
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
    return process


def _wait_for_waiting_status(
    row: Mapping[str, Any],
    process: ProcessHandle,
    *,
    launched_after_ns: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    path = Path(row["status_path"])
    while time.monotonic() < deadline:
        code = process.poll()
        if path.is_file() and path.stat().st_mtime_ns >= launched_after_ns:
            payload = _read_json(path, label=f"migrated status for {row['name']}")
            if (
                payload.get("role") == row["role"]
                and payload.get("status") == "waiting"
                and payload.get("detail") == row["waiting_detail"]
                and payload.get("pid") == process.pid
                and payload.get("child_pid") is None
                and code is None
            ):
                lifetime_lock = row.get("lifetime_lock")
                lock_evidence = None
                if lifetime_lock is not None:
                    lock_evidence = inspect_linux_lifetime_lock(
                        process.pid,
                        Path(lifetime_lock["path"]),
                    )
                return {
                    "name": row["name"],
                    "role": row["role"],
                    "pid": process.pid,
                    "status": "waiting",
                    "detail": row["waiting_detail"],
                    "status_identity": file_identity(path),
                    "lifetime_lock": lock_evidence,
                }
        if code is not None:
            raise RuntimeError(
                f"migrated supervisor exited before waiting status: {row['name']}:{code}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"migrated supervisor status timed out: {row['name']}")


def _stop_old_process(
    row: Mapping[str, Any],
    *,
    grace_seconds: float,
    inspect_process: Callable[[int], Mapping[str, Any]] = inspect_linux_process,
    session_scanner: Callable[
        [Sequence[int]], Sequence[Mapping[str, Any]]
    ] = scan_selected_session_members,
    lock_inspector: Callable[[int, Path], Mapping[str, Any]] = (
        inspect_linux_lifetime_lock
    ),
) -> dict[str, Any]:
    runtime = row["runtime"]
    pid = int(runtime["pid"])
    observed = dict(inspect_process(pid))
    if observed != runtime:
        raise ValueError(f"old supervisor changed immediately before signal: {row['name']}")
    spec = _SPEC_BY_NAME[row["name"]]
    _validate_runtime_barrier(spec, observed)
    _status_snapshot(spec, runtime=observed)
    lock = row.get("lifetime_lock")
    if lock is not None and dict(lock_inspector(pid, Path(lock["path"]))) != lock:
        raise ValueError(f"old supervisor lifetime lock changed: {row['name']}")
    _validate_barriers((row,))
    if list(session_scanner((pid,))):
        raise ValueError(
            f"old supervisor gained a child or session member: {row['name']}"
        )
    if dict(inspect_process(pid)) != runtime:
        raise ValueError(f"old supervisor changed immediately before signal: {row['name']}")
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            observed = dict(inspect_process(pid))
        except ProcessLookupError:
            return {"name": row["name"], "pid": pid, "signal": "SIGTERM", "forced": False}
        if observed.get("start_ticks") != runtime["start_ticks"]:
            return {"name": row["name"], "pid": pid, "signal": "SIGTERM", "forced": False}
        time.sleep(0.1)
    try:
        observed = dict(inspect_process(pid))
    except ProcessLookupError:
        return {
            "name": row["name"],
            "pid": pid,
            "signal": "SIGTERM",
            "forced": False,
        }
    if observed != runtime:
        raise RuntimeError(f"old supervisor identity changed before SIGKILL: {row['name']}")
    os.killpg(pid, signal.SIGKILL)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            observed = dict(inspect_process(pid))
        except ProcessLookupError:
            return {"name": row["name"], "pid": pid, "signal": "SIGKILL", "forced": True}
        if observed.get("start_ticks") != runtime["start_ticks"]:
            return {"name": row["name"], "pid": pid, "signal": "SIGKILL", "forced": True}
        time.sleep(0.1)
    raise TimeoutError(f"old supervisor did not exit after SIGKILL: {row['name']}")


def _terminate_handle(
    row: Mapping[str, Any],
    process: ProcessHandle,
    *,
    grace_seconds: float,
) -> dict[str, Any]:
    if process.poll() is not None:
        return {"name": row["name"], "pid": process.pid, "signaled": False}
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"name": row["name"], "pid": process.pid, "signaled": False}
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    forced = process.poll() is None
    if forced:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            forced = False
    try:
        process.wait(timeout=grace_seconds)
    except (subprocess.TimeoutExpired, TypeError):
        pass
    return {
        "name": row["name"],
        "pid": process.pid,
        "signaled": True,
        "forced": forced,
    }


def _safe_publish(
    publish: Callable[[Mapping[str, Any]], None],
    payload: Mapping[str, Any],
    warnings: list[str],
) -> None:
    try:
        publish(payload)
    except Exception as error:
        warnings.append(f"{type(error).__name__}:{error}")


def execute_supervisor_migration(
    *,
    approval_path: Path,
    expected_approval_sha256: str,
    publish: Callable[[Mapping[str, Any]], None],
    stop_grace_seconds: float = 10.0,
    status_timeout_seconds: float = 30.0,
    inspect_process: Callable[[int], Mapping[str, Any]] = inspect_linux_process,
    session_scanner: Callable[[Sequence[int]], Sequence[Mapping[str, Any]]]
    | None = None,
    source_validator: Callable[[Mapping[str, Any]], None] = _validate_plan_sources,
    old_stopper: Callable[..., Mapping[str, Any]] = _stop_old_process,
    launcher: Callable[[Mapping[str, Any]], ProcessHandle] = _launch_process,
    status_waiter: Callable[..., Mapping[str, Any]] = _wait_for_waiting_status,
    terminator: Callable[..., Mapping[str, Any]] = _terminate_handle,
    lock_inspector: Callable[[int, Path], Mapping[str, Any]] = (
        inspect_linux_lifetime_lock
    ),
) -> dict[str, Any]:
    if stop_grace_seconds <= 0 or status_timeout_seconds <= 0:
        raise ValueError("migration timeouts must be positive")
    approval_identity = file_identity(approval_path)
    if approval_identity["sha256"] != expected_approval_sha256:
        raise ValueError("supervisor migration approval SHA256 differs")
    approval = _read_json(approval_path, label="supervisor migration approval")
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or approval.get("role") != APPROVAL_ROLE
        or approval.get("status") != "approved"
        or approval.get("selected_process_names") != list(SUPERVISOR_NAMES)
        or approval.get("scope") != APPROVAL_SCOPE
    ):
        raise ValueError("supervisor migration approval contract differs")
    plan_identity = file_identity(approval["plan"]["path"])
    if plan_identity != approval["plan"]:
        raise ValueError("supervisor migration plan changed after approval")
    plan = _read_json(Path(plan_identity["path"]), label="supervisor migration plan")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("role") != PLAN_ROLE
        or plan.get("status") != "ready"
        or plan.get("selected_process_names") != list(SUPERVISOR_NAMES)
        or [row.get("name") for row in plan.get("old_processes", [])]
        != list(SUPERVISOR_NAMES)
        or [row.get("name") for row in plan.get("new_processes", [])]
        != list(SUPERVISOR_NAMES)
        or plan.get("scope") != PLAN_SCOPE
        or approval.get("standing_authorization")
        != plan.get("standing_authorization")
        or approval.get("target_git") != plan.get("target_git")
        or approval.get("formal_git") != plan.get("formal_git")
    ):
        raise ValueError("supervisor migration plan contract differs")

    scanner = session_scanner or scan_selected_session_members
    source_validator(plan)
    _validate_old_processes(
        plan,
        inspect_process=inspect_process,
        session_scanner=scanner,
        lock_inspector=lock_inspector,
    )
    warnings: list[str] = []
    base = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "role": EXECUTION_ROLE,
        "started_at": _timestamp(),
        "approval": approval_identity,
        "plan": plan_identity,
        "selected_process_names": list(SUPERVISOR_NAMES),
        "scope": dict(APPROVAL_SCOPE),
    }
    old_by_name = {row["name"]: row for row in plan["old_processes"]}
    new_by_name = {row["name"]: row for row in plan["new_processes"]}
    stopped = []
    launched: list[tuple[Mapping[str, Any], ProcessHandle]] = []
    launched_evidence = []
    rollback = []
    restored = []
    phase = "stopping_old_supervisors"
    _safe_publish(
        publish,
        {
            **base,
            "status": "running",
            "phase": phase,
            "stopped": [],
            "launched": [],
            "rollback": [],
            "restored": [],
            "publish_warnings": [],
        },
        warnings,
    )
    try:
        for name in reversed(SUPERVISOR_NAMES):
            _validate_barriers(plan["old_processes"])
            result = dict(
                old_stopper(
                    old_by_name[name],
                    grace_seconds=stop_grace_seconds,
                    inspect_process=inspect_process,
                    session_scanner=scanner,
                    lock_inspector=lock_inspector,
                )
            )
            stopped.append(result)
            _safe_publish(
                publish,
                {
                    **base,
                    "status": "running",
                    "phase": phase,
                    "stopped": list(stopped),
                    "launched": [],
                    "rollback": [],
                    "restored": [],
                    "publish_warnings": list(warnings),
                },
                warnings,
            )
        _validate_barriers(plan["old_processes"])
        phase = "launching_hardened_supervisors"
        for name in SUPERVISOR_NAMES:
            row = new_by_name[name]
            launched_after_ns = time.time_ns()
            process = launcher(row)
            launched.append((row, process))
            evidence = dict(
                status_waiter(
                    row,
                    process,
                    launched_after_ns=launched_after_ns,
                    timeout_seconds=status_timeout_seconds,
                )
            )
            launched_evidence.append(evidence)
            _validate_barriers(plan["new_processes"])
            _safe_publish(
                publish,
                {
                    **base,
                    "status": "running",
                    "phase": phase,
                    "stopped": list(stopped),
                    "launched": list(launched_evidence),
                    "rollback": [],
                    "restored": [],
                    "publish_warnings": list(warnings),
                },
                warnings,
            )
    except Exception as error:
        phase = "rolling_back_new_and_restoring_old"
        for row, process in reversed(launched):
            rollback.append(
                dict(
                    terminator(
                        row,
                        process,
                        grace_seconds=stop_grace_seconds,
                    )
                )
            )
        stopped_names = {value["name"] for value in stopped}
        restoration_error = None
        restoration_handles: list[tuple[Mapping[str, Any], ProcessHandle]] = []
        try:
            for name in SUPERVISOR_NAMES:
                if name not in stopped_names:
                    continue
                row = old_by_name[name]
                launched_after_ns = time.time_ns()
                process = launcher(row)
                restoration_handles.append((row, process))
                restored.append(
                    dict(
                        status_waiter(
                            row,
                            process,
                            launched_after_ns=launched_after_ns,
                            timeout_seconds=status_timeout_seconds,
                        )
                    )
                )
        except Exception as restore_error:
            restored_names = {value["name"] for value in restored}
            for row, process in reversed(restoration_handles):
                if row["name"] in restored_names:
                    continue
                rollback.append(
                    dict(
                        terminator(
                            row,
                            process,
                            grace_seconds=stop_grace_seconds,
                        )
                    )
                )
            restoration_error = {
                "error_type": type(restore_error).__name__,
                "detail": str(restore_error),
            }
        status = "failed_restored" if restoration_error is None else "failed_restore_incomplete"
        report = {
            **base,
            "status": status,
            "phase": phase,
            "failed_at": _timestamp(),
            "error_type": type(error).__name__,
            "detail": str(error),
            "stopped": stopped,
            "launched": launched_evidence,
            "rollback": rollback,
            "restored": restored,
            "restoration_error": restoration_error,
            "publish_warnings": list(warnings),
        }
        _safe_publish(publish, report, warnings)
        report["publish_warnings"] = list(warnings)
        return report

    report = {
        **base,
        "status": "pass",
        "phase": "completed",
        "completed_at": _timestamp(),
        "stopped": stopped,
        "launched": launched_evidence,
        "rollback": [],
        "restored": [],
        "publish_warnings": list(warnings),
    }
    _safe_publish(publish, report, warnings)
    report["publish_warnings"] = list(warnings)
    return report
