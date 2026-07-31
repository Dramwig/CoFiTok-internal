from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, git_provenance, write_json_report


SCHEMA_VERSION = 1
ROLE = "generation_stage_receipt"
WORKER_ROLE = "generation_stage_worker_result"
REPLAY_ERROR_EXIT_CODE = 86
WORKER_FAILURE_EXIT_CODE = 87
WORKER_SIDECAR_GRACE_SECONDS = 5.0


class StageReplayError(ValueError):
    """Existing stage evidence cannot be safely reused or replaced."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise StageReplayError(f"generation stage receipt is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise StageReplayError(f"{path} must contain a JSON object")
    return payload


def _absolute_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(expanded))


def _reject_symlink(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise StageReplayError(
                f"stage evidence path must not contain a symlink: {current}"
            )
        if current.parent == current:
            return
        current = current.parent


def _canonical_path(path: str | Path) -> Path:
    absolute = _absolute_path(path)
    _reject_symlink(absolute)
    return absolute.resolve()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = _canonical_path(path)
    _reject_symlink(resolved)
    if not resolved.is_file():
        raise FileNotFoundError(f"stage evidence file is missing: {resolved}")
    return {
        "kind": "file",
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def tree_identity(path: str | Path) -> dict[str, Any]:
    resolved = _canonical_path(path)
    _reject_symlink(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(f"stage evidence directory is missing: {resolved}")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise StageReplayError(
                f"stage evidence tree contains a symlink: {candidate}"
            )
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(resolved).as_posix()
        size = candidate.stat().st_size
        sha256 = file_sha256(candidate)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += size
    return {
        "kind": "tree",
        "path": resolved.as_posix(),
        "file_count": file_count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _identities(files: list[str], trees: list[str]) -> list[dict[str, Any]]:
    identities = [file_identity(path) for path in files]
    identities.extend(tree_identity(path) for path in trees)
    paths = [identity["path"] for identity in identities]
    if len(set(paths)) != len(paths):
        raise ValueError("stage evidence paths must be unique")
    return sorted(
        identities,
        key=lambda identity: (identity["path"], identity["kind"]),
    )


def _output_declarations(
    files: list[str],
    trees: list[str],
) -> list[dict[str, str]]:
    declarations = [
        {"kind": "file", "path": _canonical_path(path).as_posix()}
        for path in files
    ]
    declarations.extend(
        {"kind": "tree", "path": _canonical_path(path).as_posix()}
        for path in trees
    )
    paths = [Path(declaration["path"]) for declaration in declarations]
    if len(set(paths)) != len(paths):
        raise ValueError("stage output paths must be unique")
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError("stage output declarations must not overlap")
    return sorted(declarations, key=lambda declaration: declaration["path"])


def _output_identities(
    declarations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    identities = []
    for declaration in declarations:
        identity = (
            file_identity(declaration["path"])
            if declaration["kind"] == "file"
            else tree_identity(declaration["path"])
        )
        identities.append(identity)
    return sorted(
        identities,
        key=lambda identity: (identity["path"], identity["kind"]),
    )


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _require_outputs_absent(declarations: list[dict[str, str]]) -> None:
    existing = [
        declaration["path"]
        for declaration in declarations
        if _path_exists(Path(declaration["path"]))
    ]
    if existing:
        raise StageReplayError(
            "unreceipted stage outputs already exist: " + ", ".join(existing)
        )


def _archive_outputs(
    declarations: list[dict[str, str]],
    *,
    attempt: int,
    reason: str,
) -> list[dict[str, Any]]:
    archived = []
    for declaration in declarations:
        source = Path(declaration["path"])
        if not _path_exists(source):
            continue
        suffix = f".stage-attempt-{attempt:03d}-{reason}"
        target = source.with_name(source.name + suffix)
        counter = 1
        while _path_exists(target):
            target = source.with_name(source.name + suffix + f"-{counter}")
            counter += 1
        os.replace(source, target)
        archived.append(
            file_identity(target)
            if declaration["kind"] == "file"
            else tree_identity(target)
        )
    return archived


def _process_exists(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno in {errno.ESRCH, errno.EINVAL} or getattr(error, "winerror", None) == 87:
            return False
        raise
    return True


def _optional_pid(value: Any, *, name: str) -> int:
    if value is None:
        return -1
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageReplayError(f"generation stage {name} is malformed")
    return value


def _wait_for_process(pid: int, *, poll_seconds: float = 1.0) -> None:
    while _process_exists(pid):
        time.sleep(poll_seconds)


def _wait_for_path(
    path: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(poll_seconds)
    return path.is_file()


def _request(
    *,
    project: Path,
    cwd: Path,
    command: list[str],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, str]],
) -> dict[str, Any]:
    if not command:
        raise ValueError("stage command must not be empty")
    return {
        "project": project.as_posix(),
        "cwd": cwd.as_posix(),
        "command": command,
        "git": git_provenance(project),
        "inputs": inputs,
        "outputs": outputs,
    }


def _validate_state_location(
    state_path: Path,
    outputs: list[dict[str, str]],
) -> None:
    for declaration in outputs:
        output = Path(declaration["path"])
        if state_path == output or (
            declaration["kind"] == "tree" and state_path.is_relative_to(output)
        ):
            raise ValueError("stage receipt must live outside declared outputs")


def _request_sha256(request: dict[str, Any]) -> str:
    payload = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _worker_result_path(state_path: Path, attempt: int) -> Path:
    return state_path.with_name(
        f"{state_path.name}.attempt-{attempt:03d}.worker-result.json"
    )


def _validate_worker_result(
    path: Path,
    *,
    expected_attempt: int,
    expected_request_sha256: str,
    expected_command: list[str],
) -> dict[str, Any]:
    result = _read_object(path)
    if (
        result.get("schema_version") != SCHEMA_VERSION
        or result.get("role") != WORKER_ROLE
        or int(result.get("attempt", -1)) != expected_attempt
        or result.get("request_sha256") != expected_request_sha256
        or result.get("command") != expected_command
        or result.get("status") not in {"running", "completed", "launcher_failed"}
        or not isinstance(result.get("exit_code"), int)
    ):
        raise StageReplayError("generation stage worker result contract differs")
    return result


def _completed_state(
    *,
    running_state: dict[str, Any],
    attempts: list[dict[str, Any]],
    completed_outputs: list[dict[str, Any]],
    worker_result: dict[str, Any],
    worker_result_path: Path,
    recovered: bool,
) -> dict[str, Any]:
    return {
        **running_state,
        "status": "completed",
        "child_pid": None,
        "attempts": attempts,
        "completed_outputs": completed_outputs,
        "worker_result": file_identity(worker_result_path),
        "worker_exit_code": int(worker_result["exit_code"]),
        "recovered_after_parent_interruption": recovered,
        "completed_at": _utc_now(),
        "updated_at": _utc_now(),
    }


def _run_worker(
    *,
    result_path: Path,
    cwd: Path,
    attempt: int,
    request_sha256: str,
    command: list[str],
) -> int:
    result_path = _canonical_path(result_path)
    cwd = _canonical_path(cwd)
    if result_path.exists() or result_path.is_symlink():
        raise StageReplayError(
            f"generation stage worker result already exists: {result_path}"
        )
    base = {
        "schema_version": SCHEMA_VERSION,
        "role": WORKER_ROLE,
        "attempt": attempt,
        "request_sha256": request_sha256,
        "command": command,
        "cwd": cwd.as_posix(),
        "worker_pid": os.getpid(),
    }
    try:
        child = subprocess.Popen(command, cwd=cwd)
        child_pid = child.pid
        write_json_report(
            result_path,
            {
                **base,
                "status": "running",
                "child_pid": child_pid,
                "exit_code": -1,
                "error_type": None,
                "error": None,
                "started_at": _utc_now(),
            },
        )
        exit_code = child.wait()
        status = "completed"
        error_type = None
        error = None
    except BaseException as failure:
        child_pid = None
        exit_code = WORKER_FAILURE_EXIT_CODE
        status = "launcher_failed"
        error_type = type(failure).__name__
        error = str(failure)
    write_json_report(
        result_path,
        {
            **base,
            "status": status,
            "child_pid": child_pid,
            "exit_code": exit_code,
            "error_type": error_type,
            "error": error,
            "finished_at": _utc_now(),
        },
    )
    return 0


def run_stage_once(
    *,
    state_path: Path,
    project: Path,
    cwd: Path,
    command: list[str],
    input_files: list[str],
    input_trees: list[str],
    output_files: list[str],
    output_trees: list[str],
) -> dict[str, Any]:
    project = _canonical_path(project)
    cwd = _canonical_path(cwd)
    state_path = _canonical_path(state_path)
    if state_path.exists() and not state_path.is_file():
        raise StageReplayError(
            f"generation stage receipt path is not a file: {state_path}"
        )
    try:
        inputs = _identities(input_files, input_trees)
    except (FileNotFoundError, StageReplayError) as error:
        raise StageReplayError(
            f"generation stage inputs are missing or invalid: {error}"
        ) from error
    outputs = _output_declarations(output_files, output_trees)
    if not outputs:
        raise ValueError("stage must declare at least one output")
    _validate_state_location(state_path, outputs)
    request = _request(
        project=project,
        cwd=cwd,
        command=command,
        inputs=inputs,
        outputs=outputs,
    )
    request_sha256 = _request_sha256(request)
    attempts: list[dict[str, Any]] = []
    if state_path.is_file():
        state = _read_object(state_path)
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("role") != ROLE
            or state.get("request") != request
        ):
            raise StageReplayError("existing generation stage receipt request differs")
        raw_attempts = state.get("attempts", [])
        if not isinstance(raw_attempts, list):
            raise StageReplayError("generation stage receipt attempts are malformed")
        attempts = list(raw_attempts)
        if state.get("status") == "completed":
            try:
                actual_outputs = _output_identities(outputs)
            except (FileNotFoundError, StageReplayError) as error:
                raise StageReplayError(
                    "completed generation stage outputs are missing or invalid"
                ) from error
            if state.get("completed_outputs") != actual_outputs:
                raise StageReplayError(
                    "completed generation stage outputs differ from receipt"
                )
            completed_attempt = len(attempts)
            worker_result_path = _worker_result_path(
                state_path,
                completed_attempt,
            )
            worker_identity = state.get("worker_result")
            if (
                completed_attempt < 1
                or not isinstance(worker_identity, dict)
                or worker_identity != file_identity(worker_result_path)
            ):
                raise StageReplayError(
                    "completed generation stage worker result differs from receipt"
                )
            worker_result = _validate_worker_result(
                worker_result_path,
                expected_attempt=completed_attempt,
                expected_request_sha256=request_sha256,
                expected_command=command,
            )
            if (
                worker_result.get("status") != "completed"
                or int(worker_result.get("exit_code", -1)) != 0
                or int(state.get("worker_exit_code", -1)) != 0
            ):
                raise StageReplayError(
                    "completed generation stage worker did not exit successfully"
                )
            return {
                "status": "completed",
                "reused": True,
                "state": state_path.as_posix(),
                "attempt": len(attempts),
                "outputs": actual_outputs,
            }
        if state.get("status") == "running":
            child_pid = _optional_pid(
                state.get("child_pid"),
                name="receipt child PID",
            )
            if _process_exists(child_pid):
                _wait_for_process(child_pid)
            interrupted_attempt = int(state.get("attempt", len(attempts) or 1))
            worker_result_path = _worker_result_path(
                state_path,
                interrupted_attempt,
            )
            worker_result = None
            if not worker_result_path.is_file():
                _wait_for_path(
                    worker_result_path,
                    timeout_seconds=WORKER_SIDECAR_GRACE_SECONDS,
                )
            if worker_result_path.is_file():
                worker_result = _validate_worker_result(
                    worker_result_path,
                    expected_attempt=interrupted_attempt,
                    expected_request_sha256=request_sha256,
                    expected_command=command,
                )
            if worker_result is not None and worker_result["status"] == "running":
                worker_pid = _optional_pid(
                    worker_result.get("worker_pid"),
                    name="worker PID",
                )
                command_child_pid = _optional_pid(
                    worker_result.get("child_pid"),
                    name="command child PID",
                )
                if _process_exists(worker_pid):
                    _wait_for_process(worker_pid)
                worker_result = _validate_worker_result(
                    worker_result_path,
                    expected_attempt=interrupted_attempt,
                    expected_request_sha256=request_sha256,
                    expected_command=command,
                )
                if (
                    worker_result["status"] == "running"
                    and _process_exists(command_child_pid)
                ):
                    _wait_for_process(command_child_pid)
                if worker_result_path.is_file():
                    worker_result = _validate_worker_result(
                        worker_result_path,
                        expected_attempt=interrupted_attempt,
                        expected_request_sha256=request_sha256,
                        expected_command=command,
                    )
            if worker_result is not None and worker_result["exit_code"] == 0:
                try:
                    completed_outputs = _output_identities(outputs)
                except Exception:
                    archived = _archive_outputs(
                        outputs,
                        attempt=interrupted_attempt,
                        reason="invalid",
                    )
                    if attempts:
                        attempts[-1] = {
                            **attempts[-1],
                            "status": "failed",
                            "exit_code": 0,
                            "worker_result": file_identity(worker_result_path),
                            "archived_outputs": archived,
                            "finished_at": _utc_now(),
                        }
                else:
                    if attempts:
                        attempts[-1] = {
                            **attempts[-1],
                            "status": "completed",
                            "exit_code": 0,
                            "worker_result": file_identity(worker_result_path),
                            "finished_at": _utc_now(),
                        }
                    completed = _completed_state(
                        running_state=state,
                        attempts=attempts,
                        completed_outputs=completed_outputs,
                        worker_result=worker_result,
                        worker_result_path=worker_result_path,
                        recovered=True,
                    )
                    write_json_report(state_path, completed)
                    return {
                        "status": "completed",
                        "reused": True,
                        "recovered": True,
                        "state": state_path.as_posix(),
                        "attempt": interrupted_attempt,
                        "outputs": completed_outputs,
                    }
            else:
                reason = "failed" if worker_result is not None else "interrupted"
                archived = _archive_outputs(
                    outputs,
                    attempt=interrupted_attempt,
                    reason=reason,
                )
                if attempts:
                    attempts[-1] = {
                        **attempts[-1],
                        "status": reason,
                        "exit_code": (
                            None
                            if worker_result is None
                            else int(worker_result["exit_code"])
                        ),
                        "worker_result": (
                            None
                            if worker_result is None
                            else file_identity(worker_result_path)
                        ),
                        "archived_outputs": archived,
                        "finished_at": _utc_now(),
                    }
            write_json_report(
                state_path,
                {
                    **state,
                    "status": "failed",
                    "child_pid": None,
                    "attempts": attempts,
                    "updated_at": _utc_now(),
                },
            )
        elif state.get("status") != "failed":
            raise StageReplayError(
                "generation stage receipt has an unsupported status"
            )
    else:
        _require_outputs_absent(outputs)

    _require_outputs_absent(outputs)
    attempt_number = len(attempts) + 1
    worker_result_path = _worker_result_path(state_path, attempt_number)
    if worker_result_path.exists() or worker_result_path.is_symlink():
        raise StageReplayError(
            f"unbound generation stage worker result exists: {worker_result_path}"
        )
    attempt = {
        "attempt": attempt_number,
        "status": "starting",
        "worker_result_path": worker_result_path.as_posix(),
        "started_at": _utc_now(),
    }
    attempts.append(attempt)
    running_state = {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "running",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "child_pid": None,
        "attempt": attempt_number,
        "request": request,
        "attempts": attempts,
        "updated_at": _utc_now(),
    }
    write_json_report(state_path, running_state)
    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-result",
        str(worker_result_path),
        "--worker-cwd",
        str(cwd),
        "--worker-attempt",
        str(attempt_number),
        "--worker-request-sha256",
        request_sha256,
        "--",
        *command,
    ]
    worker_environment = os.environ.copy()
    python_roots = [str(project / "src"), str(project)]
    inherited_pythonpath = worker_environment.get("PYTHONPATH")
    if inherited_pythonpath:
        python_roots.append(inherited_pythonpath)
    worker_environment["PYTHONPATH"] = os.pathsep.join(python_roots)
    child = subprocess.Popen(
        worker_command,
        cwd=cwd,
        env=worker_environment,
        start_new_session=os.name != "nt",
    )
    attempt["status"] = "running"
    attempt["child_pid"] = child.pid
    running_state["child_pid"] = child.pid
    running_state["attempts"] = attempts
    running_state["updated_at"] = _utc_now()
    write_json_report(state_path, running_state)
    worker_process_exit = child.wait()
    if not worker_result_path.is_file():
        exit_code = (
            worker_process_exit
            if worker_process_exit != 0
            else WORKER_FAILURE_EXIT_CODE
        )
        attempt["exit_code"] = exit_code
        attempt["status"] = "failed"
        attempt["archived_outputs"] = _archive_outputs(
            outputs,
            attempt=attempt_number,
            reason="worker-failed",
        )
        attempt["finished_at"] = _utc_now()
        write_json_report(
            state_path,
            {
                **running_state,
                "status": "failed",
                "child_pid": None,
                "attempts": attempts,
                "updated_at": _utc_now(),
            },
        )
        raise subprocess.CalledProcessError(exit_code, worker_command)
    worker_result = _validate_worker_result(
        worker_result_path,
        expected_attempt=attempt_number,
        expected_request_sha256=request_sha256,
        expected_command=command,
    )
    if worker_result["status"] == "running":
        command_child_pid = int(worker_result.get("child_pid", -1))
        if _process_exists(command_child_pid):
            _wait_for_process(command_child_pid)
        exit_code = WORKER_FAILURE_EXIT_CODE
    else:
        exit_code = int(worker_result["exit_code"])
    attempt["exit_code"] = exit_code
    attempt["worker_result"] = file_identity(worker_result_path)
    attempt["finished_at"] = _utc_now()
    if exit_code != 0:
        attempt["status"] = "failed"
        attempt["archived_outputs"] = _archive_outputs(
            outputs,
            attempt=attempt_number,
            reason="failed",
        )
        write_json_report(
            state_path,
            {
                **running_state,
                "status": "failed",
                "child_pid": None,
                "attempts": attempts,
                "updated_at": _utc_now(),
            },
        )
        raise subprocess.CalledProcessError(exit_code, command)

    try:
        completed_outputs = _output_identities(outputs)
    except Exception:
        attempt["status"] = "failed"
        attempt["archived_outputs"] = _archive_outputs(
            outputs,
            attempt=attempt_number,
            reason="invalid",
        )
        write_json_report(
            state_path,
            {
                **running_state,
                "status": "failed",
                "child_pid": None,
                "attempts": attempts,
                "updated_at": _utc_now(),
            },
        )
        raise
    attempt["status"] = "completed"
    completed = _completed_state(
        running_state=running_state,
        attempts=attempts,
        completed_outputs=completed_outputs,
        worker_result=worker_result,
        worker_result_path=worker_result_path,
        recovered=False,
    )
    write_json_report(state_path, completed)
    return {
        "status": "completed",
        "reused": False,
        "state": state_path.as_posix(),
        "attempt": attempt_number,
        "outputs": completed_outputs,
    }


def _parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-result", type=Path, required=True)
    parser.add_argument("--worker-cwd", type=Path, required=True)
    parser.add_argument("--worker-attempt", type=int, required=True)
    parser.add_argument("--worker-request-sha256", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise ValueError("generation stage worker command must not be empty")
    return args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one generation stage with source-bound, content-verified restart semantics."
        )
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--input-file", action="append", default=[])
    parser.add_argument("--input-tree", action="append", default=[])
    parser.add_argument("--output-file", action="append", default=[])
    parser.add_argument("--output-tree", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> None:
    args = _parse_args()
    result = run_stage_once(
        state_path=args.state,
        project=args.project,
        cwd=args.cwd,
        command=args.command,
        input_files=args.input_file,
        input_trees=args.input_tree,
        output_files=args.output_file,
        output_trees=args.output_tree,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        if "--worker-result" in sys.argv[1:]:
            worker_args = _parse_worker_args()
            raise SystemExit(
                _run_worker(
                    result_path=worker_args.worker_result,
                    cwd=worker_args.worker_cwd,
                    attempt=worker_args.worker_attempt,
                    request_sha256=worker_args.worker_request_sha256,
                    command=worker_args.command,
                )
            )
        main()
    except StageReplayError as error:
        print(
            f"generation stage replay rejected: {error}",
            file=sys.stderr,
        )
        raise SystemExit(REPLAY_ERROR_EXIT_CODE)
