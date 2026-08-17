from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import write_json_report

try:
    import build_generation_statistical_claim_language_guard as guard_builder
    import run_generation_quality_bridge_claim_qualification_waiter as source_waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_statistical_claim_language_guard as guard_builder,
    )
    from scripts import (
        run_generation_quality_bridge_claim_qualification_waiter as source_waiter,
    )


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_quality_bridge_claim_language_guard_waiter"
SOURCE_KIND = "quality_bridge_100k"
CLAIM_BOUNDARY = {
    **guard_builder.CLAIM_BOUNDARY,
    "source_claim_wait_only": True,
    "source_waiter_signals_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _git_identity(project: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(
            git("status", "--porcelain", "--untracked-files=no")
        ),
    }


def _source_process_active(*, pid: int, output_root: Path) -> bool:
    command_path = Path(f"/proc/{pid}/cmdline")
    if not command_path.is_file():
        return False
    try:
        command = command_path.read_bytes().replace(b"\0", b" ").decode(
            "utf-8",
            errors="replace",
        )
    except OSError:
        return False
    return (
        "run_generation_quality_bridge_claim_qualification_waiter.py" in command
        and output_root.as_posix() in command
    )


def validate_source_status(
    report: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_output_root: Path,
    expected_source_report: Path,
    expected_control_git: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(report.get("status", ""))
    phase = str(report.get("phase", ""))
    if (
        int(report.get("schema_version", -1)) != source_waiter.WAITER_SCHEMA_VERSION
        or report.get("role") != source_waiter.WAITER_ROLE
        or int(report.get("pid", -1)) != expected_pid
        or status not in {"waiting", "running", "pass", "hold", "failed"}
        or report.get("claim_boundary") != source_waiter.CLAIM_BOUNDARY
    ):
        raise ValueError("quality-bridge claim source waiter contract differs")
    if status in {"pass", "hold"} and phase != "completed":
        raise ValueError("quality-bridge claim source waiter phase differs")
    if status == "failed" and phase not in {"source", "failed"}:
        raise ValueError("quality-bridge claim source failure phase differs")

    expected = report.get("expected")
    if status != "failed" or isinstance(expected, Mapping):
        if (
            not isinstance(expected, Mapping)
            or expected.get("control_git") != dict(expected_control_git)
            or expected.get("output_root") != expected_output_root.as_posix()
            or expected.get("qualification_output")
            != expected_source_report.as_posix()
        ):
            raise ValueError("quality-bridge claim source expectation differs")

    qualification = report.get("statistical_claim_qualification")
    if status in {"pass", "hold"}:
        if (
            not isinstance(qualification, Mapping)
            or not isinstance(qualification.get("identity"), Mapping)
            or qualification.get("status") != status
            or not isinstance(qualification.get("decision"), str)
            or not isinstance(qualification.get("claim_policy"), Mapping)
        ):
            raise ValueError("quality-bridge claim source evidence is incomplete")
        identity = qualification["identity"]
        if identity.get("path") != expected_source_report.as_posix():
            raise ValueError("quality-bridge claim source report path differs")
    return {
        "status": status,
        "detail": report.get("detail"),
        "phase": phase,
        "pid": expected_pid,
        "terminal": status in {"pass", "hold", "failed"},
        "qualification": (
            dict(qualification) if isinstance(qualification, Mapping) else None
        ),
        "updated_at": report.get("updated_at"),
    }


def _write_pid(path: Path, *, expected: Mapping[str, Any]) -> None:
    write_json_report(
        path,
        {
            "schema_version": 1,
            "role": WAITER_ROLE,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "expected_control_revision": expected["control_git"]["revision"],
            "expected_source_pid": expected["source_waiter"]["pid"],
            "started_at": _utc_now(),
        },
    )


def _remove_pid(path: Path) -> None:
    if not path.is_file():
        return
    try:
        report = read_json_object(path, name="claim language guard waiter PID")
    except ValueError:
        return
    if report.get("role") == WAITER_ROLE and int(report.get("pid", -1)) == os.getpid():
        path.unlink()


def _status(
    *,
    status: str,
    detail: str,
    phase: str,
    expected: Mapping[str, Any],
    source_waiter_state: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    guard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": WAITER_SCHEMA_VERSION,
        "role": WAITER_ROLE,
        "status": status,
        "detail": detail,
        "phase": phase,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected": dict(expected),
        "source_waiter": (
            dict(source_waiter_state)
            if source_waiter_state is not None
            else None
        ),
        "source": dict(source) if source is not None else None,
        "claim_language_guard": dict(guard) if guard is not None else None,
        "claim_boundary": CLAIM_BOUNDARY,
        "updated_at": _utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact quality-bridge claim qualification, then build "
            "one CPU-only non-authorizing metric-semantics language guard."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-output-root", type=Path, required=True)
    parser.add_argument("--source-status", type=Path, required=True)
    parser.add_argument("--source-pid-file", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--guard-output", type=Path, required=True)
    parser.add_argument("--expected-source-pid", type=int, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-source-control-revision", required=True)
    parser.add_argument("--expected-source-control-tree", required=True)
    parser.add_argument("--expected-source-control-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    args = parser.parse_args()
    if (
        args.expected_source_pid < 1
        or args.poll_seconds <= 0
        or args.timeout_seconds <= 0
    ):
        parser.error("claim language guard waiter arguments are invalid")
    return args


def _run_locked(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(
        args.project,
        name="claim language guard waiter control checkout",
    ).resolve()
    source_output_root = reject_symlink_chain(
        args.source_output_root,
        name="quality claim source output root",
    ).resolve()
    source_status_path = reject_symlink_chain(
        args.source_status,
        name="quality claim source waiter status",
    ).resolve()
    source_pid_path = reject_symlink_chain(
        args.source_pid_file,
        name="quality claim source waiter PID file",
    ).resolve()
    source_report_path = reject_symlink_chain(
        args.source_report,
        name="quality claim source report",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="claim language guard waiter output root",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="claim language guard waiter status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="claim language guard waiter PID file",
    ).resolve()
    guard_output = reject_symlink_chain(
        args.guard_output,
        name="claim language guard output",
    ).resolve()
    if (
        source_output_root == output_root
        or not _is_within(source_status_path, source_output_root)
        or not _is_within(source_pid_path, source_output_root)
        or not _is_within(source_report_path, source_output_root)
        or not _is_within(status_output, output_root)
        or not _is_within(pid_file, output_root)
        or not _is_within(guard_output, output_root)
    ):
        raise ValueError("claim language guard waiter path scope differs")

    control_git = {
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    source_control_git = {
        "revision": args.expected_source_control_revision,
        "tree": args.expected_source_control_tree,
        "branch": args.expected_source_control_branch,
        "tracked_dirty": False,
    }
    if _git_identity(project) != control_git:
        raise ValueError("claim language guard waiter control checkout differs")
    expected = {
        "control_git": control_git,
        "source_kind": SOURCE_KIND,
        "source_waiter": {
            "pid": args.expected_source_pid,
            "control_git": source_control_git,
            "output_root": source_output_root.as_posix(),
            "status": source_status_path.as_posix(),
            "pid_file": source_pid_path.as_posix(),
        },
        "source_report": source_report_path.as_posix(),
        "output_root": output_root.as_posix(),
        "guard_output": guard_output.as_posix(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_pid(pid_file, expected=expected)
    deadline = time.monotonic() + args.timeout_seconds
    source_state: dict[str, Any] | None = None

    def publish(
        *,
        status: str,
        detail: str,
        phase: str,
        source: Mapping[str, Any] | None = None,
        guard: Mapping[str, Any] | None = None,
    ) -> None:
        write_json_report(
            status_output,
            _status(
                status=status,
                detail=detail,
                phase=phase,
                expected=expected,
                source_waiter_state=source_state,
                source=source,
                guard=guard,
            ),
        )

    while True:
        if _git_identity(project) != control_git:
            raise ValueError("claim language guard waiter control checkout changed")
        if source_status_path.is_file():
            source_state = validate_source_status(
                read_json_object(
                    source_status_path,
                    name="quality claim source waiter status",
                ),
                expected_pid=args.expected_source_pid,
                expected_output_root=source_output_root,
                expected_source_report=source_report_path,
                expected_control_git=source_control_git,
            )
            if source_state["status"] == "failed":
                publish(
                    status="failed",
                    detail="quality_claim_source_waiter_failed",
                    phase="source",
                )
                return 1
            process_active = _source_process_active(
                pid=args.expected_source_pid,
                output_root=source_output_root,
            )
            if (
                source_state["status"] in {"pass", "hold"}
                and not source_pid_path.is_file()
                and not process_active
                and source_report_path.is_file()
            ):
                break
        publish(
            status="waiting",
            detail="waiting_for_quality_claim_source",
            phase="source",
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            publish(
                status="failed",
                detail="claim_language_guard_waiter_timeout",
                phase="source",
            )
            return 1
        time.sleep(min(args.poll_seconds, remaining))

    source_status_identity = file_identity(source_status_path)
    source_status_report = read_json_object(
        source_status_path,
        name="terminal quality claim source waiter status",
    )
    source_state = validate_source_status(
        source_status_report,
        expected_pid=args.expected_source_pid,
        expected_output_root=source_output_root,
        expected_source_report=source_report_path,
        expected_control_git=source_control_git,
    )
    if file_identity(source_status_path) != source_status_identity:
        raise ValueError("quality claim source waiter status changed")
    source_identity = file_identity(source_report_path)
    if source_state["qualification"]["identity"] != source_identity:
        raise ValueError("quality claim source report identity differs")
    source_report = read_json_object(
        source_report_path,
        name="quality claim source report",
    )
    if (
        source_report.get("status") != source_state["status"]
        or source_report.get("decision")
        != source_state["qualification"]["decision"]
        or source_report.get("claim_policy")
        != source_state["qualification"]["claim_policy"]
    ):
        raise ValueError("quality claim source report summary differs")
    report = guard_builder.build_guard(
        source_kind=SOURCE_KIND,
        source_report_path=source_report_path,
        expected_source_report_sha256=source_identity["sha256"],
    )
    if report["status"] != source_state["status"]:
        raise ValueError("claim language guard status differs from source")
    guard_identity = prepare_manifest(
        guard_output,
        report,
        resume=guard_output.is_file(),
        overwrite=False,
    )
    source = {
        "waiter_status": source_status_identity,
        "claim_report": source_identity,
        "status": source_state["status"],
        "decision": source_state["qualification"]["decision"],
        "claim_policy": source_state["qualification"]["claim_policy"],
    }
    guard = {
        "identity": guard_identity,
        "status": report["status"],
        "decision": report["decision"],
        "metric_roles": report["metric_roles"],
        "claim_policy": report["claim_policy"],
        "claim_text": report["claim_text"],
    }
    publish(
        status=str(report["status"]),
        detail=str(report["decision"]),
        phase="completed",
        source=source,
        guard=guard,
    )
    return 0


def run_waiter(args: argparse.Namespace) -> int:
    output_root = reject_symlink_chain(
        args.output_root,
        name="claim language guard waiter output root",
    ).resolve()
    with exclusive_output_lock(output_root, role=WAITER_ROLE):
        try:
            return _run_locked(args)
        except Exception as error:
            status_output = reject_symlink_chain(
                args.status_output,
                name="claim language guard waiter status",
            ).resolve()
            if _is_within(status_output, output_root):
                write_json_report(
                    status_output,
                    {
                        "schema_version": WAITER_SCHEMA_VERSION,
                        "role": WAITER_ROLE,
                        "status": "failed",
                        "detail": f"{type(error).__name__}: {error}",
                        "phase": "failed",
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "claim_boundary": CLAIM_BOUNDARY,
                        "updated_at": _utc_now(),
                    },
                )
            raise
        finally:
            pid_file = reject_symlink_chain(
                args.pid_file,
                name="claim language guard waiter PID file",
            ).resolve()
            if _is_within(pid_file, output_root):
                _remove_pid(pid_file)


def main() -> int:
    args = parse_args()
    try:
        return run_waiter(args)
    except OutputLockError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
