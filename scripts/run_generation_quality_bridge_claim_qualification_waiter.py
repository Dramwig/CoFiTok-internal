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
    import build_generation_quality_bridge_statistical_claim_qualification as qualification_builder
    import run_generation_quality_bridge_terminal_uncertainty_waiter as terminal_waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_quality_bridge_statistical_claim_qualification as qualification_builder,
    )
    from scripts import (
        run_generation_quality_bridge_terminal_uncertainty_waiter as terminal_waiter,
    )


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_quality_bridge_claim_qualification_waiter"
CLAIM_BOUNDARY = {
    **qualification_builder.CLAIM_BOUNDARY,
    "terminal_source_wait_only": True,
    "terminal_waiter_signals_allowed": False,
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


def _terminal_process_active(*, pid: int, output_root: Path) -> bool:
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
        "run_generation_quality_bridge_terminal_uncertainty_waiter.py" in command
        and output_root.as_posix() in command
    )


def validate_terminal_status(
    report: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_output_root: Path,
    expected_control_git: Mapping[str, Any],
    expected_quality_git: Mapping[str, Any],
    expected_evaluator_git: Mapping[str, Any],
) -> dict[str, Any]:
    expected = report.get("expected")
    status = str(report.get("status", ""))
    if (
        int(report.get("schema_version", -1))
        != terminal_waiter.WAITER_SCHEMA_VERSION
        or report.get("role") != terminal_waiter.WAITER_ROLE
        or int(report.get("pid", -1)) != expected_pid
        or status not in {"waiting", "running", "pass", "hold", "failed"}
        or report.get("claim_boundary") != terminal_waiter.CLAIM_BOUNDARY
        or not isinstance(expected, Mapping)
        or expected.get("output_root") != expected_output_root.as_posix()
        or expected.get("control_git") != dict(expected_control_git)
        or expected.get("quality_git") != dict(expected_quality_git)
        or expected.get("evaluator_git") != dict(expected_evaluator_git)
    ):
        raise ValueError("quality-bridge terminal waiter contract differs")
    phase = str(report.get("phase", ""))
    if status in {"pass", "hold"} and phase != "completed":
        raise ValueError("quality-bridge terminal waiter phase differs")
    if status == "failed" and phase != "failed":
        raise ValueError("quality-bridge terminal waiter failure phase differs")
    manifest = report.get("execution_manifest")
    audit = report.get("audit")
    quality = report.get("quality_bridge")
    if status in {"pass", "hold"} and (
        not isinstance(manifest, Mapping)
        or not isinstance(manifest.get("identity"), Mapping)
        or not isinstance(audit, Mapping)
        or not isinstance(audit.get("source"), Mapping)
        or not isinstance(quality, Mapping)
        or not isinstance(quality.get("result"), Mapping)
    ):
        raise ValueError("quality-bridge terminal waiter evidence is incomplete")
    return {
        "status": status,
        "detail": report.get("detail"),
        "phase": phase,
        "pid": expected_pid,
        "terminal": status in {"pass", "hold", "failed"},
        "execution_manifest": dict(manifest) if isinstance(manifest, Mapping) else None,
        "audit": dict(audit) if isinstance(audit, Mapping) else None,
        "quality_bridge": dict(quality) if isinstance(quality, Mapping) else None,
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
            "started_at": _utc_now(),
        },
    )


def _remove_pid(path: Path) -> None:
    if not path.is_file():
        return
    try:
        report = read_json_object(path, name="quality claim waiter PID")
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
    terminal: Mapping[str, Any] | None = None,
    sources: Mapping[str, Any] | None = None,
    qualification: Mapping[str, Any] | None = None,
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
        "terminal_waiter": dict(terminal) if terminal is not None else None,
        "sources": dict(sources) if sources is not None else None,
        "statistical_claim_qualification": (
            dict(qualification) if qualification is not None else None
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "updated_at": _utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact full-data quality-bridge terminal uncertainty "
            "result, then build one CPU-only non-authorizing claim qualification."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--terminal-output-root", type=Path, required=True)
    parser.add_argument("--terminal-status", type=Path, required=True)
    parser.add_argument("--terminal-pid-file", type=Path, required=True)
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--uncertainty-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--qualification-output", type=Path, required=True)
    parser.add_argument("--expected-terminal-pid", type=int, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-terminal-control-revision", required=True)
    parser.add_argument("--expected-terminal-control-tree", required=True)
    parser.add_argument("--expected-terminal-control-branch", required=True)
    parser.add_argument("--expected-quality-revision", required=True)
    parser.add_argument("--expected-quality-branch", required=True)
    parser.add_argument("--expected-evaluator-revision", required=True)
    parser.add_argument("--expected-evaluator-tree", required=True)
    parser.add_argument("--expected-evaluator-branch", default="")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    args = parser.parse_args()
    if args.expected_terminal_pid < 1 or args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("quality claim waiter arguments are invalid")
    return args


def _run_locked(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(
        args.project,
        name="quality claim waiter control checkout",
    ).resolve()
    terminal_output_root = reject_symlink_chain(
        args.terminal_output_root,
        name="terminal uncertainty output root",
    ).resolve()
    terminal_status_path = reject_symlink_chain(
        args.terminal_status,
        name="terminal uncertainty waiter status",
    ).resolve()
    terminal_pid_path = reject_symlink_chain(
        args.terminal_pid_file,
        name="terminal uncertainty waiter PID file",
    ).resolve()
    quality_result_path = reject_symlink_chain(
        args.quality_result,
        name="quality-bridge result",
    ).resolve()
    manifest_path = reject_symlink_chain(
        args.execution_manifest,
        name="terminal uncertainty execution manifest",
    ).resolve()
    uncertainty_path = reject_symlink_chain(
        args.uncertainty_report,
        name="terminal uncertainty report",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="quality claim waiter output root",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="quality claim waiter status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="quality claim waiter PID file",
    ).resolve()
    qualification_output = reject_symlink_chain(
        args.qualification_output,
        name="quality claim qualification output",
    ).resolve()
    if (
        terminal_output_root == output_root
        or not _is_within(terminal_status_path, terminal_output_root)
        or not _is_within(terminal_pid_path, terminal_output_root)
        or not _is_within(manifest_path, terminal_output_root)
        or not _is_within(uncertainty_path, terminal_output_root)
        or not _is_within(status_output, output_root)
        or not _is_within(pid_file, output_root)
        or not _is_within(qualification_output, output_root)
    ):
        raise ValueError("quality claim waiter path scope differs")

    control_git = {
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    terminal_control_git = {
        "revision": args.expected_terminal_control_revision,
        "tree": args.expected_terminal_control_tree,
        "branch": args.expected_terminal_control_branch,
        "tracked_dirty": False,
    }
    quality_git = {
        "revision": args.expected_quality_revision,
        "branch": args.expected_quality_branch,
        "tracked_dirty": False,
    }
    evaluator_git = {
        "revision": args.expected_evaluator_revision,
        "tree": args.expected_evaluator_tree,
        "branch": args.expected_evaluator_branch,
        "tracked_dirty": False,
    }
    if _git_identity(project) != control_git:
        raise ValueError("quality claim waiter control checkout differs")
    expected = {
        "control_git": control_git,
        "terminal_waiter": {
            "pid": args.expected_terminal_pid,
            "control_git": terminal_control_git,
            "output_root": terminal_output_root.as_posix(),
            "status": terminal_status_path.as_posix(),
            "pid_file": terminal_pid_path.as_posix(),
        },
        "quality_git": quality_git,
        "evaluator_git": evaluator_git,
        "quality_result": quality_result_path.as_posix(),
        "execution_manifest": manifest_path.as_posix(),
        "uncertainty_report": uncertainty_path.as_posix(),
        "output_root": output_root.as_posix(),
        "qualification_output": qualification_output.as_posix(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_pid(pid_file, expected=expected)
    deadline = time.monotonic() + args.timeout_seconds
    terminal_state: dict[str, Any] | None = None

    def publish(
        *,
        status: str,
        detail: str,
        phase: str,
        sources: Mapping[str, Any] | None = None,
        qualification: Mapping[str, Any] | None = None,
    ) -> None:
        write_json_report(
            status_output,
            _status(
                status=status,
                detail=detail,
                phase=phase,
                expected=expected,
                terminal=terminal_state,
                sources=sources,
                qualification=qualification,
            ),
        )

    while True:
        if _git_identity(project) != control_git:
            raise ValueError("quality claim waiter control checkout changed")
        if terminal_status_path.is_file():
            terminal_state = validate_terminal_status(
                read_json_object(
                    terminal_status_path,
                    name="quality-bridge terminal waiter status",
                ),
                expected_pid=args.expected_terminal_pid,
                expected_output_root=terminal_output_root,
                expected_control_git=terminal_control_git,
                expected_quality_git=quality_git,
                expected_evaluator_git=evaluator_git,
            )
            if terminal_state["status"] == "failed":
                publish(
                    status="failed",
                    detail="terminal_uncertainty_waiter_failed",
                    phase="source",
                )
                return 1
            process_active = _terminal_process_active(
                pid=args.expected_terminal_pid,
                output_root=terminal_output_root,
            )
            if (
                terminal_state["status"] in {"pass", "hold"}
                and not terminal_pid_path.is_file()
                and not process_active
                and quality_result_path.is_file()
                and manifest_path.is_file()
                and uncertainty_path.is_file()
            ):
                break
        publish(
            status="waiting",
            detail="waiting_for_terminal_quality_and_uncertainty_sources",
            phase="source",
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            publish(
                status="failed",
                detail="quality_claim_waiter_timeout",
                phase="source",
            )
            return 1
        time.sleep(min(args.poll_seconds, remaining))

    sources = {
        "quality_result": file_identity(quality_result_path),
        "execution_manifest": file_identity(manifest_path),
        "uncertainty_report": file_identity(uncertainty_path),
        "terminal_status": file_identity(terminal_status_path),
    }
    if (
        terminal_state["quality_bridge"]["result"] != sources["quality_result"]
        or terminal_state["execution_manifest"]["identity"]
        != sources["execution_manifest"]
        or terminal_state["audit"]["source"] != sources["uncertainty_report"]
    ):
        raise ValueError("quality claim waiter terminal source identities differ")
    report = qualification_builder.build_qualification(
        quality_result_path=quality_result_path,
        expected_quality_result_sha256=sources["quality_result"]["sha256"],
        execution_manifest_path=manifest_path,
        expected_execution_manifest_sha256=sources["execution_manifest"]["sha256"],
        uncertainty_report_path=uncertainty_path,
        expected_uncertainty_report_sha256=sources["uncertainty_report"]["sha256"],
        expected_quality_revision=args.expected_quality_revision,
        expected_quality_branch=args.expected_quality_branch,
        expected_evaluator_revision=args.expected_evaluator_revision,
        expected_evaluator_branch=args.expected_evaluator_branch,
        output_root=terminal_output_root,
    )
    qualification_identity = prepare_manifest(
        qualification_output,
        report,
        resume=qualification_output.is_file(),
        overwrite=False,
    )
    publish(
        status=str(report["status"]),
        detail=str(report["decision"]),
        phase="completed",
        sources=sources,
        qualification={
            "identity": qualification_identity,
            "status": report["status"],
            "decision": report["decision"],
            "claim_policy": report["claim_policy"],
        },
    )
    return 0


def run_waiter(args: argparse.Namespace) -> int:
    output_root = reject_symlink_chain(
        args.output_root,
        name="quality claim waiter output root",
    ).resolve()
    with exclusive_output_lock(output_root, role=WAITER_ROLE):
        try:
            return _run_locked(args)
        except Exception as error:
            status_output = reject_symlink_chain(
                args.status_output,
                name="quality claim waiter status",
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
                name="quality claim waiter PID file",
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
