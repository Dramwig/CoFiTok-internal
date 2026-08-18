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
    import build_generation_runtime_compute_claim_guard as guard_builder
    import wait_generation_quality_bridge_runtime_compute_fairness as source_waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import build_generation_runtime_compute_claim_guard as guard_builder
    from scripts import (
        wait_generation_quality_bridge_runtime_compute_fairness as source_waiter,
    )


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_quality_bridge_runtime_claim_guard_waiter"
DEPLOYMENT_SCHEMA_VERSION = 1
DEPLOYMENT_ROLE = "generation_quality_bridge_runtime_claim_guard_deployment"
SOURCE_SCOPE = {
    "cpu_only": True,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "full_300k_launch_allowed": False,
}
CLAIM_BOUNDARY = {
    **guard_builder.CLAIM_BOUNDARY,
    "source_runtime_fairness_wait_only": True,
    "source_waiter_signals_allowed": False,
    "pair_monitor_signals_allowed": False,
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


def _source_process_active(
    *,
    pid: int,
    source_final_report: Path,
) -> bool:
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
        "wait_generation_quality_bridge_runtime_compute_fairness.py" in command
        and source_final_report.as_posix() in command
    )


def validate_source_deployment(
    report: Mapping[str, Any],
    *,
    expected_source_git: Mapping[str, Any],
    expected_source_status: Path,
    expected_source_final_report: Path,
) -> None:
    auditor = report.get("auditor")
    checkout = auditor.get("checkout") if isinstance(auditor, Mapping) else None
    targets = report.get("targets")
    scope = report.get("scope")
    if (
        report.get("schema_version") != 1
        or report.get("role") != source_waiter.DEPLOYMENT_ROLE
        or report.get("status") != "pass"
        or checkout != dict(expected_source_git)
        or not isinstance(targets, Mapping)
        or targets.get("status_output") != expected_source_status.as_posix()
        or targets.get("audit_output") != expected_source_final_report.as_posix()
        or not isinstance(scope, Mapping)
        or scope.get("cpu_only_waiter") is not True
        or scope.get("gpu_execution_allowed") is not False
        or scope.get("training_process_signals_allowed") is not False
        or scope.get("unrelated_process_signals_allowed") is not False
        or scope.get("promotion_or_release_allowed") is not False
        or scope.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("runtime fairness source deployment contract differs")


def validate_source_status(
    report: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_deployment_identity: Mapping[str, Any],
    expected_source_final_report: Path,
) -> dict[str, Any]:
    status = str(report.get("status", ""))
    if (
        report.get("schema_version") != 1
        or report.get("role") != source_waiter.WAITER_ROLE
        or int(report.get("pid", -1)) != expected_pid
        or status not in {"waiting", "pass", "failed"}
        or report.get("scope") != SOURCE_SCOPE
        or report.get("deployment_receipt")
        != dict(expected_deployment_identity)
    ):
        raise ValueError("runtime fairness source waiter contract differs")
    audit = report.get("audit_output")
    if status == "waiting":
        if audit is not None or report.get("detail") != (
            "waiting_for_both_exact_100k_training_reports"
        ):
            raise ValueError("runtime fairness source waiting state differs")
    elif status == "pass":
        if (
            report.get("detail") != "terminal_runtime_compute_fairness_verified"
            or not isinstance(audit, Mapping)
            or audit.get("path") != expected_source_final_report.as_posix()
        ):
            raise ValueError("runtime fairness source terminal state differs")
    elif report.get("detail") != "terminal_runtime_compute_fairness_failed":
        raise ValueError("runtime fairness source failure state differs")
    return {
        "status": status,
        "detail": report.get("detail"),
        "pid": expected_pid,
        "terminal": status in {"pass", "failed"},
        "audit_output": dict(audit) if isinstance(audit, Mapping) else None,
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
        report = read_json_object(path, name="runtime claim guard waiter PID")
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
    deployment_receipt: Mapping[str, Any],
    source_waiter_state: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    pair_monitor: Mapping[str, Any] | None = None,
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
        "deployment_receipt": dict(deployment_receipt),
        "source_waiter": (
            dict(source_waiter_state)
            if source_waiter_state is not None
            else None
        ),
        "source": dict(source) if source is not None else None,
        "pair_monitor": (
            dict(pair_monitor) if pair_monitor is not None else None
        ),
        "runtime_claim_guard": dict(guard) if guard is not None else None,
        "claim_boundary": CLAIM_BOUNDARY,
        "updated_at": _utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact runtime/compute fairness report and terminal "
            "pair monitor, then build one CPU-only runtime claim guard."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--source-status", type=Path, required=True)
    parser.add_argument("--source-final-report", type=Path, required=True)
    parser.add_argument("--source-deployment-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-source-deployment-receipt-sha256", required=True
    )
    parser.add_argument("--pair-monitor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--guard-output", type=Path, required=True)
    parser.add_argument("--expected-source-pid", type=int, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-source-control-revision", required=True)
    parser.add_argument("--expected-source-control-tree", required=True)
    parser.add_argument("--expected-source-control-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-monitor-name", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    args = parser.parse_args()
    if (
        args.expected_source_pid < 1
        or args.poll_seconds <= 0.0
        or args.timeout_seconds <= 0.0
    ):
        parser.error("runtime claim guard waiter arguments are invalid")
    return args


def _run_locked(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(
        args.project,
        name="runtime claim guard control checkout",
    ).resolve()
    quality_root = reject_symlink_chain(
        args.quality_output_root,
        name="quality bridge output root",
    ).resolve()
    source_status_path = reject_symlink_chain(
        args.source_status,
        name="runtime fairness source waiter status",
    ).resolve()
    source_final_path = reject_symlink_chain(
        args.source_final_report,
        name="runtime fairness source report",
    ).resolve()
    source_deployment_path = reject_symlink_chain(
        args.source_deployment_receipt,
        name="runtime fairness source deployment receipt",
    ).resolve()
    pair_monitor_path = reject_symlink_chain(
        args.pair_monitor,
        name="quality bridge pair monitor",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="runtime claim guard output root",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="runtime claim guard waiter status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="runtime claim guard waiter PID file",
    ).resolve()
    deployment_output = reject_symlink_chain(
        args.deployment_receipt_output,
        name="runtime claim guard deployment receipt",
    ).resolve()
    guard_output = reject_symlink_chain(
        args.guard_output,
        name="runtime claim guard output",
    ).resolve()
    if (
        output_root == quality_root
        or not all(
            _is_within(path, quality_root)
            for path in (
                source_status_path,
                source_final_path,
                source_deployment_path,
                pair_monitor_path,
                output_root,
            )
        )
        or not all(
            _is_within(path, output_root)
            for path in (
                status_output,
                pid_file,
                deployment_output,
                guard_output,
            )
        )
    ):
        raise ValueError("runtime claim guard waiter path scope differs")

    control_git = {
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    if _git_identity(project) != control_git:
        raise ValueError("runtime claim guard control checkout differs")
    source_deployment_identity = file_identity(source_deployment_path)
    if (
        source_deployment_identity["sha256"]
        != args.expected_source_deployment_receipt_sha256
    ):
        raise ValueError("runtime fairness source deployment SHA256 differs")
    source_deployment_report = read_json_object(
        source_deployment_path,
        name="runtime fairness source deployment receipt",
    )
    source_checkout = source_deployment_report.get("auditor", {}).get(
        "checkout", {}
    )
    if not isinstance(source_checkout, Mapping):
        raise ValueError("runtime fairness source checkout identity is missing")
    source_control_git = dict(source_checkout)
    expected_source_control = {
        "path": source_control_git.get("path"),
        "revision": args.expected_source_control_revision,
        "tree": args.expected_source_control_tree,
        "branch": args.expected_source_control_branch,
        "tracked_dirty": False,
    }
    if source_control_git != expected_source_control:
        raise ValueError("runtime fairness source checkout identity differs")
    validate_source_deployment(
        source_deployment_report,
        expected_source_git=expected_source_control,
        expected_source_status=source_status_path,
        expected_source_final_report=source_final_path,
    )
    expected = {
        "control_git": control_git,
        "source_waiter": {
            "pid": args.expected_source_pid,
            "control_git": expected_source_control,
            "status": source_status_path.as_posix(),
            "final_report": source_final_path.as_posix(),
            "deployment_receipt": source_deployment_identity,
        },
        "pair_monitor": {
            "path": pair_monitor_path.as_posix(),
            "monitor_name": args.expected_monitor_name,
            "training_revision": args.expected_training_revision,
            "training_branch": args.expected_training_branch,
        },
        "output_root": output_root.as_posix(),
        "guard_output": guard_output.as_posix(),
    }
    deployment = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "role": DEPLOYMENT_ROLE,
        "status": "pass",
        "control_git": control_git,
        "expected": expected,
        "scope": CLAIM_BOUNDARY,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    deployment_identity = prepare_manifest(
        deployment_output,
        deployment,
        resume=deployment_output.is_file(),
        overwrite=False,
    )
    _write_pid(pid_file, expected=expected)
    deadline = time.monotonic() + args.timeout_seconds
    source_state: dict[str, Any] | None = None

    def publish(
        *,
        status: str,
        detail: str,
        phase: str,
        source: Mapping[str, Any] | None = None,
        pair: Mapping[str, Any] | None = None,
        guard: Mapping[str, Any] | None = None,
    ) -> None:
        write_json_report(
            status_output,
            _status(
                status=status,
                detail=detail,
                phase=phase,
                expected=expected,
                deployment_receipt=deployment_identity,
                source_waiter_state=source_state,
                source=source,
                pair_monitor=pair,
                guard=guard,
            ),
        )

    while True:
        if _git_identity(project) != control_git:
            raise ValueError("runtime claim guard control checkout changed")
        if file_identity(source_deployment_path) != source_deployment_identity:
            raise ValueError("runtime fairness source deployment changed")
        if source_status_path.is_file():
            source_state = validate_source_status(
                read_json_object(
                    source_status_path,
                    name="runtime fairness source waiter status",
                ),
                expected_pid=args.expected_source_pid,
                expected_deployment_identity=source_deployment_identity,
                expected_source_final_report=source_final_path,
            )
            if source_state["status"] == "failed":
                publish(
                    status="failed",
                    detail="runtime_fairness_source_waiter_failed",
                    phase="source",
                )
                return 1
        source_ready = (
            source_state is not None
            and source_state["status"] == "pass"
            and source_final_path.is_file()
            and not _source_process_active(
                pid=args.expected_source_pid,
                source_final_report=source_final_path,
            )
        )
        pair_ready = False
        if pair_monitor_path.is_file():
            pair = read_json_object(
                pair_monitor_path,
                name="quality bridge pair monitor",
            )
            if pair.get("status") in {"failed", "stalled"}:
                publish(
                    status="failed",
                    detail="quality_bridge_pair_monitor_failed",
                    phase="pair_monitor",
                )
                return 1
            pair_ready = (
                pair.get("status") == "pass"
                and pair.get("stage") == "complete"
            )
        if source_ready and pair_ready:
            break
        publish(
            status="waiting",
            detail=(
                "waiting_for_terminal_pair_monitor"
                if source_ready
                else "waiting_for_runtime_fairness_source"
            ),
            phase="source" if not source_ready else "pair_monitor",
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            publish(
                status="failed",
                detail="runtime_claim_guard_waiter_timeout",
                phase="source",
            )
            return 1
        time.sleep(min(args.poll_seconds, remaining))

    source_status_identity = file_identity(source_status_path)
    source_state = validate_source_status(
        read_json_object(
            source_status_path,
            name="terminal runtime fairness source waiter status",
        ),
        expected_pid=args.expected_source_pid,
        expected_deployment_identity=source_deployment_identity,
        expected_source_final_report=source_final_path,
    )
    source_identity = file_identity(source_final_path)
    if source_state["audit_output"] != source_identity:
        raise ValueError("runtime fairness source report identity differs")
    pair_identity = file_identity(pair_monitor_path)
    report = guard_builder.build_guard(
        runtime_fairness_report_path=source_final_path,
        expected_runtime_fairness_sha256=source_identity["sha256"],
        pair_monitor_path=pair_monitor_path,
        expected_pair_monitor_sha256=pair_identity["sha256"],
        expected_monitor_name=args.expected_monitor_name,
    )
    training_git = report["matched_training_contract"]["training_git"]
    if (
        training_git.get("revision") != args.expected_training_revision
        or training_git.get("branch") != args.expected_training_branch
        or training_git.get("tracked_dirty") is not False
    ):
        raise ValueError("runtime claim guard training identity differs")
    if (
        file_identity(source_status_path) != source_status_identity
        or file_identity(source_final_path) != source_identity
        or file_identity(pair_monitor_path) != pair_identity
    ):
        raise ValueError("runtime claim guard sources changed during build")
    guard_identity = prepare_manifest(
        guard_output,
        report,
        resume=guard_output.is_file(),
        overwrite=False,
    )
    source = {
        "waiter_status": source_status_identity,
        "runtime_fairness_report": source_identity,
        "status": source_state["status"],
    }
    pair = {
        "identity": pair_identity,
        "status": "pass",
        "stage": "complete",
        "gpu_contention": report["gpu_contention"],
    }
    guard = {
        "identity": guard_identity,
        "status": report["status"],
        "decision": report["decision"],
        "claim_policy": report["claim_policy"],
        "claim_text": report["claim_text"],
    }
    publish(
        status="pass",
        detail=str(report["decision"]),
        phase="completed",
        source=source,
        pair=pair,
        guard=guard,
    )
    return 0


def run_waiter(args: argparse.Namespace) -> int:
    output_root = reject_symlink_chain(
        args.output_root,
        name="runtime claim guard output root",
    ).resolve()
    with exclusive_output_lock(output_root, role=WAITER_ROLE):
        try:
            return _run_locked(args)
        except Exception as error:
            status_output = reject_symlink_chain(
                args.status_output,
                name="runtime claim guard waiter status",
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
                name="runtime claim guard waiter PID file",
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
