from __future__ import annotations

import argparse
import hashlib
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


GUARD_SCHEMA_VERSION = 1
GUARD_ROLE = "generation_quality_bridge_controller_identity_guard"
BINDING_SCHEMA_VERSION = 1
BINDING_ROLE = "generation_quality_bridge_controller_identity_binding"
DEPLOYMENT_SCHEMA_VERSION = 1
DEPLOYMENT_ROLE = "generation_quality_bridge_controller_identity_guard_deployment"
EXECUTION_ROLE = "stability_full_data_quality_bridge_execution"
SCOPE = {
    "cpu_only": True,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "process_signals_allowed": False,
    "controller_restart_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "diagnostic_non_authorizing": True,
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


def _decode_argv(raw: bytes) -> list[str]:
    if not raw:
        raise ValueError("controller cmdline is empty")
    values = raw.split(b"\0")
    if values and values[-1] == b"":
        values.pop()
    argv = [value.decode("utf-8", errors="replace") for value in values]
    if not argv or any(not value for value in argv):
        raise ValueError("controller argv is malformed")
    return argv


def read_process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any] | None:
    process = proc_root / str(pid)
    if not process.is_dir():
        return None
    try:
        raw_cmdline = (process / "cmdline").read_bytes()
        stat_tail = (process / "stat").read_text(encoding="ascii").rsplit(
            ")", 1
        )[1].split()
        executable = os.readlink(process / "exe")
        cwd = os.readlink(process / "cwd")
        start_ticks = int(stat_tail[19])
        ppid = int(stat_tail[1])
        argv = _decode_argv(raw_cmdline)
    except (
        FileNotFoundError,
        IndexError,
        OSError,
        PermissionError,
        ProcessLookupError,
        ValueError,
    ):
        return None
    return {
        "pid": pid,
        "ppid": ppid,
        "start_ticks": start_ticks,
        "executable": executable,
        "cwd": cwd,
        "argv": argv,
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
    }


def process_identity_mismatches(
    observed: Mapping[str, Any] | None,
    *,
    expected: Mapping[str, Any],
) -> list[str]:
    if observed is None:
        return ["process_missing"]
    mismatches = []
    for field in (
        "pid",
        "start_ticks",
        "executable",
        "cwd",
        "cmdline_sha256",
    ):
        if observed.get(field) != expected.get(field):
            mismatches.append(field)
    argv = observed.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) != 2
        or Path(str(argv[0])).name != "bash"
    ):
        mismatches.append("argv_shape")
    return mismatches


def validate_execution_status(
    report: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    status = str(report.get("status", ""))
    git = report.get("git")
    if (
        report.get("schema_version") != 1
        or report.get("role") != EXECUTION_ROLE
        or status not in {"running", "failed", "completed"}
        or int(report.get("pid", -1)) != expected_pid
        or not isinstance(git, Mapping)
        or git.get("revision") != expected_revision
        or git.get("branch") != expected_branch
        or git.get("tracked_dirty") is not False
        or report.get("quality_bridge_only") is not True
        or report.get("full_training_launch_allowed") is not False
        or report.get("full_300k_launch_allowed") is not False
        or report.get("report_is_promotion_gate") is not False
    ):
        raise ValueError("quality bridge execution status contract differs")
    exit_code = report.get("exit_code")
    if status == "failed" and not isinstance(exit_code, int):
        raise ValueError("failed execution status lacks an exit code")
    if status in {"running", "completed"} and exit_code is not None:
        raise ValueError("non-failed execution status has an exit code")
    return {
        "status": status,
        "detail": report.get("detail"),
        "exit_code": exit_code,
        "pid": expected_pid,
        "updated_at": report.get("updated_at"),
    }


def summarize_pair_monitor(
    report: Mapping[str, Any],
    *,
    expected_monitor_name: str,
    expected_revision: str,
    expected_branch: str,
    bound_pid: int,
) -> dict[str, Any]:
    git = report.get("git")
    processes = report.get("processes")
    runbook_rows = processes.get("runbook") if isinstance(processes, Mapping) else []
    if (
        report.get("monitor") != expected_monitor_name
        or not isinstance(git, Mapping)
        or git.get("revision") != expected_revision
        or git.get("branch") != expected_branch
        or git.get("tracked_dirty") is not False
        or not isinstance(runbook_rows, list)
    ):
        raise ValueError("quality bridge pair monitor contract differs")
    reported_pids = []
    for row in runbook_rows:
        raw_pid, separator, _ = str(row).strip().partition(" ")
        if not separator:
            continue
        try:
            reported_pids.append(int(raw_pid))
        except ValueError:
            continue
    unique_pids = sorted(set(reported_pids))
    return {
        "status": report.get("status"),
        "stage": report.get("stage"),
        "updated_at": report.get("updated_at"),
        "reported_runbook_pids": unique_pids,
        "bound_pid_reported": bound_pid in unique_pids,
        "nonbound_reported_pids": [pid for pid in unique_pids if pid != bound_pid],
        "substring_candidate_contamination_observed": any(
            pid != bound_pid for pid in unique_pids
        ),
    }


def _write_pid(path: Path, *, expected: Mapping[str, Any]) -> None:
    write_json_report(
        path,
        {
            "schema_version": 1,
            "role": GUARD_ROLE,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "expected_control_revision": expected["control_git"]["revision"],
            "bound_controller_pid": expected["controller"]["pid"],
            "bound_controller_start_ticks": expected["controller"]["start_ticks"],
            "started_at": _utc_now(),
        },
    )


def _remove_pid(path: Path) -> None:
    if not path.is_file():
        return
    try:
        report = read_json_object(path, name="controller identity guard PID")
    except ValueError:
        return
    if report.get("role") == GUARD_ROLE and int(report.get("pid", -1)) == os.getpid():
        path.unlink()


def _status(
    *,
    status: str,
    detail: str,
    expected: Mapping[str, Any],
    binding_identity: Mapping[str, Any],
    deployment_identity: Mapping[str, Any],
    execution: Mapping[str, Any] | None,
    observed: Mapping[str, Any] | None,
    mismatches: list[str],
    identity_loss_polls: int,
    pair_monitor: Mapping[str, Any] | None,
    quality_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": GUARD_SCHEMA_VERSION,
        "role": GUARD_ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected": dict(expected),
        "binding": dict(binding_identity),
        "deployment_receipt": dict(deployment_identity),
        "execution": dict(execution) if execution is not None else None,
        "observed_controller": dict(observed) if observed is not None else None,
        "identity_mismatches": list(mismatches),
        "identity_loss_polls": identity_loss_polls,
        "pair_monitor": dict(pair_monitor) if pair_monitor is not None else None,
        "quality_result": dict(quality_result) if quality_result is not None else None,
        "scope": SCOPE,
        "updated_at": _utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously verify one exact quality-bridge controller identity "
            "without signaling, restarting, or launching any process."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--execution-status", type=Path, required=True)
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--pair-monitor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--binding-output", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--expected-controller-pid", type=int, required=True)
    parser.add_argument("--expected-controller-start-ticks", type=int, required=True)
    parser.add_argument("--expected-controller-executable", required=True)
    parser.add_argument("--expected-controller-cwd", required=True)
    parser.add_argument("--expected-controller-cmdline-sha256", required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-monitor-name", required=True)
    parser.add_argument("--required-identity-loss-polls", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.expected_controller_pid < 1
        or args.expected_controller_start_ticks < 1
        or args.required_identity_loss_polls < 1
        or args.poll_seconds <= 0.0
        or args.timeout_seconds <= 0.0
        or len(args.expected_controller_cmdline_sha256) != 64
    ):
        parser.error("controller identity guard arguments are invalid")
    return args


def _run_locked(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(
        args.project,
        name="controller identity guard checkout",
    ).resolve()
    quality_root = reject_symlink_chain(
        args.quality_output_root,
        name="quality bridge output root",
    ).resolve()
    execution_status_path = reject_symlink_chain(
        args.execution_status,
        name="quality bridge execution status",
    ).resolve()
    quality_result_path = reject_symlink_chain(
        args.quality_result,
        name="quality bridge result",
    ).resolve()
    pair_monitor_path = reject_symlink_chain(
        args.pair_monitor,
        name="quality bridge pair monitor",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="controller identity guard output root",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="controller identity guard status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="controller identity guard PID file",
    ).resolve()
    binding_output = reject_symlink_chain(
        args.binding_output,
        name="controller identity binding",
    ).resolve()
    deployment_output = reject_symlink_chain(
        args.deployment_receipt_output,
        name="controller identity guard deployment receipt",
    ).resolve()
    proc_root = args.proc_root.resolve()
    if (
        output_root == quality_root
        or not all(
            _is_within(path, quality_root)
            for path in (
                execution_status_path,
                quality_result_path,
                pair_monitor_path,
                output_root,
            )
        )
        or not all(
            _is_within(path, output_root)
            for path in (
                status_output,
                pid_file,
                binding_output,
                deployment_output,
            )
        )
    ):
        raise ValueError("controller identity guard path scope differs")

    control_git = {
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    if _git_identity(project) != control_git:
        raise ValueError("controller identity guard checkout differs")
    expected_controller = {
        "pid": args.expected_controller_pid,
        "start_ticks": args.expected_controller_start_ticks,
        "executable": args.expected_controller_executable,
        "cwd": args.expected_controller_cwd,
        "cmdline_sha256": args.expected_controller_cmdline_sha256,
    }
    initial = read_process_identity(
        args.expected_controller_pid,
        proc_root=proc_root,
    )
    initial_mismatches = process_identity_mismatches(
        initial,
        expected=expected_controller,
    )
    if initial_mismatches:
        raise ValueError(
            "initial controller identity differs: " + ", ".join(initial_mismatches)
        )
    assert initial is not None
    expected = {
        "control_git": control_git,
        "controller": expected_controller,
        "training_git": {
            "revision": args.expected_training_revision,
            "branch": args.expected_training_branch,
            "tracked_dirty": False,
        },
        "monitor_name": args.expected_monitor_name,
        "sources": {
            "execution_status": execution_status_path.as_posix(),
            "quality_result": quality_result_path.as_posix(),
            "pair_monitor": pair_monitor_path.as_posix(),
        },
        "output_root": output_root.as_posix(),
    }
    binding = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "role": BINDING_ROLE,
        "status": "pass",
        "controller": initial,
        "expected": expected,
        "scope": SCOPE,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    binding_identity = prepare_manifest(
        binding_output,
        binding,
        resume=binding_output.is_file(),
        overwrite=False,
    )
    deployment = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "role": DEPLOYMENT_ROLE,
        "status": "pass",
        "control_git": control_git,
        "binding": binding_identity,
        "expected": expected,
        "scope": SCOPE,
    }
    deployment_identity = prepare_manifest(
        deployment_output,
        deployment,
        resume=deployment_output.is_file(),
        overwrite=False,
    )
    _write_pid(pid_file, expected=expected)
    deadline = time.monotonic() + args.timeout_seconds
    identity_loss_polls = 0

    def publish(
        *,
        status: str,
        detail: str,
        execution: Mapping[str, Any] | None,
        observed: Mapping[str, Any] | None,
        mismatches: list[str],
        pair: Mapping[str, Any] | None,
        quality_result: Mapping[str, Any] | None = None,
    ) -> None:
        write_json_report(
            status_output,
            _status(
                status=status,
                detail=detail,
                expected=expected,
                binding_identity=binding_identity,
                deployment_identity=deployment_identity,
                execution=execution,
                observed=observed,
                mismatches=mismatches,
                identity_loss_polls=identity_loss_polls,
                pair_monitor=pair,
                quality_result=quality_result,
            ),
        )

    while True:
        if _git_identity(project) != control_git:
            raise ValueError("controller identity guard checkout changed")
        if file_identity(binding_output) != binding_identity:
            raise ValueError("controller identity binding changed")
        execution = validate_execution_status(
            read_json_object(
                execution_status_path,
                name="quality bridge execution status",
            ),
            expected_pid=args.expected_controller_pid,
            expected_revision=args.expected_training_revision,
            expected_branch=args.expected_training_branch,
        )
        pair = summarize_pair_monitor(
            read_json_object(pair_monitor_path, name="quality bridge pair monitor"),
            expected_monitor_name=args.expected_monitor_name,
            expected_revision=args.expected_training_revision,
            expected_branch=args.expected_training_branch,
            bound_pid=args.expected_controller_pid,
        )
        if execution["status"] == "completed":
            if not quality_result_path.is_file():
                publish(
                    status="failed",
                    detail="completed_execution_lacks_quality_result",
                    execution=execution,
                    observed=None,
                    mismatches=[],
                    pair=pair,
                )
                return 1
            result_identity = file_identity(quality_result_path)
            publish(
                status="pass",
                detail="bound_controller_reached_terminal_verified_execution_status",
                execution=execution,
                observed=read_process_identity(
                    args.expected_controller_pid,
                    proc_root=proc_root,
                ),
                mismatches=[],
                pair=pair,
                quality_result=result_identity,
            )
            return 0
        if execution["status"] == "failed":
            publish(
                status="failed",
                detail="bound_quality_bridge_execution_failed",
                execution=execution,
                observed=read_process_identity(
                    args.expected_controller_pid,
                    proc_root=proc_root,
                ),
                mismatches=[],
                pair=pair,
            )
            return 1

        observed = read_process_identity(
            args.expected_controller_pid,
            proc_root=proc_root,
        )
        mismatches = process_identity_mismatches(
            observed,
            expected=expected_controller,
        )
        if mismatches:
            identity_loss_polls += 1
            terminal = identity_loss_polls >= args.required_identity_loss_polls
            publish(
                status="failed" if terminal else "waiting",
                detail=(
                    "bound_controller_identity_lost_before_terminal_completion"
                    if terminal
                    else "confirming_bound_controller_identity_loss"
                ),
                execution=execution,
                observed=observed,
                mismatches=mismatches,
                pair=pair,
            )
            if terminal:
                return 1
        else:
            identity_loss_polls = 0
            publish(
                status="observing",
                detail="exact_bound_controller_identity_is_active",
                execution=execution,
                observed=observed,
                mismatches=[],
                pair=pair,
            )
        if args.once:
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            publish(
                status="failed",
                detail="controller_identity_guard_timeout",
                execution=execution,
                observed=observed,
                mismatches=mismatches,
                pair=pair,
            )
            return 1
        time.sleep(min(args.poll_seconds, remaining))


def main() -> None:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="controller identity guard output root",
    ).resolve()
    try:
        with exclusive_output_lock(output_root, role=GUARD_ROLE):
            try:
                raise SystemExit(_run_locked(args))
            finally:
                _remove_pid(args.pid_file.resolve())
    except OutputLockError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
