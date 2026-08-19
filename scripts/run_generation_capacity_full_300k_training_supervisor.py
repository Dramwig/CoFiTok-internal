from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.capacity_full_training_launch import (
    CAPACITY_FULL_TRAINING_LAUNCH_BOUNDARY,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.process_monitoring import publish_child_heartbeat
from cofitok.reporting import write_json_report


SUPERVISOR_SCHEMA_VERSION = 1
SUPERVISOR_ROLE = "capacity_full_300k_training_supervisor"
RUNBOOK_NAME = "generation_capacity_full_300k_execute.sh"
RETRYABLE_EXIT_CODES = frozenset({9, 12, 15, 87, 88, 89, 137, 143})
SUPERVISOR_BOUNDARY = {
    "fresh_matched_300k_training_launch_allowed_after_exact_receipt": True,
    "capacity_100k_checkpoint_resume_allowed": False,
    "exact_target_checkpoint_resume_allowed": True,
    "training_authorization_is_quality_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "frozen_promotion_gate_replaced": False,
    "release_authorization_allowed": False,
    "process_signaling_allowed": False,
    "unrelated_gpu_process_modification_allowed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(project: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _require_git(
    project: Path,
    *,
    revision: str,
    tree: str,
    branch: str,
    label: str,
) -> dict[str, Any]:
    expected = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    actual = _git(project)
    if actual != expected:
        raise ValueError(f"capacity-full {label} checkout differs")
    return actual


def _gpu_rows() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi compute-process query failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _runbook_processes(runbook: Path) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    target = runbook.resolve().as_posix()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().split(b"\0")
            arguments = [part.decode("utf-8", errors="replace") for part in parts if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not arguments:
            continue
        executable = Path(arguments[0]).name
        if executable not in {"bash", "sh"}:
            continue
        resolved_arguments = []
        for argument in arguments[1:]:
            try:
                resolved_arguments.append(Path(argument).resolve().as_posix())
            except (OSError, ValueError):
                continue
        if target in resolved_arguments:
            matches.append({"pid": int(entry.name), "command": arguments})
    return sorted(matches, key=lambda row: int(row["pid"]))


def _readiness_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = read_json_object(path, name="capacity-full readiness supervisor status")
    if (
        report.get("schema_version") != 1
        or report.get("role") != "capacity_full_300k_readiness_supervisor"
        or report.get("readiness_execution_only") is not True
        or report.get("training_launch_performed") is not False
        or report.get("full_300k_launch_performed") is not False
    ):
        raise ValueError("capacity-full readiness supervisor status differs")
    return report


def _status(
    *,
    status: str,
    detail: str,
    expected: Mapping[str, Any],
    readiness_status: Mapping[str, Any] | None = None,
    launch_receipt: Mapping[str, Any] | None = None,
    gpu_rows: list[str] | None = None,
    idle_polls: int = 0,
    attempt: int = 0,
    runbook_processes: list[dict[str, Any]] | None = None,
    child_pid: int | None = None,
    child_process_group_id: int | None = None,
    child_exit_code: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SUPERVISOR_SCHEMA_VERSION,
        "role": SUPERVISOR_ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected": dict(expected),
        "readiness_supervisor_status": (
            dict(readiness_status) if readiness_status is not None else None
        ),
        "training_launch_receipt": (
            dict(launch_receipt) if launch_receipt is not None else None
        ),
        "gpu_compute_rows": list(gpu_rows or []),
        "idle_polls": idle_polls,
        "required_idle_polls": int(expected["required_idle_polls"]),
        "attempt": attempt,
        "runbook_processes": list(runbook_processes or []),
        "child_pid": child_pid,
        "child_process_group_id": child_process_group_id,
        "child_exit_code": child_exit_code,
        "training_launch_performed": attempt > 0,
        "formal_generation_completion_claimed": False,
        "authorization_boundary": dict(SUPERVISOR_BOUNDARY),
        "error": error,
        "updated_at": _utc_now(),
    }


def _receipt_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--execution-project",
        str(args.execution_project.resolve()),
        "--training-project",
        str(args.training_project.resolve()),
        "--formal-project",
        str(args.formal_project.resolve()),
        "--readiness",
        str(args.readiness.resolve()),
        "--readiness-supervisor-status",
        str(args.readiness_supervisor_status.resolve()),
        "--readiness-deployment-receipt",
        str(args.readiness_deployment_receipt.resolve()),
        "--expected-readiness-deployment-sha256",
        args.expected_readiness_deployment_sha256,
        "--readiness-deployment-clarification",
        str(args.readiness_deployment_clarification.resolve()),
        "--expected-readiness-clarification-sha256",
        args.expected_readiness_clarification_sha256,
        "--standing-authorization",
        str(args.standing_authorization.resolve()),
        "--expected-standing-authorization-sha256",
        args.expected_standing_authorization_sha256,
        "--cofitok-config",
        str(args.cofitok_config.resolve()),
        "--dense-config",
        str(args.dense_config.resolve()),
        "--launch-storage-capacity",
        str(args.launch_storage_capacity.resolve()),
        "--cofitok-run-dir",
        str(args.cofitok_run_dir.resolve()),
        "--dense-run-dir",
        str(args.dense_run_dir.resolve()),
        "--expected-execution-revision",
        args.expected_execution_revision,
        "--expected-execution-tree",
        args.expected_execution_tree,
        "--expected-execution-branch",
        args.expected_execution_branch,
        "--expected-training-revision",
        args.expected_training_revision,
        "--expected-training-tree",
        args.expected_training_tree,
        "--expected-training-branch",
        args.expected_training_branch,
        "--expected-readiness-revision",
        args.expected_readiness_revision,
        "--expected-readiness-tree",
        args.expected_readiness_tree,
        "--expected-readiness-branch",
        args.expected_readiness_branch,
    ]


def _build_or_verify_receipt(
    args: argparse.Namespace,
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    execution = args.execution_project.resolve()
    training = args.training_project.resolve()
    receipt = args.training_launch_receipt.resolve()
    storage = args.launch_storage_capacity.resolve()
    common = _receipt_arguments(args)
    if receipt.is_file():
        identity = file_identity(receipt)
        subprocess.run(
            [
                sys.executable,
                str(
                    execution
                    / "scripts/verify_generation_capacity_full_300k_training_launch_receipt.py"
                ),
                *common,
                "--receipt",
                str(receipt),
                "--expected-receipt-sha256",
                identity["sha256"],
            ],
            cwd=execution,
            env=dict(environment),
            check=True,
            capture_output=True,
            text=True,
        )
        return identity
    if storage.exists():
        raise ValueError("capacity-full launch storage exists without a receipt")
    if any(
        path.is_dir() and any(path.iterdir())
        for path in (args.cofitok_run_dir.resolve(), args.dense_run_dir.resolve())
    ):
        raise ValueError("capacity-full training state exists without a receipt")
    storage.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(training / "scripts/check_generation_storage_capacity.py"),
            "--path",
            str(args.checkpoint_root.resolve()),
            "--output",
            str(storage),
            "--stage",
            "full_training",
            "--reference-checkpoint",
            str(args.reference_cofitok.resolve()),
            "--reference-checkpoint",
            str(args.reference_dense.resolve()),
            "--checkpoint-count",
            "16",
            "--checkpoint-size-multiplier",
            "1.0",
            "--sample-count",
            "116640",
            "--estimated-sample-kib",
            "256",
            "--additional-gib",
            "16",
            "--safety-margin-gib",
            "64",
        ],
        cwd=training,
        env={**dict(environment), "PYTHONPATH": str(training / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        subprocess.run(
            [
                sys.executable,
                str(
                    execution
                    / "scripts/build_generation_capacity_full_300k_training_launch_receipt.py"
                ),
                *common,
                "--output",
                str(receipt),
            ],
            cwd=execution,
            env=dict(environment),
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if not receipt.is_file():
            storage.unlink(missing_ok=True)
    return file_identity(receipt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for exact capacity-full readiness, build a source-bound "
            "experimental training receipt, confirm GPU idleness, and supervise "
            "only the fresh matched 300K execution."
        )
    )
    parser.add_argument("--execution-project", type=Path, required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-output-root", type=Path, required=True)
    parser.add_argument("--full-output-root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--readiness-supervisor-status", type=Path, required=True)
    parser.add_argument("--readiness-deployment-receipt", type=Path, required=True)
    parser.add_argument("--expected-readiness-deployment-sha256", required=True)
    parser.add_argument("--readiness-deployment-clarification", type=Path, required=True)
    parser.add_argument("--expected-readiness-clarification-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--launch-storage-capacity", type=Path, required=True)
    parser.add_argument("--training-launch-receipt", type=Path, required=True)
    parser.add_argument("--cofitok-run-dir", type=Path, required=True)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument("--reference-cofitok", type=Path, required=True)
    parser.add_argument("--reference-dense", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--child-log", type=Path, required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-readiness-revision", required=True)
    parser.add_argument("--expected-readiness-tree", required=True)
    parser.add_argument("--expected-readiness-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--retry-seconds", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=31_536_000.0)
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.required_idle_polls < 1
        or args.max_attempts < 1
        or args.retry_seconds < 0
        or args.timeout_seconds <= 0
    ):
        parser.error("capacity-full training supervisor timing is invalid")
    return args


def main() -> int:
    args = _parse_args()
    execution = args.execution_project.resolve()
    training = args.training_project.resolve()
    runbook = args.runbook.resolve()
    full_root = args.full_output_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    if full_root != checkpoint_root / "stability_capacity_full_300k_v1":
        raise ValueError("capacity-full training output root differs")
    if runbook != execution / "artifacts/runbooks" / RUNBOOK_NAME:
        raise ValueError("capacity-full training runbook is not execution-checkout bound")
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    execution_git = _require_git(
        execution,
        revision=args.expected_execution_revision,
        tree=args.expected_execution_tree,
        branch=args.expected_execution_branch,
        label="execution",
    )
    training_git = _require_git(
        training,
        revision=args.expected_training_revision,
        tree=args.expected_training_tree,
        branch=args.expected_training_branch,
        label="training",
    )
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    deployment_identity = file_identity(args.readiness_deployment_receipt)
    if deployment_identity["sha256"] != args.expected_readiness_deployment_sha256:
        raise ValueError("capacity-full readiness deployment SHA256 differs")
    clarification_identity = file_identity(args.readiness_deployment_clarification)
    if clarification_identity["sha256"] != args.expected_readiness_clarification_sha256:
        raise ValueError("capacity-full readiness clarification SHA256 differs")
    expected = {
        "execution_git": execution_git,
        "training_git": training_git,
        "readiness_git": {
            "revision": args.expected_readiness_revision,
            "tree": args.expected_readiness_tree,
            "branch": args.expected_readiness_branch,
            "tracked_dirty": False,
        },
        "source_output_root": args.source_output_root.resolve().as_posix(),
        "full_output_root": full_root.as_posix(),
        "standing_authorization": standing_identity,
        "readiness_deployment_receipt": deployment_identity,
        "readiness_deployment_clarification": clarification_identity,
        "required_idle_polls": args.required_idle_polls,
        "max_attempts": args.max_attempts,
    }
    deadline = time.monotonic() + args.timeout_seconds
    idle_polls = 0
    attempts = 0
    readiness_status: dict[str, Any] | None = None
    receipt_identity: dict[str, Any] | None = None

    def publish(**values: Any) -> None:
        write_json_report(
            args.status_output,
            _status(
                expected=expected,
                readiness_status=readiness_status,
                launch_receipt=receipt_identity,
                idle_polls=idle_polls,
                attempt=attempts,
                **values,
            ),
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(execution / "src"), str(training / "src"))
    )
    while True:
        if time.monotonic() >= deadline:
            publish(status="failed", detail="capacity_full_training_supervisor_timeout")
            return 1
        readiness_status = _readiness_state(args.readiness_supervisor_status)
        if readiness_status is None:
            idle_polls = 0
            publish(status="waiting", detail="waiting_for_capacity_full_readiness_status")
            time.sleep(args.poll_seconds)
            continue
        readiness_state = readiness_status.get("status")
        if readiness_state in {"failed", "not_selected"}:
            publish(
                status="not_selected" if readiness_state == "not_selected" else "failed",
                detail="capacity_full_readiness_did_not_authorize_training_receipt",
            )
            return 0 if readiness_state == "not_selected" else 1
        if readiness_state != "pass" or not args.readiness.is_file():
            idle_polls = 0
            publish(status="waiting", detail="waiting_for_passed_capacity_full_readiness")
            time.sleep(args.poll_seconds)
            continue
        _require_git(
            execution,
            revision=args.expected_execution_revision,
            tree=args.expected_execution_tree,
            branch=args.expected_execution_branch,
            label="execution",
        )
        _require_git(
            training,
            revision=args.expected_training_revision,
            tree=args.expected_training_tree,
            branch=args.expected_training_branch,
            label="training",
        )
        receipt_identity = _build_or_verify_receipt(args, environment=environment)
        receipt = read_json_object(
            args.training_launch_receipt,
            name="capacity-full training launch receipt",
        )
        if receipt.get("authorization_boundary") != CAPACITY_FULL_TRAINING_LAUNCH_BOUNDARY:
            raise ValueError("capacity-full training receipt boundary differs")
        processes = _runbook_processes(runbook)
        if processes:
            idle_polls = 0
            publish(
                status="observing",
                detail="observing_existing_capacity_full_training_runbook",
                runbook_processes=processes,
            )
            time.sleep(args.poll_seconds)
            continue
        rows = _gpu_rows()
        if rows:
            idle_polls = 0
            publish(status="waiting", detail="waiting_for_gpu_idle", gpu_rows=rows)
            time.sleep(args.poll_seconds)
            continue
        idle_polls += 1
        if idle_polls < args.required_idle_polls:
            publish(status="waiting", detail="confirming_gpu_idle")
            time.sleep(args.poll_seconds)
            continue
        if attempts >= args.max_attempts:
            publish(
                status="failed",
                detail="capacity_full_training_exhausted_retry_budget",
            )
            return 1
        attempts += 1
        receipt_identity = _build_or_verify_receipt(args, environment=environment)
        child_environment = dict(environment)
        child_environment.update(
            {
                "EXECUTION_PROJECT": execution.as_posix(),
                "TRAINING_PROJECT": training.as_posix(),
                "FORMAL_PROJECT": args.formal_project.resolve().as_posix(),
                "PYTHON": sys.executable,
                "CHECKPOINT_ROOT": checkpoint_root.as_posix(),
                "SOURCE_OUTPUT_ROOT": args.source_output_root.resolve().as_posix(),
                "FULL_OUTPUT_ROOT": full_root.as_posix(),
                "STANDING_AUTHORIZATION": args.standing_authorization.resolve().as_posix(),
                "EXPECTED_EXECUTION_REVISION": args.expected_execution_revision,
                "EXPECTED_EXECUTION_TREE": args.expected_execution_tree,
                "EXPECTED_EXECUTION_BRANCH": args.expected_execution_branch,
                "EXPECTED_TRAINING_REVISION": args.expected_training_revision,
                "EXPECTED_TRAINING_TREE": args.expected_training_tree,
                "EXPECTED_TRAINING_BRANCH": args.expected_training_branch,
                "EXPECTED_READINESS_REVISION": args.expected_readiness_revision,
                "EXPECTED_READINESS_TREE": args.expected_readiness_tree,
                "EXPECTED_READINESS_BRANCH": args.expected_readiness_branch,
                "EXPECTED_READINESS_DEPLOYMENT_SHA256": args.expected_readiness_deployment_sha256,
                "EXPECTED_READINESS_CLARIFICATION_SHA256": args.expected_readiness_clarification_sha256,
                "EXPECTED_STANDING_AUTHORIZATION_SHA256": args.expected_standing_authorization_sha256,
                "EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256": receipt_identity["sha256"],
            }
        )
        args.child_log.parent.mkdir(parents=True, exist_ok=True)
        with args.child_log.open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                ["bash", str(runbook)],
                cwd=execution,
                env=child_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            group_id = child.pid
            heartbeat_error_reported = False
            while child.poll() is None:
                heartbeat_error_reported = publish_child_heartbeat(
                    lambda: publish(
                        status="running",
                        detail="capacity_full_fresh_matched_300k_running",
                        child_pid=child.pid,
                        child_process_group_id=group_id,
                        gpu_rows=_gpu_rows(),
                    ),
                    error_reported=heartbeat_error_reported,
                )
                time.sleep(min(args.poll_seconds, 60.0))
        exit_code = int(child.returncode)
        if exit_code == 0:
            publish(
                status="pass",
                detail="capacity_full_fresh_matched_300k_training_completed",
                child_pid=child.pid,
                child_process_group_id=group_id,
                child_exit_code=0,
            )
            return 0
        idle_polls = 0
        if exit_code not in RETRYABLE_EXIT_CODES:
            publish(
                status="failed",
                detail="capacity_full_training_nonretryable_failure",
                child_pid=child.pid,
                child_process_group_id=group_id,
                child_exit_code=exit_code,
            )
            return 1
        publish(
            status="waiting",
            detail="capacity_full_training_retryable_exit",
            child_pid=child.pid,
            child_process_group_id=group_id,
            child_exit_code=exit_code,
        )
        time.sleep(args.retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
