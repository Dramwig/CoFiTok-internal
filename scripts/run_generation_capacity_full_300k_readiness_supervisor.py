from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from cofitok.generation.capacity_full_readiness import (
    CAPACITY_FULL_READINESS_BOUNDARY,
    validate_capacity_full_readiness,
)
from cofitok.generation.capacity_full_readiness_decision import (
    CAPACITY_FULL_ROOT_ID,
    validate_capacity_full_readiness_decision,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import file_sha256, write_json_report


SUPERVISOR_SCHEMA_VERSION = 1
SUPERVISOR_ROLE = "capacity_full_300k_readiness_supervisor"
DECISION_WAITER_ROLE = "capacity_full_300k_readiness_decision_waiter"
DECISION_DEPLOYMENT_ROLE = (
    "capacity_full_300k_readiness_decision_waiter_deployment"
)
READINESS_RUNBOOK_NAME = (
    "generation_capacity_full_300k_readiness_after_decision.sh"
)
SUPERVISOR_BOUNDARY = {
    "readiness_gpu_runtime_benchmark_allowed": True,
    "readiness_artifact_build_allowed": True,
    "training_launch_allowed_by_supervisor": False,
    "full_300k_launch_allowed_by_supervisor": False,
    "capacity_100k_checkpoint_resume_allowed": False,
    "unrelated_gpu_process_modification_allowed": False,
    "process_signaling_allowed": False,
    "promotion_or_release_allowed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_identity(project: Path) -> dict[str, Any]:
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
        "tracked_dirty": bool(run("status", "--porcelain")),
    }


def verify_supervisor_checkout(
    project: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    expected = {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    actual = _git_identity(project)
    if actual != expected:
        raise ValueError("capacity-full readiness supervisor checkout differs")
    return actual


def _same_path(left: Any, right: Path) -> bool:
    return isinstance(left, str) and Path(left).resolve() == right.resolve()


def validate_decision_waiter_deployment(
    report: Mapping[str, Any],
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    source_output_root: Path,
    full_output_root: Path,
    decision_waiter_status: Path,
    decision_path: Path,
    standing_authorization_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if file_sha256(receipt_path) != expected_receipt_sha256:
        raise ValueError("capacity-full decision deployment receipt SHA256 differs")
    expected_git = {
        "revision": expected_decision_revision,
        "tree": expected_decision_tree,
        "branch": expected_decision_branch,
        "tracked_dirty": False,
    }
    boundary = report.get("authorization_boundary")
    fresh = report.get("fresh_training_boundary")
    waiter = report.get("waiter")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "active"
        or report.get("role") != DECISION_DEPLOYMENT_ROLE
        or report.get("git") != expected_git
        or not _same_path(report.get("source_output_root"), source_output_root)
        or not _same_path(report.get("full_output_root"), full_output_root)
        or report.get("standing_authorization")
        != dict(standing_authorization_identity)
        or not isinstance(boundary, Mapping)
        or boundary.get("gpu_execution_allowed") is not False
        or boundary.get("readiness_gpu_benchmark_launch_allowed") is not False
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("process_signaling_allowed") is not False
        or not isinstance(fresh, Mapping)
        or fresh.get("capacity_100k_checkpoint_resume_allowed") is not False
        or fresh.get("full_training_state_absent") is not True
        or fresh.get("training_launch_allowed") is not False
        or fresh.get("full_300k_launch_allowed") is not False
        or not isinstance(waiter, Mapping)
        or not _same_path(waiter.get("status_path"), decision_waiter_status)
        or not _same_path(waiter.get("decision_path"), decision_path)
    ):
        raise ValueError("capacity-full decision deployment contract differs")
    for name in (
        "runbook",
        "result_waiter_deployment_receipt",
        "standing_authorization",
    ):
        expected = report.get(name)
        if not isinstance(expected, Mapping) or file_identity(expected["path"]) != dict(
            expected
        ):
            raise ValueError(
                f"capacity-full decision deployment source changed: {name}"
            )
    pid_pointer = waiter.get("pid_pointer")
    if not isinstance(pid_pointer, Mapping) or file_identity(
        pid_pointer["path"]
    ) != dict(pid_pointer):
        raise ValueError("capacity-full decision waiter PID pointer changed")
    return {
        "identity": file_identity(receipt_path),
        "git": expected_git,
        "decision_path": decision_path.resolve().as_posix(),
        "status_path": decision_waiter_status.resolve().as_posix(),
        "training_launch_allowed": False,
        "full_300k_launch_allowed": False,
    }


def validate_decision_waiter_status(
    report: Mapping[str, Any],
    *,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    source_output_root: Path,
    full_output_root: Path,
    decision_path: Path,
) -> dict[str, Any]:
    state = report.get("status")
    expected_git = {
        "revision": expected_decision_revision,
        "tree": expected_decision_tree,
        "branch": expected_decision_branch,
        "tracked_dirty": False,
    }
    boundary = report.get("authorization_boundary")
    if (
        report.get("schema_version") != 1
        or report.get("role") != DECISION_WAITER_ROLE
        or state not in {"waiting", "completed", "not_selected", "failed"}
        or report.get("git") != expected_git
        or not _same_path(report.get("source_output_root"), source_output_root)
        or not _same_path(report.get("full_output_root"), full_output_root)
        or not isinstance(boundary, Mapping)
        or boundary.get("gpu_execution_allowed") is not False
        or boundary.get("readiness_gpu_benchmark_launch_allowed") is not False
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("process_signaling_allowed") is not False
    ):
        raise ValueError("capacity-full decision waiter status differs")
    if state == "failed":
        raise RuntimeError(
            "capacity-full decision waiter failed: " + str(report.get("error"))
        )
    identity = report.get("readiness_decision")
    if state == "completed":
        if (
            report.get("detail")
            != "capacity_full_300k_readiness_decision_emitted"
            or report.get("readiness_execution_allowed") is not True
            or not isinstance(identity, Mapping)
            or file_identity(decision_path) != dict(identity)
        ):
            raise ValueError("completed capacity-full decision is not source-bound")
    elif report.get("readiness_execution_allowed") is not False or identity is not None:
        raise ValueError("incomplete capacity-full decision waiter granted execution")
    return {
        "status": state,
        "detail": report.get("detail"),
        "updated_at": report.get("updated_at"),
        "observed_recommendation": report.get("observed_recommendation"),
        "decision": dict(identity) if isinstance(identity, Mapping) else None,
        "readiness_execution_allowed": state == "completed",
    }


def validate_decision_for_readiness(
    decision_path: Path,
    *,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    expected_result_revision: str,
    expected_result_tree: str,
    expected_result_branch: str,
    full_output_root: Path,
    standing_authorization_identity: Mapping[str, Any],
) -> dict[str, Any]:
    decision = read_json_object(decision_path, name="capacity-full readiness decision")
    evidence = validate_capacity_full_readiness_decision(
        decision,
        expected_decision_revision=expected_decision_revision,
        expected_decision_tree=expected_decision_tree,
        expected_decision_branch=expected_decision_branch,
        expected_result_revision=expected_result_revision,
        expected_result_tree=expected_result_tree,
        expected_result_branch=expected_result_branch,
    )
    if (
        Path(evidence["full_output_root"]).resolve() != full_output_root.resolve()
        or decision.get("source_reports", {}).get("standing_authorization")
        != dict(standing_authorization_identity)
    ):
        raise ValueError("capacity-full readiness decision target differs")
    for name, expected in decision["source_reports"].items():
        if file_identity(expected["path"]) != expected:
            raise ValueError(f"capacity-full readiness decision source changed: {name}")
    return {
        "identity": file_identity(decision_path),
        "readiness_plan": evidence["readiness_plan"],
        "execution_authorization": evidence["execution_authorization"],
        "training_run_dirs": evidence["training_run_dirs"],
    }


def _gpu_compute_rows() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        rows.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_memory_mib": (
                    int(fields[2]) if fields[2].isdigit() else fields[2]
                ),
            }
        )
    return rows


def _readiness_processes(output_root: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,etimes=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    output_text = output_root.as_posix()
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4 or not all(value.isdigit() for value in fields[:3]):
            continue
        if int(fields[0]) == os.getpid():
            continue
        command = fields[3]
        relevant = (
            READINESS_RUNBOOK_NAME in command
            or (
                "scripts/select_generation_training_runtime.py" in command
                and output_text in command
            )
        )
        if relevant:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "elapsed_seconds": int(fields[2]),
                    "command": command,
                }
            )
    return rows


def _nonempty_training_state(full_output_root: Path) -> list[str]:
    found = []
    for name in ("cofitok", "dense_identity"):
        path = full_output_root / name
        if path.is_file() or (path.is_dir() and any(path.iterdir())):
            found.append(path.as_posix())
    return found


def validate_existing_readiness(
    *,
    project: Path,
    checkpoint_root: Path,
    full_output_root: Path,
    decision_path: Path,
    decision_sha256: str,
    expected_readiness_revision: str,
    expected_readiness_tree: str,
    expected_readiness_branch: str,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    expected_result_revision: str,
    expected_result_tree: str,
    expected_result_branch: str,
) -> dict[str, Any]:
    reports = full_output_root / "reports"
    readiness = reports / "capacity_full_300k_readiness.json"
    cofitok_config = (
        project
        / "configs/generation/"
        "imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
    )
    dense_config = (
        project
        / "configs/generation/"
        "imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json"
    )
    command = [
        sys.executable,
        str(project / "scripts/verify_generation_capacity_full_300k_readiness.py"),
        "--readiness",
        str(readiness),
        "--decision",
        str(decision_path),
        "--expected-decision-sha256",
        decision_sha256,
        "--cofitok-config",
        str(cofitok_config),
        "--dense-config",
        str(dense_config),
        "--config-validation",
        str(reports / "config_validation.json"),
        "--storage-capacity",
        str(reports / "storage_capacity.json"),
        "--runtime-selection",
        str(reports / "runtime_selection.json"),
        "--cofitok-run-dir",
        str(full_output_root / "cofitok"),
        "--dense-run-dir",
        str(full_output_root / "dense_identity"),
        "--benchmark-root",
        str(reports / "runtime_benchmark"),
        "--storage-path",
        str(checkpoint_root),
        "--expected-readiness-revision",
        expected_readiness_revision,
        "--expected-readiness-tree",
        expected_readiness_tree,
        "--expected-readiness-branch",
        expected_readiness_branch,
        "--expected-decision-revision",
        expected_decision_revision,
        "--expected-decision-tree",
        expected_decision_tree,
        "--expected-decision-branch",
        expected_decision_branch,
        "--expected-result-revision",
        expected_result_revision,
        "--expected-result-tree",
        expected_result_tree,
        "--expected-result-branch",
        expected_result_branch,
    ]
    result = subprocess.run(
        command,
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "capacity-full readiness replay failed"
            + (f": {diagnostic[-4000:]}" if diagnostic else "")
        )
    report = read_json_object(readiness, name="capacity-full readiness")
    evidence = validate_capacity_full_readiness(
        report,
        expected_readiness_revision=expected_readiness_revision,
        expected_readiness_tree=expected_readiness_tree,
        expected_readiness_branch=expected_readiness_branch,
    )
    if report.get("authorization_boundary") != CAPACITY_FULL_READINESS_BOUNDARY:
        raise ValueError("capacity-full readiness authorization boundary differs")
    return {
        "identity": file_identity(readiness),
        "full_training_launch_allowed_by_artifact": evidence[
            "full_training_launch_allowed"
        ],
        "training_launch_performed_by_supervisor": False,
        "full_300k_launch_performed_by_supervisor": False,
        "runtime_selection": evidence["runtime_selection"],
        "storage_capacity": evidence["storage_capacity"],
    }


def _status(
    *,
    status: str,
    detail: str,
    expected: Mapping[str, Any],
    decision_waiter: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    deployment: Mapping[str, Any] | None = None,
    gpu_rows: list[dict[str, Any]] | None = None,
    idle_polls: int = 0,
    attempt: int = 0,
    readiness_processes: list[dict[str, Any]] | None = None,
    child_pid: int | None = None,
    child_exit_code: int | None = None,
    readiness: Mapping[str, Any] | None = None,
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
        "decision_waiter": (
            dict(decision_waiter) if decision_waiter is not None else None
        ),
        "decision": dict(decision) if decision is not None else None,
        "decision_waiter_deployment": (
            dict(deployment) if deployment is not None else None
        ),
        "gpu_compute_rows": list(gpu_rows or []),
        "idle_polls": idle_polls,
        "required_idle_polls": int(expected["required_idle_polls"]),
        "attempt": attempt,
        "readiness_processes": list(readiness_processes or []),
        "child_pid": child_pid,
        "child_exit_code": child_exit_code,
        "readiness": dict(readiness) if readiness is not None else None,
        "readiness_execution_only": True,
        "training_launch_performed": False,
        "full_300k_launch_performed": False,
        "authorization_boundary": dict(SUPERVISOR_BOUNDARY),
        "error": error,
        "updated_at": _utc_now(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact post-100K decision and five idle GPU polls, then "
            "execute only the bounded 250M fresh-300K readiness benchmark."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-output-root", type=Path, required=True)
    parser.add_argument("--full-output-root", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--decision-waiter-status", type=Path, required=True)
    parser.add_argument(
        "--decision-waiter-deployment-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--expected-decision-waiter-deployment-receipt-sha256", required=True
    )
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--expected-readiness-revision", required=True)
    parser.add_argument("--expected-readiness-tree", required=True)
    parser.add_argument("--expected-readiness-branch", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--expected-result-revision", required=True)
    parser.add_argument("--expected-result-tree", required=True)
    parser.add_argument("--expected-result-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=7_776_000.0)
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.required_idle_polls < 1
        or args.max_attempts < 1
        or args.timeout_seconds <= 0
    ):
        parser.error("capacity-full readiness supervisor timing is invalid")
    return args


def main() -> int:
    args = _parse_args()
    project = args.project.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    source_root = args.source_output_root.resolve()
    full_root = args.full_output_root.resolve()
    decision_path = args.decision.resolve()
    waiter_status_path = args.decision_waiter_status.resolve()
    deployment_receipt_path = args.decision_waiter_deployment_receipt.resolve()
    standing_path = args.standing_authorization.resolve()
    runbook = args.runbook.resolve()
    readiness_path = full_root / "reports/capacity_full_300k_readiness.json"
    if full_root != checkpoint_root / CAPACITY_FULL_ROOT_ID:
        raise ValueError("capacity-full readiness output root differs")
    expected_decision_path = (
        source_root / "reports/capacity_full_300k_readiness_decision.json"
    )
    if decision_path != expected_decision_path:
        raise ValueError("capacity-full readiness decision path differs")
    if runbook != (project / "artifacts/runbooks" / READINESS_RUNBOOK_NAME).resolve():
        raise ValueError("capacity-full readiness runbook is not checkout-bound")
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    checkout = verify_supervisor_checkout(
        project,
        expected_revision=args.expected_readiness_revision,
        expected_tree=args.expected_readiness_tree,
        expected_branch=args.expected_readiness_branch,
    )
    standing_identity = file_identity(standing_path)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    standing = validate_standing_experiment_authorization(
        read_json_object(standing_path, name="standing experiment authorization")
    )
    if (
        standing["preserved_safety_boundaries"]
        != STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing authorization safety boundary differs")
    deployment = validate_decision_waiter_deployment(
        read_json_object(
            deployment_receipt_path,
            name="capacity-full decision waiter deployment receipt",
        ),
        receipt_path=deployment_receipt_path,
        expected_receipt_sha256=(
            args.expected_decision_waiter_deployment_receipt_sha256
        ),
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        source_output_root=source_root,
        full_output_root=full_root,
        decision_waiter_status=waiter_status_path,
        decision_path=decision_path,
        standing_authorization_identity=standing_identity,
    )
    expected = {
        "readiness_git": checkout,
        "decision_git": {
            "revision": args.expected_decision_revision,
            "tree": args.expected_decision_tree,
            "branch": args.expected_decision_branch,
            "tracked_dirty": False,
        },
        "result_git": {
            "revision": args.expected_result_revision,
            "tree": args.expected_result_tree,
            "branch": args.expected_result_branch,
            "tracked_dirty": False,
        },
        "source_output_root": source_root.as_posix(),
        "full_output_root": full_root.as_posix(),
        "standing_authorization": standing_identity,
        "decision_waiter_deployment_receipt": file_identity(
            deployment_receipt_path
        ),
        "required_idle_polls": args.required_idle_polls,
        "max_attempts": args.max_attempts,
        "readiness_execution_only": True,
        "capacity_100k_checkpoint_resume_allowed": False,
        "training_launch_allowed_by_supervisor": False,
        "full_300k_launch_allowed_by_supervisor": False,
    }
    deadline = time.monotonic() + args.timeout_seconds
    idle_polls = 0
    attempts = 0
    waiter_observation: dict[str, Any] | None = None
    decision_observation: dict[str, Any] | None = None

    def publish(**values: Any) -> None:
        write_json_report(
            args.status_output,
            _status(
                expected=expected,
                decision_waiter=waiter_observation,
                decision=decision_observation,
                deployment=deployment,
                idle_polls=idle_polls,
                attempt=attempts,
                **values,
            ),
        )

    while True:
        if time.monotonic() >= deadline:
            publish(
                status="failed",
                detail="capacity_full_readiness_supervisor_timeout",
            )
            return 1
        if not waiter_status_path.is_file():
            idle_polls = 0
            publish(
                status="waiting",
                detail="waiting_for_capacity_full_readiness_decision_status",
            )
            time.sleep(args.poll_seconds)
            continue
        waiter_observation = validate_decision_waiter_status(
            read_json_object(
                waiter_status_path,
                name="capacity-full readiness decision waiter status",
            ),
            expected_decision_revision=args.expected_decision_revision,
            expected_decision_tree=args.expected_decision_tree,
            expected_decision_branch=args.expected_decision_branch,
            source_output_root=source_root,
            full_output_root=full_root,
            decision_path=decision_path,
        )
        if waiter_observation["status"] == "not_selected":
            publish(
                status="not_selected",
                detail="capacity_result_selected_other_fail_closed_branch",
            )
            return 0
        if waiter_observation["status"] != "completed":
            idle_polls = 0
            publish(
                status="waiting",
                detail="waiting_for_source_bound_capacity_full_readiness_decision",
            )
            time.sleep(args.poll_seconds)
            continue
        decision_observation = validate_decision_for_readiness(
            decision_path,
            expected_decision_revision=args.expected_decision_revision,
            expected_decision_tree=args.expected_decision_tree,
            expected_decision_branch=args.expected_decision_branch,
            expected_result_revision=args.expected_result_revision,
            expected_result_tree=args.expected_result_tree,
            expected_result_branch=args.expected_result_branch,
            full_output_root=full_root,
            standing_authorization_identity=standing_identity,
        )
        decision_sha256 = decision_observation["identity"]["sha256"]
        training_state = _nonempty_training_state(full_root)
        if training_state:
            publish(
                status="failed",
                detail="capacity_full_training_state_exists_before_readiness",
                error=", ".join(training_state),
            )
            return 1
        if readiness_path.is_file():
            readiness = validate_existing_readiness(
                project=project,
                checkpoint_root=checkpoint_root,
                full_output_root=full_root,
                decision_path=decision_path,
                decision_sha256=decision_sha256,
                expected_readiness_revision=args.expected_readiness_revision,
                expected_readiness_tree=args.expected_readiness_tree,
                expected_readiness_branch=args.expected_readiness_branch,
                expected_decision_revision=args.expected_decision_revision,
                expected_decision_tree=args.expected_decision_tree,
                expected_decision_branch=args.expected_decision_branch,
                expected_result_revision=args.expected_result_revision,
                expected_result_tree=args.expected_result_tree,
                expected_result_branch=args.expected_result_branch,
            )
            publish(
                status="pass",
                detail="existing_capacity_full_readiness_replayed",
                readiness=readiness,
            )
            return 0
        processes = _readiness_processes(full_root)
        if processes:
            idle_polls = 0
            publish(
                status="waiting",
                detail="waiting_for_existing_capacity_full_readiness_process",
                readiness_processes=processes,
            )
            time.sleep(args.poll_seconds)
            continue
        gpu_rows = _gpu_compute_rows()
        if gpu_rows:
            idle_polls = 0
            publish(
                status="waiting",
                detail="waiting_for_gpu_idle",
                gpu_rows=gpu_rows,
            )
            time.sleep(args.poll_seconds)
            continue
        idle_polls += 1
        if idle_polls < args.required_idle_polls:
            publish(status="waiting", detail="confirming_gpu_idle", gpu_rows=[])
            time.sleep(args.poll_seconds)
            continue
        if attempts >= args.max_attempts:
            publish(
                status="failed",
                detail="capacity_full_readiness_exhausted_gpu_race_retries",
            )
            return 1
        attempts += 1
        verify_supervisor_checkout(
            project,
            expected_revision=args.expected_readiness_revision,
            expected_tree=args.expected_readiness_tree,
            expected_branch=args.expected_readiness_branch,
        )
        standing_identity_now = file_identity(standing_path)
        if standing_identity_now != standing_identity:
            raise ValueError("standing experiment authorization changed before launch")
        deployment = validate_decision_waiter_deployment(
            read_json_object(
                deployment_receipt_path,
                name="capacity-full decision waiter deployment receipt",
            ),
            receipt_path=deployment_receipt_path,
            expected_receipt_sha256=(
                args.expected_decision_waiter_deployment_receipt_sha256
            ),
            expected_decision_revision=args.expected_decision_revision,
            expected_decision_tree=args.expected_decision_tree,
            expected_decision_branch=args.expected_decision_branch,
            source_output_root=source_root,
            full_output_root=full_root,
            decision_waiter_status=waiter_status_path,
            decision_path=decision_path,
            standing_authorization_identity=standing_identity,
        )
        decision_observation = validate_decision_for_readiness(
            decision_path,
            expected_decision_revision=args.expected_decision_revision,
            expected_decision_tree=args.expected_decision_tree,
            expected_decision_branch=args.expected_decision_branch,
            expected_result_revision=args.expected_result_revision,
            expected_result_tree=args.expected_result_tree,
            expected_result_branch=args.expected_result_branch,
            full_output_root=full_root,
            standing_authorization_identity=standing_identity,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHON": sys.executable,
                "PROJECT": project.as_posix(),
                "CHECKPOINT_ROOT": checkpoint_root.as_posix(),
                "SOURCE_OUTPUT_ROOT": source_root.as_posix(),
                "DECISION": decision_path.as_posix(),
                "EXPECTED_DECISION_SHA256": decision_sha256,
                "EXPECTED_READINESS_REVISION": args.expected_readiness_revision,
                "EXPECTED_READINESS_TREE": args.expected_readiness_tree,
                "EXPECTED_READINESS_BRANCH": args.expected_readiness_branch,
                "EXPECTED_DECISION_REVISION": args.expected_decision_revision,
                "EXPECTED_DECISION_TREE": args.expected_decision_tree,
                "EXPECTED_DECISION_BRANCH": args.expected_decision_branch,
                "EXPECTED_RESULT_REVISION": args.expected_result_revision,
                "EXPECTED_RESULT_TREE": args.expected_result_tree,
                "EXPECTED_RESULT_BRANCH": args.expected_result_branch,
                "FULL_OUTPUT_ROOT": full_root.as_posix(),
            }
        )
        child = subprocess.Popen(
            ["bash", str(runbook)],
            cwd=project,
            env=environment,
        )
        while child.poll() is None:
            publish(
                status="running",
                detail="capacity_full_readiness_benchmark_running",
                child_pid=child.pid,
                gpu_rows=_gpu_compute_rows(),
            )
            time.sleep(min(args.poll_seconds, 60.0))
        exit_code = int(child.returncode)
        if exit_code == 0:
            readiness = validate_existing_readiness(
                project=project,
                checkpoint_root=checkpoint_root,
                full_output_root=full_root,
                decision_path=decision_path,
                decision_sha256=decision_sha256,
                expected_readiness_revision=args.expected_readiness_revision,
                expected_readiness_tree=args.expected_readiness_tree,
                expected_readiness_branch=args.expected_readiness_branch,
                expected_decision_revision=args.expected_decision_revision,
                expected_decision_tree=args.expected_decision_tree,
                expected_decision_branch=args.expected_decision_branch,
                expected_result_revision=args.expected_result_revision,
                expected_result_tree=args.expected_result_tree,
                expected_result_branch=args.expected_result_branch,
            )
            publish(
                status="pass",
                detail="capacity_full_readiness_completed_without_training_launch",
                child_pid=child.pid,
                child_exit_code=0,
                readiness=readiness,
            )
            return 0
        idle_polls = 0
        if exit_code == 9 and attempts < args.max_attempts:
            publish(
                status="waiting",
                detail="capacity_full_readiness_gpu_race_returned_to_wait",
                child_pid=child.pid,
                child_exit_code=exit_code,
            )
            time.sleep(args.poll_seconds)
            continue
        publish(
            status="failed",
            detail=(
                "capacity_full_readiness_exhausted_gpu_race_retries"
                if exit_code == 9
                else "capacity_full_readiness_nonretryable_failure"
            ),
            child_pid=child.pid,
            child_exit_code=exit_code,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
