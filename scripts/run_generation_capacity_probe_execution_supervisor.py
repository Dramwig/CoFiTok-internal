from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_RECOMMENDATION_ID,
    validate_capacity_probe_preparation_contract,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)
from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_RESULT_BOUNDARY,
    CAPACITY_PROBE_RESULT_ROLE,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import file_sha256, write_json_report


SUPERVISOR_SCHEMA_VERSION = 1
SUPERVISOR_ROLE = "generation_capacity_probe_execution_supervisor"
PREPARATION_WAITER_ROLE = "generation_capacity_probe_preparation_waiter"
EXECUTION_ROLE = "stability_full_data_capacity_probe_execution"
EXECUTION_RUNBOOK_NAME = (
    "generation_stability_capacity_probe_250m_10k_execute.sh"
)
RETRYABLE_STAGES = {
    "cofitok_training",
    "dense_identity_training",
    "base128_cofitok_evaluation",
    "base128_dense_identity_evaluation",
    "base256_cofitok_evaluation",
    "base256_dense_identity_evaluation",
}
NONRETRYABLE_STAGES = {
    "initialization",
    "authorization",
    "config_validation",
    "runtime_selection",
    "storage_preflight",
    "launch_receipt_build",
    "launch_receipt_replay",
    "partial_training_replay",
    "cofitok_training_validation",
    "dense_identity_training_validation",
    "result_build",
    "result_replay",
}
GPU_RACE_EXIT_CODES = {9, 12, 75}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_identity(
    project: Path,
    *,
    include_untracked: bool = True,
) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status_arguments = ["status", "--porcelain"]
    if not include_untracked:
        status_arguments.append("--untracked-files=no")
    return {
        "revision": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run(*status_arguments)),
    }


def verify_execution_checkout(
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
        raise ValueError("capacity-probe supervisor checkout identity differs")
    return actual


def validate_preparation_waiter_status(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_preparation_path: Path,
) -> dict[str, Any]:
    status = str(report.get("status", ""))
    boundary = report.get("authorization_boundary")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != PREPARATION_WAITER_ROLE
        or status not in {"waiting", "completed", "not_selected"}
        or report.get("self_git")
        != {
            "revision": expected_revision,
            "tree": expected_tree,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or not isinstance(boundary, Mapping)
        or boundary.get("capacity_probe_execution_allowed") is not False
        or boundary.get("gpu_use_allowed") is not False
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("release_allowed") is not False
        or Path(str(report.get("preparation_path", ""))).resolve()
        != expected_preparation_path.resolve()
    ):
        raise ValueError("capacity-probe preparation waiter contract differs")
    preparation = report.get("preparation")
    recommendation = report.get("observed_recommendation")
    if status == "completed":
        if (
            recommendation != CAPACITY_PROBE_RECOMMENDATION_ID
            or not isinstance(preparation, Mapping)
            or file_identity(expected_preparation_path) != dict(preparation)
        ):
            raise ValueError("completed capacity preparation is not source-bound")
    elif preparation is not None:
        raise ValueError("incomplete capacity waiter cannot bind a preparation")
    return {
        "status": status,
        "detail": report.get("detail"),
        "polls": int(report.get("polls", 0)),
        "recommendation": recommendation,
        "preparation": dict(preparation) if isinstance(preparation, Mapping) else None,
        "updated_at": report.get("updated_at"),
    }


def validate_preparation_for_execution(
    report: Mapping[str, Any],
    *,
    expected_output_root: str,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    selection = validate_capacity_probe_preparation_contract(
        report,
        expected_output_root=expected_output_root,
    )
    if report.get("git") != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity preparation Git identity differs")
    return selection


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
    rows: list[dict[str, Any]] = []
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


def _capacity_processes(output_root: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,etimes=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    output_text = output_root.as_posix()
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4 or not all(value.isdigit() for value in fields[:3]):
            continue
        command = fields[3]
        relevant = (
            command.startswith("bash ")
            and EXECUTION_RUNBOOK_NAME in command
        ) or (
            "scripts/train_generation.py" in command
            and "stability_capacity_probe" in command
        ) or (
            "run_generation_training_watchdog.py" in command
            and output_text in command
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


def _execution_stage(report: Mapping[str, Any] | None) -> str:
    if not isinstance(report, Mapping):
        return "initialization"
    stage = report.get("stage")
    return str(stage) if isinstance(stage, str) and stage else "initialization"


def classify_execution_exit(
    *,
    exit_code: int,
    stage: str,
) -> str:
    if exit_code in GPU_RACE_EXIT_CODES:
        return "wait"
    if stage in RETRYABLE_STAGES:
        return "retry"
    if stage in NONRETRYABLE_STAGES or stage == "launch_guard":
        return "fail"
    return "fail"


def validate_completed_execution(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    result_path: Path,
) -> dict[str, Any]:
    result_identity = file_identity(result_path) if result_path.is_file() else None
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != EXECUTION_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != "completed"
        or report.get("detail")
        != "bounded capacity-probe evidence verified; no 100K/300K authorization created"
        or report.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or report.get("capacity_probe_only") is not True
        or int(report.get("intentional_training_stop_step", -1)) != 10_000
        or report.get("configured_100k_completion_allowed") is not False
        or report.get("full_300k_launch_allowed") is not False
        or report.get("report_is_promotion_gate") is not False
        or report.get("release_authorization_allowed") is not False
        or report.get("result") != result_identity
    ):
        raise ValueError("capacity-probe completed execution contract differs")
    result = read_json_object(result_path, name="capacity probe result")
    if (
        result.get("status") != "completed"
        or result.get("role") != CAPACITY_PROBE_RESULT_ROLE
        or result.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or result.get("authorization_boundary") != CAPACITY_PROBE_RESULT_BOUNDARY
        or result.get("claim_policy", {}).get(
            "formal_generation_claim_allowed"
        )
        is not False
    ):
        raise ValueError("capacity-probe completed result boundary differs")
    return {
        "status": "verified",
        "result": result_identity,
        "configured_100k_completion_allowed": False,
        "full_300k_launch_allowed": False,
    }


def _status(
    *,
    status: str,
    detail: str,
    expected: Mapping[str, Any],
    preparation_waiter: Mapping[str, Any] | None = None,
    preparation: Mapping[str, Any] | None = None,
    gpu_rows: list[dict[str, Any]] | None = None,
    idle_polls: int = 0,
    attempt: int = 0,
    child_pid: int | None = None,
    child_exit_code: int | None = None,
    execution_stage: str | None = None,
    capacity_processes: list[dict[str, Any]] | None = None,
    completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SUPERVISOR_SCHEMA_VERSION,
        "role": SUPERVISOR_ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected": dict(expected),
        "preparation_waiter": (
            dict(preparation_waiter) if preparation_waiter is not None else None
        ),
        "preparation": dict(preparation) if preparation is not None else None,
        "gpu_compute_rows": list(gpu_rows or []),
        "idle_polls": idle_polls,
        "required_idle_polls": int(expected["required_idle_polls"]),
        "attempt": attempt,
        "child_pid": child_pid,
        "child_exit_code": child_exit_code,
        "execution_stage": execution_stage,
        "capacity_processes": list(capacity_processes or []),
        "completion": dict(completion) if completion is not None else None,
        "authorization_boundary": {
            "capacity_probe_execution_allowed": True,
            "scope_limited_to_exact_250m_10k_probe": True,
            "unrelated_gpu_process_modification_allowed": False,
            "configured_100k_completion_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
            "release_allowed": False,
        },
        "updated_at": _utc_now(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the source-bound capacity preparation and five idle GPU "
            "polls, then supervise only the bounded matched 250M/10K probe."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--preparation-project", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--preparation-waiter-status", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-preparation-revision", required=True)
    parser.add_argument("--expected-preparation-tree", required=True)
    parser.add_argument("--expected-preparation-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-seconds", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=7_776_000.0)
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.required_idle_polls < 1
        or args.max_attempts < 1
        or args.retry_seconds <= 0
        or args.timeout_seconds <= 0
    ):
        parser.error("capacity supervisor timing arguments are invalid")
    return args


def main() -> int:
    args = _parse_args()
    project = args.project.resolve()
    preparation_project = args.preparation_project.resolve()
    preparation_path = args.preparation.resolve()
    output_root = args.output_root.resolve()
    report_root = output_root / "reports"
    execution_status_path = report_root / "execution_status.json"
    result_path = report_root / "capacity_probe_result.json"
    runbook = args.runbook.resolve()
    if runbook != (project / "artifacts/runbooks" / EXECUTION_RUNBOOK_NAME).resolve():
        raise ValueError("capacity supervisor runbook is not checkout-bound")
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    checkout = verify_execution_checkout(
        project,
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
    )
    # The preparation waiter legitimately writes untracked PID/log files beside
    # its checkout. Only tracked drift can change the replayed preparation code.
    preparation_checkout = _git_identity(
        preparation_project,
        include_untracked=False,
    )
    if preparation_checkout != {
        "revision": args.expected_preparation_revision,
        "tree": args.expected_preparation_tree,
        "branch": args.expected_preparation_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity preparation checkout identity differs")
    if (
        not args.standing_authorization.is_file()
        or file_sha256(args.standing_authorization)
        != args.expected_standing_authorization_sha256
    ):
        raise ValueError("standing experiment authorization identity differs")
    standing = validate_standing_experiment_authorization(
        read_json_object(
            args.standing_authorization,
            name="standing experiment authorization",
        )
    )
    if standing["preserved_safety_boundaries"] != (
        STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing authorization safety boundary differs")
    expected = {
        "execution_git": checkout,
        "preparation_git": preparation_checkout,
        "output_root": output_root.as_posix(),
        "standing_authorization": file_identity(args.standing_authorization),
        "required_idle_polls": args.required_idle_polls,
        "max_attempts": args.max_attempts,
        "intentional_training_stop_step": 10_000,
        "configured_100k_completion_allowed": False,
        "full_300k_launch_allowed": False,
    }
    deadline = time.monotonic() + args.timeout_seconds
    idle_polls = 0
    attempts = 0
    waiter_observation: dict[str, Any] | None = None
    preparation_observation: dict[str, Any] | None = None

    def publish(**values: Any) -> None:
        write_json_report(
            args.status_output,
            _status(
                expected=expected,
                preparation_waiter=waiter_observation,
                preparation=preparation_observation,
                idle_polls=idle_polls,
                attempt=attempts,
                **values,
            ),
        )

    while True:
        if execution_status_path.is_file():
            execution_status = read_json_object(
                execution_status_path,
                name="capacity probe execution status",
            )
            if execution_status.get("status") == "completed":
                completion = validate_completed_execution(
                    execution_status,
                    expected_revision=args.expected_revision,
                    expected_branch=args.expected_branch,
                    result_path=result_path,
                )
                publish(
                    status="pass",
                    detail="bounded_capacity_probe_completed",
                    completion=completion,
                )
                return 0

        if not args.preparation_waiter_status.is_file():
            publish(
                status="waiting",
                detail="waiting_for_capacity_preparation_waiter_status",
            )
        else:
            waiter_observation = validate_preparation_waiter_status(
                read_json_object(
                    args.preparation_waiter_status,
                    name="capacity preparation waiter status",
                ),
                expected_revision=args.expected_preparation_revision,
                expected_tree=args.expected_preparation_tree,
                expected_branch=args.expected_preparation_branch,
                expected_preparation_path=preparation_path,
            )
            if waiter_observation["status"] == "not_selected":
                publish(
                    status="not_selected",
                    detail="quality_bridge_selected_another_evidence_branch",
                )
                return 0
            if waiter_observation["status"] != "completed":
                publish(
                    status="waiting",
                    detail="waiting_for_source_bound_capacity_preparation",
                )
            else:
                preparation = read_json_object(
                    preparation_path,
                    name="capacity probe preparation",
                )
                selection = validate_preparation_for_execution(
                    preparation,
                    expected_output_root=output_root.as_posix(),
                    expected_revision=args.expected_preparation_revision,
                    expected_branch=args.expected_preparation_branch,
                )
                preparation_observation = {
                    "identity": file_identity(preparation_path),
                    "selection": selection,
                }
                processes = _capacity_processes(output_root)
                if processes:
                    idle_polls = 0
                    publish(
                        status="waiting",
                        detail="waiting_for_existing_capacity_probe_process",
                        capacity_processes=processes,
                    )
                else:
                    gpu_rows = _gpu_compute_rows()
                    if gpu_rows:
                        idle_polls = 0
                        publish(
                            status="waiting",
                            detail="waiting_for_gpu_idle",
                            gpu_rows=gpu_rows,
                        )
                    else:
                        idle_polls += 1
                        if idle_polls < args.required_idle_polls:
                            publish(
                                status="waiting",
                                detail="confirming_gpu_idle",
                                gpu_rows=[],
                            )
                        else:
                            if attempts >= args.max_attempts:
                                publish(
                                    status="failed",
                                    detail="capacity_probe_exhausted_bounded_retries",
                                )
                                return 1
                            attempts += 1
                            environment = os.environ.copy()
                            environment.update(
                                {
                                    "PYTHON": sys.executable,
                                    "CHECKPOINT_ROOT": args.checkpoint_root.resolve().as_posix(),
                                    "STANDING_AUTHORIZATION": args.standing_authorization.resolve().as_posix(),
                                    "EXPECTED_STANDING_AUTHORIZATION_SHA256": args.expected_standing_authorization_sha256,
                                    "EXPECTED_TARGET_REVISION": args.expected_revision,
                                    "EXPECTED_TARGET_BRANCH": args.expected_branch,
                                    "EXPECTED_PREPARATION_SHA256": preparation_observation["identity"]["sha256"],
                                    "CAPACITY_PROBE_EXECUTION_ALLOWED": "true",
                                    "PREPARATION_PROJECT": preparation_project.as_posix(),
                                    "EXPECTED_PREPARATION_REVISION": args.expected_preparation_revision,
                                    "EXPECTED_PREPARATION_BRANCH": args.expected_preparation_branch,
                                }
                            )
                            optional = {
                                "EXPECTED_EXECUTION_AUTHORIZATION_SHA256": report_root / "execution_authorization.json",
                                "EXPECTED_LAUNCH_RECEIPT_SHA256": report_root / "launch_receipt.json",
                                "EXPECTED_RESULT_SHA256": result_path,
                            }
                            for name, path in optional.items():
                                if path.is_file():
                                    environment[name] = file_sha256(path)
                            status_before = (
                                file_identity(execution_status_path)
                                if execution_status_path.is_file()
                                else None
                            )
                            child = subprocess.Popen(
                                ["bash", str(runbook)],
                                cwd=project,
                                env=environment,
                            )
                            publish(
                                status="running",
                                detail="capacity_probe_controller_running",
                                child_pid=child.pid,
                            )
                            exit_code = child.wait()
                            status_after = (
                                file_identity(execution_status_path)
                                if execution_status_path.is_file()
                                else None
                            )
                            execution_status = None
                            if status_after is not None and status_after != status_before:
                                execution_status = read_json_object(
                                    execution_status_path,
                                    name="capacity probe execution status",
                                )
                            if exit_code == 0:
                                if not isinstance(execution_status, Mapping):
                                    raise RuntimeError(
                                        "capacity runbook exited without execution status"
                                    )
                                completion = validate_completed_execution(
                                    execution_status,
                                    expected_revision=args.expected_revision,
                                    expected_branch=args.expected_branch,
                                    result_path=result_path,
                                )
                                publish(
                                    status="pass",
                                    detail="bounded_capacity_probe_completed",
                                    child_pid=child.pid,
                                    child_exit_code=0,
                                    completion=completion,
                                )
                                return 0
                            stage = _execution_stage(execution_status)
                            decision = classify_execution_exit(
                                exit_code=exit_code,
                                stage=stage,
                            )
                            idle_polls = 0
                            if decision == "fail":
                                publish(
                                    status="failed",
                                    detail="capacity_probe_nonretryable_failure",
                                    child_pid=child.pid,
                                    child_exit_code=exit_code,
                                    execution_stage=stage,
                                )
                                return 1
                            if decision == "wait":
                                attempts -= 1
                                publish(
                                    status="waiting",
                                    detail="capacity_probe_launch_race_returned_to_wait",
                                    child_pid=child.pid,
                                    child_exit_code=exit_code,
                                    execution_stage=stage,
                                )
                            elif attempts >= args.max_attempts:
                                publish(
                                    status="failed",
                                    detail="capacity_probe_exhausted_bounded_retries",
                                    child_pid=child.pid,
                                    child_exit_code=exit_code,
                                    execution_stage=stage,
                                )
                                return 1
                            else:
                                publish(
                                    status="retrying",
                                    detail="capacity_probe_recoverable_stage_retry_scheduled",
                                    child_pid=child.pid,
                                    child_exit_code=exit_code,
                                    execution_stage=stage,
                                )
                                time.sleep(args.retry_seconds * attempts)

        if time.monotonic() >= deadline:
            publish(status="failed", detail="capacity_probe_supervisor_timeout")
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
