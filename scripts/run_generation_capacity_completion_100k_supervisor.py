from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any, Mapping

from cofitok.generation.capacity_completion_decision import (
    CAPACITY_COMPLETION_RECOMMENDATION_ID,
    validate_capacity_completion_decision,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import file_sha256, write_json_report


SUPERVISOR_SCHEMA_VERSION = 1
SUPERVISOR_ROLE = "generation_capacity_completion_100k_supervisor"
DECISION_WAITER_ROLE = "capacity_completion_100k_source_replay_waiter"
EXECUTION_ROLE = "stability_full_data_capacity_completion_100k_execution"
RUNBOOK_NAME = "generation_capacity_completion_250m_100k_execute.sh"
CAPACITY_COMPLETION_RUNS = ("base256_cofitok", "base256_dense_identity")
RACE_EXIT_CODES = {9, 12, 75}
RETRYABLE_STAGES = {
    "cofitok_training",
    "cofitok_milestone_evaluation",
    "dense_training",
    "dense_milestone_evaluation",
    "milestone_build",
    "cofitok_terminal_evaluation",
    "dense_terminal_evaluation",
    "class_fidelity_qualification",
    "completion_replay",
}
DECISION_WAITER_BOUNDARY = {
    "cpu_only_waiter": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "process_signaling_allowed": False,
    "source_reports_must_be_physically_replayed": True,
    "existing_result_or_decision_must_be_byte_equivalent": True,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
}
EXECUTION_BOUNDARY = {
    "matched_250m_resume_to_100000_allowed": True,
    "configured_100k_training_complete": True,
    "terminal_evidence_complete": True,
    "additional_training_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "new_source_compatible_result_required": True,
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


def _gpu_rows() -> list[dict[str, Any]]:
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


def _is_relevant_command(command: str, *, output_root: str) -> bool:
    if command.startswith("bash ") and RUNBOOK_NAME in command:
        return True
    if output_root not in command:
        return False
    return any(
        marker in command
        for marker in (
            "scripts/train_generation.py",
            "scripts/generate_samples.py",
            "scripts/preflight_generation_sampling.py",
            "scripts/evaluate_generation_checkpoint.py",
            "scripts/evaluate_generation_metrics.py",
            "scripts/evaluate_generation_class_fidelity.py",
            "generation_full_milestone_eval.sh",
        )
    )


def _active_processes(output_root: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,etimes=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    root = output_root.as_posix()
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4 or not all(value.isdigit() for value in fields[:3]):
            continue
        command = fields[3]
        if _is_relevant_command(command, output_root=root):
            rows.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "elapsed_seconds": int(fields[2]),
                    "command": command,
                }
            )
    return rows


def _child_activity(
    output_root: Path,
    *,
    execution_status: Path,
    child_started_at: float,
    now: float | None = None,
) -> dict[str, Any]:
    """Return source-progress activity without counting supervisor status writes."""
    completion = output_root / "reports/capacity_completion_100k"
    candidates = [
        execution_status,
        completion / "source_checkpoint_archive.json",
        completion / "storage_capacity.json",
        completion / "launch_receipt.json",
        completion / "training/cofitok.json",
        completion / "training/dense_identity.json",
        completion / "milestone_step_00100000.json",
        completion / "class_fidelity/qualification_report.json",
    ]
    for run_name in CAPACITY_COMPLETION_RUNS:
        run = output_root / run_name
        milestone = run / "milestones/step_00100000"
        milestone_samples = milestone / "samples_2048_ddim50_cfg15"
        terminal = run / "terminal_100k"
        terminal_samples = terminal / "samples_10000_ddim100_cfg15"
        candidates.extend(
            [
                run / "train_metrics.jsonl",
                run / "latest.json",
                run / "training_report.json",
                milestone / "sampling_preflight.json",
                milestone / "checkpoint_eval/checkpoint_evaluation_report.json",
                milestone_samples / "sampling_progress.json",
                milestone_samples / "sampling_report.json",
                milestone_samples / "metrics/generation_metrics_report.json",
                terminal / "sampling_preflight.json",
                terminal / "checkpoint_eval/checkpoint_evaluation_report.json",
                terminal_samples / "sampling_progress.json",
                terminal_samples / "sampling_report.json",
                terminal_samples / "metrics/generation_metrics_report.json",
                terminal_samples / "class_fidelity/class_fidelity_report.json",
            ]
        )
    newest_path: Path | None = None
    newest_mtime = child_started_at
    for path in candidates:
        if not path.is_file():
            continue
        mtime = path.stat().st_mtime
        if mtime >= newest_mtime:
            newest_path = path
            newest_mtime = mtime
    observed_at = time.time() if now is None else now
    return {
        "newest_path": newest_path.resolve().as_posix() if newest_path else None,
        "newest_mtime_epoch": newest_mtime,
        "age_seconds": max(0.0, observed_at - newest_mtime),
    }


def _terminate_owned_child(
    child: subprocess.Popen[Any],
    *,
    grace_seconds: float,
) -> dict[str, Any]:
    """Stop only the process group created for this supervisor's child."""
    if child.poll() is not None:
        return {
            "signal": None,
            "forced_kill": False,
            "returncode": int(child.returncode),
        }
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.05, grace_seconds)))
    forced = child.poll() is None
    if forced:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    returncode = child.wait()
    return {
        "signal": "SIGTERM",
        "forced_kill": forced,
        "returncode": int(returncode),
    }


def validate_decision_waiter_status(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: Path,
    expected_decision_path: Path,
    expected_result_path: Path,
) -> dict[str, Any]:
    status = report.get("status")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != DECISION_WAITER_ROLE
        or status not in {"waiting", "completed", "failed"}
        or report.get("git")
        != {
            "revision": expected_revision,
            "tree": expected_tree,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or Path(str(report.get("output_root", ""))).resolve()
        != expected_output_root.resolve()
        or report.get("authorization_boundary") != DECISION_WAITER_BOUNDARY
    ):
        raise ValueError("capacity completion decision waiter contract differs")
    decision = report.get("capacity_completion_decision")
    result = report.get("capacity_scaling_result")
    sources = report.get("source_identities")
    execution_authorized = report.get("execution_authorized")
    if status == "completed":
        if (
            not isinstance(decision, Mapping)
            or not isinstance(result, Mapping)
            or not isinstance(sources, Mapping)
            or type(execution_authorized) is not bool
            or file_identity(expected_decision_path) != dict(decision)
            or file_identity(expected_result_path) != dict(result)
        ):
            raise ValueError("terminal capacity completion decision is not source-bound")
        for identity in sources.values():
            if not isinstance(identity, Mapping):
                raise ValueError("capacity completion source identity is malformed")
            if file_identity(identity["path"]) != dict(identity):
                raise ValueError("capacity completion decision source changed")
    elif status == "waiting":
        if any(value is not None for value in (decision, result, sources, execution_authorized)):
            raise ValueError("waiting capacity completion decision binds evidence")
    elif not isinstance(report.get("error"), str):
        raise ValueError("failed capacity completion waiter has no error")
    return {
        "status": status,
        "detail": report.get("detail"),
        "decision": dict(decision) if isinstance(decision, Mapping) else None,
        "result": dict(result) if isinstance(result, Mapping) else None,
        "source_identities": (
            {str(key): dict(value) for key, value in sources.items()}
            if isinstance(sources, Mapping)
            else None
        ),
        "execution_authorized": execution_authorized,
        "error": report.get("error"),
    }


def _completed_execution(
    path: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = read_json_object(path, name="capacity completion execution status")
    if report.get("status") != "completed":
        return None
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != EXECUTION_ROLE
        or report.get("stage") != "complete"
        or report.get("git")
        != {
            "revision": expected_revision,
            "tree": expected_tree,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or Path(str(report.get("output_root", ""))).resolve()
        != expected_output_root.resolve()
        or report.get("authorization_boundary") != EXECUTION_BOUNDARY
    ):
        raise ValueError("completed capacity completion execution contract differs")
    identities: list[Mapping[str, Any]] = []
    for key in (
        "launch_receipt",
        "source_checkpoint_archive",
        "milestone_100000",
        "class_fidelity_qualification",
    ):
        identity = report.get(key)
        if not isinstance(identity, Mapping):
            raise ValueError(f"completed capacity completion {key} is missing")
        identities.append(identity)
    methods = report.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {
        "cofitok",
        "dense_identity",
    }:
        raise ValueError("completed capacity completion method set differs")
    for method in methods.values():
        if not isinstance(method, Mapping) or set(method) != {
            "training_validation",
            "sampling_preflight",
            "generation_metrics",
            "checkpoint_evaluation",
            "class_fidelity",
        }:
            raise ValueError("completed capacity completion method evidence differs")
        for identity in method.values():
            if not isinstance(identity, Mapping):
                raise ValueError("completed method evidence identity is missing")
            identities.append(identity)
    for identity in identities:
        if file_identity(identity["path"]) != dict(identity):
            raise ValueError("completed capacity completion artifact changed")
    return report


def _status(
    args: argparse.Namespace,
    *,
    state: str,
    detail: str,
    attempts: int,
    idle_polls: int,
    execution_git: Mapping[str, Any],
    decision_observation: Mapping[str, Any] | None,
    active_processes: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    child_activity: Mapping[str, Any] | None = None,
    child_termination: Mapping[str, Any] | None = None,
    child_pid: int | None = None,
    child_exit_code: int | None = None,
    execution_result: Mapping[str, Any] | None = None,
) -> None:
    write_json_report(
        args.status_output,
        {
            "schema_version": SUPERVISOR_SCHEMA_VERSION,
            "role": SUPERVISOR_ROLE,
            "status": state,
            "detail": detail,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "updated_at": _utc_now(),
            "execution_git": dict(execution_git),
            "attempt": attempts,
            "idle_polls": idle_polls,
            "required_idle_polls": args.required_idle_polls,
            "poll_seconds": args.poll_seconds,
            "child_pid": child_pid,
            "child_exit_code": child_exit_code,
            "decision_observation": (
                dict(decision_observation)
                if isinstance(decision_observation, Mapping)
                else None
            ),
            "active_processes": active_processes,
            "gpu_compute_rows": gpu_rows,
            "child_activity": (
                dict(child_activity)
                if isinstance(child_activity, Mapping)
                else None
            ),
            "child_termination": (
                dict(child_termination)
                if isinstance(child_termination, Mapping)
                else None
            ),
            "execution_result": (
                dict(execution_result)
                if isinstance(execution_result, Mapping)
                else None
            ),
            "authorization_boundary": {
                "matched_250m_resume_from_50000_to_100000_allowed": True,
                "fresh_training_allowed": False,
                "unrelated_gpu_processes_must_not_be_signaled": True,
                "full_300k_launch_allowed": False,
                "promotion_or_release_allowed": False,
                "new_terminal_result_required": True,
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the source-replayed 100K completion decision and five idle "
            "GPU polls, then supervise only the exact matched 50K-to-100K segment."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--decision-project", type=Path, required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--decision-waiter-status", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--scaling-result", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--child-log", type=Path, required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--expected-scaling-decision-revision", required=True)
    parser.add_argument("--expected-scaling-decision-branch", required=True)
    parser.add_argument("--expected-scaling-execution-revision", required=True)
    parser.add_argument("--expected-scaling-execution-tree", required=True)
    parser.add_argument("--expected-scaling-execution-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-seconds", type=float, default=120.0)
    parser.add_argument("--stall-seconds", type=float, default=1_800.0)
    parser.add_argument("--termination-grace-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=10_368_000.0)
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.required_idle_polls < 1
        or args.max_attempts < 1
        or args.retry_seconds <= 0
        or args.stall_seconds <= 0
        or args.termination_grace_seconds <= 0
        or args.timeout_seconds <= 0
    ):
        parser.error("capacity completion supervisor timing arguments are invalid")
    return args


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    decision_project = args.decision_project.resolve()
    training_project = args.training_project.resolve()
    output_root = args.output_root.resolve()
    runbook = args.runbook.resolve()
    execution_git = _git_identity(project)
    expected_execution = {
        "revision": args.expected_execution_revision,
        "tree": args.expected_execution_tree,
        "branch": args.expected_execution_branch,
        "tracked_dirty": False,
    }
    if execution_git != expected_execution:
        raise ValueError("capacity completion supervisor checkout identity differs")
    expected_decision_git = {
        "revision": args.expected_decision_revision,
        "tree": args.expected_decision_tree,
        "branch": args.expected_decision_branch,
        "tracked_dirty": False,
    }
    if _git_identity(decision_project) != expected_decision_git:
        raise ValueError("capacity completion decision checkout identity differs")
    expected_training_git = {
        "revision": args.expected_training_revision,
        "tree": args.expected_training_tree,
        "branch": args.expected_training_branch,
        "tracked_dirty": False,
    }
    if _git_identity(training_project) != expected_training_git:
        raise ValueError("capacity completion training checkout identity differs")
    expected_runbook = (project / "artifacts/runbooks" / RUNBOOK_NAME).resolve()
    if runbook != expected_runbook or not runbook.is_file():
        raise ValueError("capacity completion supervisor runbook is not checkout-bound")
    expected_output_root = (
        args.checkpoint_root.resolve()
        / "stability_full_data_100k_capacity_probe_250m_10k_v1"
    )
    if output_root != expected_output_root:
        raise ValueError("capacity completion output root differs")
    reports = output_root / "reports"
    expected_decision = reports / "capacity_completion_100k_decision.json"
    expected_result = reports / "capacity_scaling_50k_result.json"
    if args.decision.resolve() != expected_decision.resolve():
        raise ValueError("capacity completion decision path differs")
    if args.scaling_result.resolve() != expected_result.resolve():
        raise ValueError("capacity scaling result path differs")
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    standing = validate_standing_experiment_authorization(
        read_json_object(
            args.standing_authorization,
            name="standing experiment authorization",
        )
    )
    if standing["preserved_safety_boundaries"] != (
        STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing experiment safety boundary differs")

    execution_status = reports / "capacity_completion_100k/execution_status.json"
    launch_receipt = reports / "capacity_completion_100k/launch_receipt.json"
    deadline = time.monotonic() + args.timeout_seconds
    idle_polls = 0
    attempts = 0
    observation: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        completed = _completed_execution(
            execution_status,
            expected_revision=args.expected_execution_revision,
            expected_tree=args.expected_execution_tree,
            expected_branch=args.expected_execution_branch,
            expected_output_root=output_root,
        )
        if completed is not None:
            _status(
                args,
                state="completed",
                detail="matched_250m_step_100k_execution_verified",
                attempts=attempts,
                idle_polls=idle_polls,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=[],
                gpu_rows=_gpu_rows(),
                execution_result=file_identity(execution_status),
            )
            return 0
        if not args.decision_waiter_status.is_file():
            _status(
                args,
                state="waiting",
                detail="waiting_for_capacity_completion_decision_waiter_status",
                attempts=attempts,
                idle_polls=0,
                execution_git=execution_git,
                decision_observation=None,
                active_processes=_active_processes(output_root),
                gpu_rows=_gpu_rows(),
            )
            time.sleep(args.poll_seconds)
            continue
        observation = validate_decision_waiter_status(
            read_json_object(
                args.decision_waiter_status,
                name="capacity completion decision waiter status",
            ),
            expected_revision=args.expected_decision_revision,
            expected_tree=args.expected_decision_tree,
            expected_branch=args.expected_decision_branch,
            expected_output_root=output_root,
            expected_decision_path=args.decision,
            expected_result_path=args.scaling_result,
        )
        if observation["status"] == "waiting":
            _status(
                args,
                state="waiting",
                detail="waiting_for_source_replayed_capacity_completion_decision",
                attempts=attempts,
                idle_polls=0,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=_active_processes(output_root),
                gpu_rows=_gpu_rows(),
            )
            time.sleep(args.poll_seconds)
            continue
        if observation["status"] == "failed":
            _status(
                args,
                state="failed",
                detail="capacity_completion_decision_waiter_failed",
                attempts=attempts,
                idle_polls=0,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=_active_processes(output_root),
                gpu_rows=_gpu_rows(),
            )
            return 1
        decision = read_json_object(args.decision, name="capacity completion decision")
        evidence = validate_capacity_completion_decision(
            decision,
            expected_decision_revision=args.expected_decision_revision,
            expected_decision_tree=args.expected_decision_tree,
            expected_decision_branch=args.expected_decision_branch,
        )
        if observation["execution_authorized"] is not evidence["execution_authorized"]:
            raise ValueError("capacity completion waiter authorization differs")
        recommendation = evidence["recommended_next_stage"]
        if evidence["execution_authorized"] is not True:
            _status(
                args,
                state="not_selected",
                detail="capacity_scaling_result_did_not_select_100k_completion",
                attempts=attempts,
                idle_polls=0,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=[],
                gpu_rows=_gpu_rows(),
            )
            return 0
        if recommendation.get("id") != CAPACITY_COMPLETION_RECOMMENDATION_ID:
            raise ValueError("capacity completion recommendation cannot launch")

        active = _active_processes(output_root)
        gpu = _gpu_rows()
        if active:
            idle_polls = 0
            _status(
                args,
                state="running",
                detail="observing_existing_capacity_completion_execution",
                attempts=attempts,
                idle_polls=idle_polls,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=active,
                gpu_rows=gpu,
            )
            time.sleep(args.poll_seconds)
            continue
        if gpu:
            idle_polls = 0
            _status(
                args,
                state="waiting",
                detail="waiting_for_gpu_idle_without_modifying_unrelated_processes",
                attempts=attempts,
                idle_polls=idle_polls,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=[],
                gpu_rows=gpu,
            )
            time.sleep(args.poll_seconds)
            continue
        idle_polls += 1
        if idle_polls < args.required_idle_polls:
            _status(
                args,
                state="waiting",
                detail="confirming_consecutive_idle_gpu_polls",
                attempts=attempts,
                idle_polls=idle_polls,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=[],
                gpu_rows=[],
            )
            time.sleep(args.poll_seconds)
            continue

        attempts += 1
        environment = dict(os.environ)
        environment.update(
            {
                "PROJECT": project.as_posix(),
                "DECISION_PROJECT": decision_project.as_posix(),
                "TRAINING_PROJECT": training_project.as_posix(),
                "PYTHON": str(
                    Path(
                        os.environ.get(
                            "COFITOK_PYTHON",
                            "/root/autodl-tmp/conda/envs/pf-vlm/bin/python",
                        )
                    )
                ),
                "CHECKPOINT_ROOT": args.checkpoint_root.resolve().as_posix(),
                "STANDING_AUTHORIZATION": args.standing_authorization.resolve().as_posix(),
                "EXPECTED_STANDING_AUTHORIZATION_SHA256": args.expected_standing_authorization_sha256,
                "EXPECTED_COMPLETION_DECISION_SHA256": file_sha256(args.decision),
                "EXPECTED_SCALING_RESULT_SHA256": file_sha256(args.scaling_result),
                "EXPECTED_DECISION_REVISION": args.expected_decision_revision,
                "EXPECTED_DECISION_TREE": args.expected_decision_tree,
                "EXPECTED_DECISION_BRANCH": args.expected_decision_branch,
                "EXPECTED_SCALING_DECISION_REVISION": args.expected_scaling_decision_revision,
                "EXPECTED_SCALING_DECISION_BRANCH": args.expected_scaling_decision_branch,
                "EXPECTED_SCALING_EXECUTION_REVISION": args.expected_scaling_execution_revision,
                "EXPECTED_SCALING_EXECUTION_TREE": args.expected_scaling_execution_tree,
                "EXPECTED_SCALING_EXECUTION_BRANCH": args.expected_scaling_execution_branch,
                "EXPECTED_EXECUTION_REVISION": args.expected_execution_revision,
                "EXPECTED_EXECUTION_TREE": args.expected_execution_tree,
                "EXPECTED_EXECUTION_BRANCH": args.expected_execution_branch,
                "EXPECTED_TRAINING_REVISION": args.expected_training_revision,
                "EXPECTED_TRAINING_TREE": args.expected_training_tree,
                "EXPECTED_TRAINING_BRANCH": args.expected_training_branch,
                "EXPECTED_LAUNCH_RECEIPT_SHA256": (
                    file_sha256(launch_receipt) if launch_receipt.is_file() else ""
                ),
            }
        )
        args.child_log.parent.mkdir(parents=True, exist_ok=True)
        with args.child_log.open("ab") as log:
            child = subprocess.Popen(
                ["bash", str(runbook)],
                cwd=project,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        child_started_at = time.time()
        forced_reason: str | None = None
        forced_stage: str | None = None
        termination: dict[str, Any] | None = None
        while child.poll() is None:
            activity = _child_activity(
                output_root,
                execution_status=execution_status,
                child_started_at=child_started_at,
            )
            current_stage = None
            if execution_status.is_file():
                current_stage = read_json_object(
                    execution_status,
                    name="capacity completion execution status",
                ).get("stage")
            if time.monotonic() >= deadline:
                forced_reason = "capacity_completion_supervisor_timeout"
            elif float(activity["age_seconds"]) > args.stall_seconds:
                forced_reason = "capacity_completion_child_stalled"
            if forced_reason is not None:
                forced_stage = str(current_stage) if current_stage else None
                _status(
                    args,
                    state="recovering",
                    detail=f"{forced_reason}_at_{forced_stage or 'unknown'}",
                    attempts=attempts,
                    idle_polls=idle_polls,
                    execution_git=execution_git,
                    decision_observation=observation,
                    active_processes=_active_processes(output_root),
                    gpu_rows=_gpu_rows(),
                    child_activity=activity,
                    child_pid=child.pid,
                )
                termination = _terminate_owned_child(
                    child,
                    grace_seconds=args.termination_grace_seconds,
                )
                break
            _status(
                args,
                state="running",
                detail="capacity_completion_child_running",
                attempts=attempts,
                idle_polls=idle_polls,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=_active_processes(output_root),
                gpu_rows=_gpu_rows(),
                child_activity=activity,
                child_pid=child.pid,
            )
            time.sleep(args.poll_seconds)
        code = int(child.returncode)
        completed = _completed_execution(
            execution_status,
            expected_revision=args.expected_execution_revision,
            expected_tree=args.expected_execution_tree,
            expected_branch=args.expected_execution_branch,
            expected_output_root=output_root,
        )
        if code == 0 and completed is not None:
            _status(
                args,
                state="completed",
                detail="matched_250m_step_100k_execution_verified",
                attempts=attempts,
                idle_polls=idle_polls,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=[],
                gpu_rows=_gpu_rows(),
                child_pid=child.pid,
                child_exit_code=code,
                child_termination=termination,
                execution_result=file_identity(execution_status),
            )
            return 0
        stage = forced_stage
        if stage is None and execution_status.is_file():
            stage = read_json_object(
                execution_status,
                name="capacity completion execution status",
            ).get("stage")
        if forced_reason == "capacity_completion_supervisor_timeout":
            _status(
                args,
                state="failed",
                detail="capacity_completion_supervisor_timeout",
                attempts=attempts,
                idle_polls=idle_polls,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=_active_processes(output_root),
                gpu_rows=_gpu_rows(),
                child_pid=child.pid,
                child_exit_code=code,
                child_termination=termination,
            )
            return 124
        retryable = code in RACE_EXIT_CODES or stage in RETRYABLE_STAGES
        if retryable and attempts < args.max_attempts:
            idle_polls = 0
            _status(
                args,
                state="waiting",
                detail=f"retrying_capacity_completion_after_{stage or 'launch_race'}",
                attempts=attempts,
                idle_polls=idle_polls,
                execution_git=execution_git,
                decision_observation=observation,
                active_processes=_active_processes(output_root),
                gpu_rows=_gpu_rows(),
                child_pid=child.pid,
                child_exit_code=code,
                child_termination=termination,
            )
            time.sleep(args.retry_seconds)
            continue
        _status(
            args,
            state="failed",
            detail=f"capacity_completion_child_failed_at_{stage or 'unknown'}",
            attempts=attempts,
            idle_polls=idle_polls,
            execution_git=execution_git,
            decision_observation=observation,
            active_processes=_active_processes(output_root),
            gpu_rows=_gpu_rows(),
            child_pid=child.pid,
            child_exit_code=code,
            child_termination=termination,
        )
        return code if code != 0 else 1
    _status(
        args,
        state="failed",
        detail="capacity_completion_supervisor_timeout",
        attempts=attempts,
        idle_polls=idle_polls,
        execution_git=execution_git,
        decision_observation=observation,
        active_processes=_active_processes(output_root),
        gpu_rows=_gpu_rows(),
    )
    return 124


if __name__ == "__main__":
    raise SystemExit(main())
