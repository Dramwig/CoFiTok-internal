from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.release import verify_generation_release_receipt
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report
from cofitok.training.authorization import capture_generation_training_authorization

try:
    from scripts.audit_generation_capacity_full_completion import (
        POSTEVAL_SUPERVISOR_BOUNDARY,
        PROFILE,
        RELEASE_CHECK,
        validate_posteval_supervisor_deployment,
        validate_posteval_supervisor_status,
    )
    from scripts.run_generation_capacity_full_300k_posteval_supervisor import (
        TRAINING_AUTHORIZATION_DECISION,
        TRAINING_AUTHORIZATION_STAGE,
        validate_posteval_result,
        validate_training_supervisor_deployment,
    )
except ModuleNotFoundError:
    from audit_generation_capacity_full_completion import (
        POSTEVAL_SUPERVISOR_BOUNDARY,
        PROFILE,
        RELEASE_CHECK,
        validate_posteval_supervisor_deployment,
        validate_posteval_supervisor_status,
    )
    from run_generation_capacity_full_300k_posteval_supervisor import (
        TRAINING_AUTHORIZATION_DECISION,
        TRAINING_AUTHORIZATION_STAGE,
        validate_posteval_result,
        validate_training_supervisor_deployment,
    )


SUPERVISOR_SCHEMA_VERSION = 1
SUPERVISOR_ROLE = "capacity_full_300k_finalization_supervisor"
RUNBOOK_NAME = "generation_capacity_full_300k_finalize_after_gate.sh"
RETRYABLE_EXIT_CODES = frozenset({9, 75, 87, 88, 89, 137, 143})
SUPERVISOR_BOUNDARY = {
    "training_launch_allowed": False,
    "posteval_launch_allowed": False,
    "inference_export_allowed_after_passed_full_gate": True,
    "completion_audit_allowed_after_verified_exports": True,
    "release_receipt_allowed_after_passed_completion_audit": True,
    "formal_generation_completion_claim_allowed_after_release_receipt": True,
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
        raise ValueError(f"capacity-full finalization {label} checkout differs")
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


def _runbook_processes(runbooks: list[Path]) -> list[dict[str, Any]]:
    targets = {path.resolve().as_posix() for path in runbooks}
    matches: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().split(b"\0")
            arguments = [part.decode("utf-8", errors="replace") for part in raw if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not arguments or Path(arguments[0]).name not in {"bash", "sh"}:
            continue
        resolved = set()
        for argument in arguments[1:]:
            try:
                resolved.add(Path(argument).resolve().as_posix())
            except (OSError, ValueError):
                continue
        if targets & resolved:
            matches.append({"pid": int(entry.name), "command": arguments})
    return sorted(matches, key=lambda row: int(row["pid"]))


def validate_completion_result(
    *,
    completion_audit_path: Path,
    release_receipt_path: Path,
    cofitok_artifact: Path,
    dense_artifact: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    audit = read_json_object(completion_audit_path, name="capacity-full completion audit")
    if (
        audit.get("schema_version") != 1
        or audit.get("profile") != PROFILE
        or audit.get("status") != "pass"
        or audit.get("complete") is not True
        or audit.get("failed_checks") != []
        or audit.get("missing_checks") != []
        or audit.get("expectations") != dict(expected)
        or audit.get("claim_boundary")
        != {
            "training_authorization_is_quality_promotion_gate": False,
            "final_gate_required_for_release": True,
            "release_receipt_requires_complete_audit": True,
        }
    ):
        raise ValueError("capacity-full completion audit did not pass exactly")
    checks = audit.get("checks")
    matches = [
        row
        for row in checks
        if isinstance(row, Mapping) and row.get("name") == RELEASE_CHECK
    ] if isinstance(checks, list) else []
    if len(matches) != 1 or matches[0].get("status") != "pass":
        raise ValueError("capacity-full completion lacks release-authorized artifacts")
    verified = {
        "cofitok": verify_generation_release_receipt(
            release_receipt_path,
            cofitok_artifact,
        ),
        "dense_identity": verify_generation_release_receipt(
            release_receipt_path,
            dense_artifact,
        ),
    }
    if any(row.get("completion_profile") != PROFILE for row in verified.values()):
        raise ValueError("capacity-full release receipt uses another completion profile")
    return {
        "completion_audit": file_identity(completion_audit_path),
        "release_receipt": file_identity(release_receipt_path),
        "artifacts": {
            "cofitok": file_identity(cofitok_artifact),
            "dense_identity": file_identity(dense_artifact),
        },
        "verification": verified,
    }


def finalization_action(posteval_result: Mapping[str, Any]) -> str:
    status = posteval_result.get("status")
    if status == "pass":
        return "finalize"
    if status == "hold":
        return "hold"
    raise ValueError("capacity-full posteval result is not terminal")


def _status(
    *,
    status: str,
    detail: str,
    expected: Mapping[str, Any],
    posteval: Mapping[str, Any] | None = None,
    completion: Mapping[str, Any] | None = None,
    gpu_rows: list[str] | None = None,
    idle_polls: int = 0,
    attempt: int = 0,
    runbook_processes: list[dict[str, Any]] | None = None,
    child_pid: int | None = None,
    child_process_group_id: int | None = None,
    child_exit_code: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    completed = completion is not None
    return {
        "schema_version": SUPERVISOR_SCHEMA_VERSION,
        "role": SUPERVISOR_ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected": dict(expected),
        "posteval": dict(posteval) if posteval is not None else None,
        "completion": dict(completion) if completion is not None else None,
        "gpu_compute_rows": list(gpu_rows or []),
        "idle_polls": idle_polls,
        "required_idle_polls": int(expected["required_idle_polls"]),
        "attempt": attempt,
        "runbook_processes": list(runbook_processes or []),
        "child_pid": child_pid,
        "child_process_group_id": child_process_group_id,
        "child_exit_code": child_exit_code,
        "finalization_launch_performed": attempt > 0,
        "inference_export_performed": completed,
        "completion_audit_passed": completed,
        "release_receipt_built": completed,
        "formal_generation_completion_claimed": completed and status == "pass",
        "authorization_boundary": dict(SUPERVISOR_BOUNDARY),
        "error": error,
        "updated_at": _utc_now(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the capacity-full final quality gate, then supervise only "
            "release-authorized inference export, completion audit, and receipt."
        )
    )
    parser.add_argument("--finalization-project", type=Path, required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--full-output-root", type=Path, required=True)
    parser.add_argument("--posteval-supervisor-status", type=Path, required=True)
    parser.add_argument("--posteval-supervisor-deployment", type=Path, required=True)
    parser.add_argument("--expected-posteval-supervisor-deployment-sha256", required=True)
    parser.add_argument("--training-supervisor-deployment", type=Path, required=True)
    parser.add_argument("--expected-training-supervisor-deployment-sha256", required=True)
    parser.add_argument("--training-launch-receipt", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--cofitok-artifact", type=Path, required=True)
    parser.add_argument("--dense-artifact", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--child-log", type=Path, required=True)
    parser.add_argument("--expected-evaluation-revision", required=True)
    parser.add_argument("--expected-evaluation-tree", required=True)
    parser.add_argument("--expected-evaluation-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
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
        parser.error("capacity-full finalization supervisor timing is invalid")
    return args


def main() -> int:
    args = _parse_args()
    finalization = args.finalization_project.resolve()
    training_project = args.training_project.resolve()
    formal_project = args.formal_project.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    full_root = args.full_output_root.resolve()
    runbook = args.runbook.resolve()
    if full_root != checkpoint_root / "stability_capacity_full_300k_v1":
        raise ValueError("capacity-full finalization output root differs")
    if runbook != finalization / "artifacts/runbooks" / RUNBOOK_NAME:
        raise ValueError("capacity-full finalization runbook is not checkout bound")
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    evaluation_git = _require_git(
        finalization,
        revision=args.expected_evaluation_revision,
        tree=args.expected_evaluation_tree,
        branch=args.expected_evaluation_branch,
        label="evaluation",
    )
    training_git = _require_git(
        training_project,
        revision=args.expected_training_revision,
        tree=args.expected_training_tree,
        branch=args.expected_training_branch,
        label="training",
    )
    training_deployment = read_json_object(
        args.training_supervisor_deployment,
        name="capacity-full training supervisor deployment",
    )
    training_deployment_evidence = validate_training_supervisor_deployment(
        training_deployment,
        receipt_path=args.training_supervisor_deployment,
        expected_sha256=args.expected_training_supervisor_deployment_sha256,
        training_project=training_project,
        formal_project=formal_project,
        full_output_root=full_root,
        expected_training_git=training_git,
    )
    posteval_deployment = read_json_object(
        args.posteval_supervisor_deployment,
        name="capacity-full posteval supervisor deployment",
    )
    posteval_deployment_evidence = validate_posteval_supervisor_deployment(
        posteval_deployment,
        receipt_path=args.posteval_supervisor_deployment,
        expected_sha256=args.expected_posteval_supervisor_deployment_sha256,
        evaluation_project=finalization,
        training_project=training_project,
        full_output_root=full_root,
        expected_evaluation_git=evaluation_git,
        expected_training_git=training_git,
        training_supervisor_deployment_identity=file_identity(
            args.training_supervisor_deployment
        ),
    )
    expected = {
        "evaluation_git": evaluation_git,
        "training_git": training_git,
        "full_output_root": full_root.as_posix(),
        "training_supervisor_deployment": training_deployment_evidence,
        "posteval_supervisor_deployment": posteval_deployment_evidence,
        "required_idle_polls": args.required_idle_polls,
        "max_attempts": args.max_attempts,
    }
    deadline = time.monotonic() + args.timeout_seconds
    idle_polls = 0
    attempts = 0
    posteval_evidence: dict[str, Any] | None = None
    completion_evidence: dict[str, Any] | None = None

    def publish(**values: Any) -> None:
        write_json_report(
            args.status_output,
            _status(
                expected=expected,
                posteval=posteval_evidence,
                completion=completion_evidence,
                idle_polls=idle_polls,
                attempt=attempts,
                **values,
            ),
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(finalization / "src"), str(training_project / "src"))
    )
    auxiliary_runbooks = [
        runbook,
        finalization / "artifacts/runbooks/generation_capacity_full_300k_export_inference_artifacts.sh",
        finalization / "artifacts/runbooks/generation_stability_ema_teacher_export_inference_artifacts.sh",
        finalization / "artifacts/runbooks/generation_capacity_full_300k_completion_audit.sh",
    ]
    while True:
        if time.monotonic() >= deadline:
            publish(status="failed", detail="capacity_full_finalization_supervisor_timeout")
            return 1
        if not args.posteval_supervisor_status.is_file():
            idle_polls = 0
            publish(status="waiting", detail="waiting_for_posteval_supervisor_status")
            time.sleep(args.poll_seconds)
            continue
        posteval_status = read_json_object(
            args.posteval_supervisor_status,
            name="capacity-full posteval supervisor status",
        )
        posteval_state = posteval_status.get("status")
        if posteval_state in {"failed", "not_selected"}:
            publish(
                status="not_selected" if posteval_state == "not_selected" else "failed",
                detail="capacity_full_posteval_did_not_complete",
            )
            return 0 if posteval_state == "not_selected" else 1
        if posteval_state not in {"pass", "hold"}:
            idle_polls = 0
            publish(status="waiting", detail="waiting_for_capacity_full_final_gate")
            time.sleep(args.poll_seconds)
            continue
        _require_git(
            finalization,
            revision=args.expected_evaluation_revision,
            tree=args.expected_evaluation_tree,
            branch=args.expected_evaluation_branch,
            label="evaluation",
        )
        result = validate_posteval_result(
            final_gate_path=args.final_gate,
            comparison_path=args.comparison,
            expected_training_revision=args.expected_training_revision,
            expected_training_branch=args.expected_training_branch,
            expected_evaluation_revision=args.expected_evaluation_revision,
            expected_evaluation_branch=args.expected_evaluation_branch,
        )
        status_evidence = validate_posteval_supervisor_status(
            posteval_status,
            status_path=args.posteval_supervisor_status,
            expected_evaluation_git=evaluation_git,
            expected_training_git=training_git,
            full_output_root=full_root,
            expected_result=result,
        )
        posteval_evidence = {"result": result, "status": status_evidence}
        if finalization_action(result) == "hold":
            publish(status="hold", detail="capacity_full_final_gate_held_no_export")
            return 0
        receipt_identity = file_identity(args.training_launch_receipt)
        authorization = capture_generation_training_authorization(
            args.training_launch_receipt
        )
        if (
            authorization.get("stage") != TRAINING_AUTHORIZATION_STAGE
            or authorization.get("decision") != TRAINING_AUTHORIZATION_DECISION
            or authorization.get("gate_sha256") != receipt_identity["sha256"]
        ):
            raise ValueError("capacity-full training authorization replay differs")
        final_gate_identity = file_identity(args.final_gate)
        completion_expectations = {
            "training_revision": args.expected_training_revision,
            "training_tree": args.expected_training_tree,
            "training_branch": args.expected_training_branch,
            "evaluation_revision": args.expected_evaluation_revision,
            "evaluation_tree": args.expected_evaluation_tree,
            "evaluation_branch": args.expected_evaluation_branch,
            "export_revision": args.expected_evaluation_revision,
            "export_branch": args.expected_evaluation_branch,
            "training_supervisor_deployment_sha256": (
                args.expected_training_supervisor_deployment_sha256
            ),
            "posteval_supervisor_deployment_sha256": (
                args.expected_posteval_supervisor_deployment_sha256
            ),
            "training_launch_receipt_sha256": receipt_identity["sha256"],
            "final_gate_sha256": final_gate_identity["sha256"],
        }
        if args.completion_audit.is_file() and args.release_receipt.is_file():
            completion_evidence = validate_completion_result(
                completion_audit_path=args.completion_audit,
                release_receipt_path=args.release_receipt,
                cofitok_artifact=args.cofitok_artifact,
                dense_artifact=args.dense_artifact,
                expected=completion_expectations,
            )
            publish(status="pass", detail="capacity_full_generation_completed")
            return 0
        processes = _runbook_processes(auxiliary_runbooks)
        if processes:
            idle_polls = 0
            publish(
                status="observing",
                detail="observing_existing_capacity_full_finalization",
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
            publish(status="failed", detail="capacity_full_finalization_exhausted_retry_budget")
            return 1
        attempts += 1
        child_environment = dict(environment)
        child_environment.update(
            {
                "FINALIZATION_PROJECT": finalization.as_posix(),
                "TRAINING_PROJECT": training_project.as_posix(),
                "FORMAL_PROJECT": formal_project.as_posix(),
                "PYTHON": sys.executable,
                "CHECKPOINT_ROOT": checkpoint_root.as_posix(),
                "SOURCE_OUTPUT_ROOT": args.posteval_supervisor_status.resolve().parents[
                    1
                ].as_posix(),
                "FULL_OUTPUT_ROOT": full_root.as_posix(),
                "EXPECTED_TRAINING_SUPERVISOR_DEPLOYMENT_SHA256": (
                    args.expected_training_supervisor_deployment_sha256
                ),
                "EXPECTED_POSTEVAL_SUPERVISOR_DEPLOYMENT_SHA256": (
                    args.expected_posteval_supervisor_deployment_sha256
                ),
                "EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256": receipt_identity[
                    "sha256"
                ],
                "EXPECTED_FINAL_GATE_SHA256": final_gate_identity["sha256"],
                "EXPECTED_TRAINING_REVISION": args.expected_training_revision,
                "EXPECTED_TRAINING_TREE": args.expected_training_tree,
                "EXPECTED_TRAINING_BRANCH": args.expected_training_branch,
                "EXPECTED_EVALUATION_REVISION": args.expected_evaluation_revision,
                "EXPECTED_EVALUATION_TREE": args.expected_evaluation_tree,
                "EXPECTED_EVALUATION_BRANCH": args.expected_evaluation_branch,
            }
        )
        args.child_log.parent.mkdir(parents=True, exist_ok=True)
        with args.child_log.open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                ["bash", str(runbook)],
                cwd=finalization,
                env=child_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            group_id = child.pid
            while child.poll() is None:
                publish(
                    status="running",
                    detail="capacity_full_release_finalization_running",
                    child_pid=child.pid,
                    child_process_group_id=group_id,
                    gpu_rows=_gpu_rows(),
                )
                time.sleep(min(args.poll_seconds, 60.0))
        exit_code = int(child.returncode)
        if exit_code == 0:
            completion_evidence = validate_completion_result(
                completion_audit_path=args.completion_audit,
                release_receipt_path=args.release_receipt,
                cofitok_artifact=args.cofitok_artifact,
                dense_artifact=args.dense_artifact,
                expected=completion_expectations,
            )
            publish(
                status="pass",
                detail="capacity_full_generation_completed",
                child_pid=child.pid,
                child_process_group_id=group_id,
                child_exit_code=0,
            )
            return 0
        idle_polls = 0
        if exit_code not in RETRYABLE_EXIT_CODES:
            publish(
                status="failed",
                detail="capacity_full_finalization_nonretryable_failure",
                child_pid=child.pid,
                child_process_group_id=group_id,
                child_exit_code=exit_code,
            )
            return 1
        publish(
            status="waiting",
            detail="capacity_full_finalization_retryable_exit",
            child_pid=child.pid,
            child_process_group_id=group_id,
            child_exit_code=exit_code,
        )
        time.sleep(args.retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
