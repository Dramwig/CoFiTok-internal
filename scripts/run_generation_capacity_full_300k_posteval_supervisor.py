from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation_gate import validate_generation_gate_authorization
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report
from cofitok.training.authorization import capture_generation_training_authorization

try:
    from scripts.build_large_scale_generation_comparison import (
        verify_comparison_source_reports,
    )
    from scripts.validate_generation_training_pair import validate_training_pair
except ModuleNotFoundError:
    from build_large_scale_generation_comparison import (
        verify_comparison_source_reports,
    )
    from validate_generation_training_pair import validate_training_pair


SUPERVISOR_SCHEMA_VERSION = 1
SUPERVISOR_ROLE = "capacity_full_300k_posteval_supervisor"
RUNBOOK_NAME = "generation_capacity_full_300k_posteval_50k.sh"
BASE_RUNBOOK_NAME = "generation_stability_ema_teacher_full_posteval_50k.sh"
TRAINING_SUPERVISOR_ROLE = "capacity_full_300k_training_supervisor"
TRAINING_DEPLOYMENT_ROLE = "capacity_full_300k_training_supervisor_deployment"
TRAINING_AUTHORIZATION_STAGE = "capacity_full_experimental"
TRAINING_AUTHORIZATION_DECISION = "authorize_fresh_matched_300k_training"
FINAL_GATE_PROFILE = "capacity_full"
RETRYABLE_EXIT_CODES = frozenset({9, 75, 87, 88, 89, 137, 143})
TRAINING_SUPERVISOR_BOUNDARY = {
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
SUPERVISOR_BOUNDARY = {
    "training_launch_allowed": False,
    "posteval_50k_allowed_after_exact_training_pass": True,
    "final_quality_gate_build_allowed": True,
    "inference_export_allowed": False,
    "release_receipt_allowed": False,
    "formal_generation_completion_claim_allowed": False,
    "process_signaling_allowed": False,
    "unrelated_gpu_process_modification_allowed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(project: Path) -> dict[str, Any]:
    def run(*arguments: str, binary: bool = False) -> str | bytes:
        result = subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
        )
        return result.stdout

    return {
        "revision": str(run("rev-parse", "HEAD")).strip(),
        "tree": str(run("rev-parse", "HEAD^{tree}")).strip(),
        "branch": str(run("branch", "--show-current")).strip(),
        "tracked_dirty": bool(
            str(run("status", "--porcelain", "--untracked-files=no")).strip()
        ),
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
        raise ValueError(f"capacity-full posteval {label} checkout differs")
    return actual


def _formal_snapshot(project: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "-C", str(project), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "path": project.resolve().as_posix(),
        "head": head,
        "porcelain_count": len(porcelain.splitlines()),
        "porcelain_sha256": hashlib.sha256(porcelain).hexdigest(),
    }


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


def validate_training_supervisor_deployment(
    report: Mapping[str, Any],
    *,
    receipt_path: Path,
    expected_sha256: str,
    training_project: Path,
    formal_project: Path,
    full_output_root: Path,
    expected_training_git: Mapping[str, Any],
) -> dict[str, Any]:
    identity = file_identity(receipt_path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("capacity-full training supervisor deployment SHA256 differs")
    checkout = report.get("checkout")
    if (
        report.get("schema_version") != 1
        or report.get("role") != TRAINING_DEPLOYMENT_ROLE
        or report.get("status") != "active"
        or report.get("git") != expected_training_git
        or not isinstance(checkout, Mapping)
        or checkout.get("git") != expected_training_git
        or Path(str(checkout.get("path", ""))).resolve() != training_project.resolve()
        or Path(str(report.get("full_output_root", ""))).resolve()
        != full_output_root.resolve()
        or report.get("authorization_boundary") != TRAINING_SUPERVISOR_BOUNDARY
        or report.get("formal_checkout_snapshot") != _formal_snapshot(formal_project)
    ):
        raise ValueError("capacity-full training supervisor deployment contract differs")
    sources = report.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("capacity-full training supervisor deployment sources are missing")
    for name, source in sources.items():
        if not isinstance(source, Mapping) or file_identity(source["path"]) != dict(source):
            raise ValueError(
                f"capacity-full training supervisor deployment source changed: {name}"
            )
    return {"identity": identity, "git": dict(expected_training_git)}


def validate_training_completion(
    status: Mapping[str, Any],
    *,
    status_path: Path,
    training_receipt_path: Path,
    cofitok_training_path: Path,
    dense_training_path: Path,
    pair_monitor_path: Path,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_training_tree: str,
    full_output_root: Path,
) -> dict[str, Any]:
    expected_git = {
        "revision": expected_training_revision,
        "tree": expected_training_tree,
        "branch": expected_training_branch,
        "tracked_dirty": False,
    }
    expected = status.get("expected")
    receipt_in_status = status.get("training_launch_receipt")
    if (
        status.get("schema_version") != 1
        or status.get("role") != TRAINING_SUPERVISOR_ROLE
        or status.get("status") != "pass"
        or status.get("detail")
        != "capacity_full_fresh_matched_300k_training_completed"
        or status.get("training_launch_performed") is not True
        or status.get("formal_generation_completion_claimed") is not False
        or status.get("child_exit_code") != 0
        or status.get("authorization_boundary") != TRAINING_SUPERVISOR_BOUNDARY
        or not isinstance(expected, Mapping)
        or expected.get("training_git") != expected_git
        or Path(str(expected.get("full_output_root", ""))).resolve()
        != full_output_root.resolve()
        or not isinstance(receipt_in_status, Mapping)
    ):
        raise ValueError("capacity-full training supervisor pass state differs")
    receipt_identity = file_identity(training_receipt_path)
    if receipt_identity != dict(receipt_in_status):
        raise ValueError("capacity-full training supervisor bound another receipt")
    authorization = capture_generation_training_authorization(training_receipt_path)
    if (
        authorization.get("stage") != TRAINING_AUTHORIZATION_STAGE
        or authorization.get("decision") != TRAINING_AUTHORIZATION_DECISION
        or authorization.get("gate_sha256") != receipt_identity["sha256"]
    ):
        raise ValueError("capacity-full training authorization replay differs")
    cofitok = read_json_object(cofitok_training_path, name="CoFiTok training report")
    dense = read_json_object(dense_training_path, name="dense training report")
    receipt = read_json_object(training_receipt_path, name="training launch receipt")
    pair = validate_training_pair(
        cofitok,
        dense,
        expected_steps=300_000,
        expected_revision=expected_training_revision,
        expected_branch=expected_training_branch,
        expected_dataset="imagenet_256",
        expected_recipe_stage="stability_full",
        expected_authorization_gate=receipt,
    )
    monitor = read_json_object(pair_monitor_path, name="capacity-full pair monitor")
    monitor_git = monitor.get("git")
    if (
        monitor.get("schema_version") != 2
        or monitor.get("monitor") != "generation_capacity_full_300k_v1"
        or monitor.get("status") != "pass"
        or monitor.get("stage") != "complete"
        or monitor.get("issues") != []
        or not isinstance(monitor_git, Mapping)
        or monitor_git.get("revision") != expected_training_revision
        or monitor_git.get("branch") != expected_training_branch
        or monitor_git.get("tracked_dirty") is not False
    ):
        raise ValueError("capacity-full pair monitor did not pass cleanly")
    return {
        "status": file_identity(status_path),
        "training_launch_receipt": receipt_identity,
        "authorization": authorization,
        "training_pair": {
            "status": pair["status"],
            "expected_steps": pair["expected_steps"],
            "expected_revision": pair["expected_revision"],
            "authorization_gate_identity_sha256": pair[
                "authorization_gate_identity_sha256"
            ],
        },
        "pair_monitor": file_identity(pair_monitor_path),
    }


def validate_posteval_result(
    *,
    final_gate_path: Path,
    comparison_path: Path,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_evaluation_revision: str,
    expected_evaluation_branch: str,
) -> dict[str, Any]:
    gate = read_json_object(final_gate_path, name="capacity-full final gate")
    sources = verify_generation_gate_source_reports(gate)
    expected_provenance = {
        "training_revision": expected_training_revision,
        "training_branch": expected_training_branch,
        "evaluation_revision": expected_evaluation_revision,
        "evaluation_branch": expected_evaluation_branch,
    }
    rows = gate.get("gates")
    if (
        gate.get("stage") != "full"
        or gate.get("source_profile") != FINAL_GATE_PROFILE
        or sources.get("source_profile") != FINAL_GATE_PROFILE
        or gate.get("provenance_contract") != expected_provenance
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, Mapping) for row in rows)
        or len({str(row.get("name", "")) for row in rows}) != len(rows)
        or any(type(row.get("passed")) is not bool for row in rows)
    ):
        raise ValueError("capacity-full final gate contract differs")
    passed = all(bool(row["passed"]) for row in rows)
    expected_gate_state = (
        ("pass", "large_scale_generation_ready") if passed else ("fail", "hold")
    )
    if (gate.get("status"), gate.get("decision")) != expected_gate_state:
        raise ValueError("capacity-full final gate status differs from its checks")
    authorization = (
        validate_generation_gate_authorization(gate, expected_stage="full")
        if passed
        else None
    )
    comparison = read_json_object(
        comparison_path,
        name="capacity-full large-scale comparison",
    )
    comparison_sources = verify_comparison_source_reports(comparison)
    expected_comparison_status = "ready" if passed else "hold"
    if (
        comparison.get("source_profile") != FINAL_GATE_PROFILE
        or comparison_sources.get("source_profile") != FINAL_GATE_PROFILE
        or comparison.get("status") != expected_comparison_status
        or comparison.get("final_gate")
        != {"status": gate["status"], "decision": gate["decision"]}
    ):
        raise ValueError("capacity-full comparison result differs")
    return {
        "status": "pass" if passed else "hold",
        "decision": gate["decision"],
        "final_gate": file_identity(final_gate_path),
        "comparison": file_identity(comparison_path),
        "authorization": authorization,
        "source_report_count": len(sources["source_reports"]),
        "diagnostic_report_count": len(sources.get("diagnostic_reports", {})),
    }


def _status(
    *,
    status: str,
    detail: str,
    expected: Mapping[str, Any],
    training: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
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
        "training": dict(training) if training is not None else None,
        "posteval_result": dict(result) if result is not None else None,
        "gpu_compute_rows": list(gpu_rows or []),
        "idle_polls": idle_polls,
        "required_idle_polls": int(expected["required_idle_polls"]),
        "attempt": attempt,
        "runbook_processes": list(runbook_processes or []),
        "child_pid": child_pid,
        "child_process_group_id": child_process_group_id,
        "child_exit_code": child_exit_code,
        "posteval_launch_performed": attempt > 0,
        "final_quality_gate_passed": (
            result is not None and result.get("status") == "pass"
        ),
        "inference_export_performed": False,
        "release_receipt_built": False,
        "formal_generation_completion_claimed": False,
        "authorization_boundary": dict(SUPERVISOR_BOUNDARY),
        "error": error,
        "updated_at": _utc_now(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for exact capacity-full matched 300K completion, confirm GPU "
            "idleness, and supervise only the formal 50K post-evaluation and final "
            "quality gate."
        )
    )
    parser.add_argument("--evaluation-project", type=Path, required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--full-output-root", type=Path, required=True)
    parser.add_argument("--training-supervisor-status", type=Path, required=True)
    parser.add_argument("--training-supervisor-deployment", type=Path, required=True)
    parser.add_argument("--expected-training-supervisor-deployment-sha256", required=True)
    parser.add_argument("--training-launch-receipt", type=Path, required=True)
    parser.add_argument("--cofitok-training", type=Path, required=True)
    parser.add_argument("--dense-training", type=Path, required=True)
    parser.add_argument("--pair-monitor", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
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
        parser.error("capacity-full posteval supervisor timing is invalid")
    return args


def main() -> int:
    args = _parse_args()
    evaluation = args.evaluation_project.resolve()
    training_project = args.training_project.resolve()
    formal_project = args.formal_project.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    full_root = args.full_output_root.resolve()
    runbook = args.runbook.resolve()
    base_runbook = evaluation / "artifacts/runbooks" / BASE_RUNBOOK_NAME
    if full_root != checkpoint_root / "stability_capacity_full_300k_v1":
        raise ValueError("capacity-full posteval output root differs")
    if runbook != evaluation / "artifacts/runbooks" / RUNBOOK_NAME:
        raise ValueError("capacity-full posteval runbook is not evaluation-checkout bound")
    if not runbook.is_file() or not base_runbook.is_file():
        raise FileNotFoundError("capacity-full posteval runbook is missing")
    evaluation_git = _require_git(
        evaluation,
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
    deployment = read_json_object(
        args.training_supervisor_deployment,
        name="capacity-full training supervisor deployment",
    )
    deployment_evidence = validate_training_supervisor_deployment(
        deployment,
        receipt_path=args.training_supervisor_deployment,
        expected_sha256=args.expected_training_supervisor_deployment_sha256,
        training_project=training_project,
        formal_project=formal_project,
        full_output_root=full_root,
        expected_training_git=training_git,
    )
    expected = {
        "evaluation_git": evaluation_git,
        "training_git": training_git,
        "full_output_root": full_root.as_posix(),
        "training_supervisor_deployment": deployment_evidence,
        "required_idle_polls": args.required_idle_polls,
        "max_attempts": args.max_attempts,
        "posteval_only": True,
    }
    deadline = time.monotonic() + args.timeout_seconds
    idle_polls = 0
    attempts = 0
    training_evidence: dict[str, Any] | None = None
    result_evidence: dict[str, Any] | None = None

    def publish(**values: Any) -> None:
        write_json_report(
            args.status_output,
            _status(
                expected=expected,
                training=training_evidence,
                result=result_evidence,
                idle_polls=idle_polls,
                attempt=attempts,
                **values,
            ),
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(evaluation / "src"), str(training_project / "src"))
    )
    while True:
        if time.monotonic() >= deadline:
            publish(status="failed", detail="capacity_full_posteval_supervisor_timeout")
            return 1
        if not args.training_supervisor_status.is_file():
            idle_polls = 0
            publish(status="waiting", detail="waiting_for_training_supervisor_status")
            time.sleep(args.poll_seconds)
            continue
        training_status = read_json_object(
            args.training_supervisor_status,
            name="capacity-full training supervisor status",
        )
        training_state = training_status.get("status")
        if training_state in {"failed", "not_selected"}:
            publish(
                status="not_selected" if training_state == "not_selected" else "failed",
                detail="capacity_full_training_did_not_complete",
            )
            return 0 if training_state == "not_selected" else 1
        if training_state != "pass":
            idle_polls = 0
            publish(status="waiting", detail="waiting_for_passed_capacity_full_training")
            time.sleep(args.poll_seconds)
            continue
        _require_git(
            evaluation,
            revision=args.expected_evaluation_revision,
            tree=args.expected_evaluation_tree,
            branch=args.expected_evaluation_branch,
            label="evaluation",
        )
        _require_git(
            training_project,
            revision=args.expected_training_revision,
            tree=args.expected_training_tree,
            branch=args.expected_training_branch,
            label="training",
        )
        training_evidence = validate_training_completion(
            training_status,
            status_path=args.training_supervisor_status,
            training_receipt_path=args.training_launch_receipt,
            cofitok_training_path=args.cofitok_training,
            dense_training_path=args.dense_training,
            pair_monitor_path=args.pair_monitor,
            expected_training_revision=args.expected_training_revision,
            expected_training_branch=args.expected_training_branch,
            expected_training_tree=args.expected_training_tree,
            full_output_root=full_root,
        )
        if args.final_gate.is_file() and args.comparison.is_file():
            result_evidence = validate_posteval_result(
                final_gate_path=args.final_gate,
                comparison_path=args.comparison,
                expected_training_revision=args.expected_training_revision,
                expected_training_branch=args.expected_training_branch,
                expected_evaluation_revision=args.expected_evaluation_revision,
                expected_evaluation_branch=args.expected_evaluation_branch,
            )
            if result_evidence["status"] == "pass":
                publish(status="pass", detail="capacity_full_final_quality_gate_passed")
            else:
                publish(status="hold", detail="capacity_full_final_quality_gate_held")
            return 0
        processes = _runbook_processes([runbook, base_runbook])
        if processes:
            idle_polls = 0
            publish(
                status="observing",
                detail="observing_existing_capacity_full_posteval_runbook",
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
            publish(status="failed", detail="capacity_full_posteval_exhausted_retry_budget")
            return 1
        attempts += 1
        child_environment = dict(environment)
        child_environment.update(
            {
                "PYTHON": sys.executable,
                "CHECKPOINT_ROOT": checkpoint_root.as_posix(),
                "FULL_OUTPUT_ROOT": full_root.as_posix(),
                "EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256": training_evidence[
                    "training_launch_receipt"
                ]["sha256"],
                "EXPECTED_TRAINING_REVISION": args.expected_training_revision,
                "EXPECTED_TRAINING_BRANCH": args.expected_training_branch,
                "EXPECTED_TARGET_REVISION": args.expected_evaluation_revision,
                "EXPECTED_TARGET_BRANCH": args.expected_evaluation_branch,
            }
        )
        args.child_log.parent.mkdir(parents=True, exist_ok=True)
        with args.child_log.open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                ["bash", str(runbook)],
                cwd=evaluation,
                env=child_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            group_id = child.pid
            while child.poll() is None:
                publish(
                    status="running",
                    detail="capacity_full_formal_50k_posteval_running",
                    child_pid=child.pid,
                    child_process_group_id=group_id,
                    gpu_rows=_gpu_rows(),
                )
                time.sleep(min(args.poll_seconds, 60.0))
        exit_code = int(child.returncode)
        if exit_code == 0:
            result_evidence = validate_posteval_result(
                final_gate_path=args.final_gate,
                comparison_path=args.comparison,
                expected_training_revision=args.expected_training_revision,
                expected_training_branch=args.expected_training_branch,
                expected_evaluation_revision=args.expected_evaluation_revision,
                expected_evaluation_branch=args.expected_evaluation_branch,
            )
            if result_evidence["status"] == "pass":
                publish(
                    status="pass",
                    detail="capacity_full_final_quality_gate_passed",
                    child_pid=child.pid,
                    child_process_group_id=group_id,
                    child_exit_code=0,
                )
            else:
                publish(
                    status="hold",
                    detail="capacity_full_final_quality_gate_held",
                    child_pid=child.pid,
                    child_process_group_id=group_id,
                    child_exit_code=0,
                )
            return 0
        idle_polls = 0
        if exit_code not in RETRYABLE_EXIT_CODES:
            publish(
                status="failed",
                detail="capacity_full_posteval_nonretryable_failure",
                child_pid=child.pid,
                child_process_group_id=group_id,
                child_exit_code=exit_code,
            )
            return 1
        publish(
            status="waiting",
            detail="capacity_full_posteval_retryable_exit",
            child_pid=child.pid,
            child_process_group_id=group_id,
            child_exit_code=exit_code,
        )
        time.sleep(args.retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
