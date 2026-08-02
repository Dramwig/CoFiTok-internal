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
from typing import Any, Callable

from cofitok.generation_gate import validate_generation_gate_authorization
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.verify_generation_stability_frozen_supplemental import (
        verify_frozen_supplemental_report,
    )
except ModuleNotFoundError:
    from verify_generation_stability_frozen_supplemental import (
        verify_frozen_supplemental_report,
    )


ROLE = "generation_stability_posttraining_supervisor"
FULL_MONITOR_NAME = "generation_stability_ema_teacher_full_matched_300k"
FULL_LAUNCH_ROLE = "stability_full_training_launch_receipt"
FINAL_GATE_PROFILE = "stability_full"
COMPLETION_PROFILE = "stability_generation_system_v1"
STAGE_REPLAY_ERROR_EXIT_CODE = 86
FULL_LAUNCH_SOURCE_NAMES = {
    "deployment_receipt",
    "promotion_gate",
    "stability_supplemental",
    "full_readiness",
    "readiness_bridge",
    "cofitok_config",
    "dense_config",
    "config_validation",
    "storage_capacity",
    "runtime_selection",
    "launch_storage_capacity",
}
FROZEN_SUPPLEMENTAL_CHECK_NAMES = {
    "base_gate_passed",
    "distribution_support_passed",
    "ema_rollout_stability_passed",
    "all_supplemental_quality_checks_passed",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _git_identity(project: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def verify_supervisor_checkout(
    project: Path,
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    actual = _git_identity(project)
    if actual != expected:
        raise ValueError("stability post-training supervisor checkout identity mismatch")
    return actual


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def validate_full_launch_receipt(
    report: dict[str, Any],
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_readiness_sha256: str,
    expected_scaling_gate_sha256: str,
    expected_training_revision: str,
    expected_training_branch: str,
    checkpoint_root: Path,
) -> dict[str, Any]:
    if file_sha256(receipt_path) != expected_receipt_sha256:
        raise ValueError("stability full launch receipt SHA256 differs")
    expected_git = {
        "revision": expected_training_revision,
        "branch": expected_training_branch,
        "tracked_dirty": False,
    }
    if (
        report.get("schema_version") != 3
        or report.get("status") != "pass"
        or report.get("role") != FULL_LAUNCH_ROLE
        or report.get("stage") != "stability_full"
        or report.get("git") != expected_git
        or report.get("readiness_sha256") != expected_readiness_sha256
        or report.get("full_training_launch_authorized") is not True
        or report.get("formal_generation_completion_claimed") is not False
    ):
        raise ValueError("stability full launch receipt contract mismatch")
    source_reports = report.get("source_reports")
    if (
        not isinstance(source_reports, dict)
        or set(source_reports) != FULL_LAUNCH_SOURCE_NAMES
    ):
        raise ValueError("stability full launch receipt sources are incomplete")
    for name, expected in source_reports.items():
        if not isinstance(expected, dict) or "path" not in expected:
            raise ValueError(f"stability full launch source is malformed: {name}")
        if _source_identity(Path(expected["path"])) != expected:
            raise ValueError(f"stability full launch source changed: {name}")
    if source_reports.get("full_readiness", {}).get("sha256") != expected_readiness_sha256:
        raise ValueError("stability full launch readiness binding differs")
    if source_reports.get("promotion_gate", {}).get("sha256") != expected_scaling_gate_sha256:
        raise ValueError("stability full launch promotion-gate binding differs")
    quality = report.get("quality_prerequisites")
    supplemental = (
        quality.get("frozen_stability_supplemental")
        if isinstance(quality, dict)
        else None
    )
    if (
        not isinstance(supplemental, dict)
        or supplemental.get("report")
        != source_reports.get("stability_supplemental")
        or supplemental.get("supplemental_non_authorizing") is not True
        or supplemental.get("required_for_full_training_launch") is not True
        or supplemental.get("full_training_launch_allowed") is not False
        or not isinstance(supplemental.get("checks"), dict)
        or set(supplemental["checks"]) != FROZEN_SUPPLEMENTAL_CHECK_NAMES
        or not all(value is True for value in supplemental["checks"].values())
    ):
        raise ValueError("stability full launch quality prerequisite differs")
    verified_supplemental = verify_frozen_supplemental_report(
        _read_object(Path(source_reports["stability_supplemental"]["path"])),
        report_path=Path(source_reports["stability_supplemental"]["path"]),
        expected_report_sha256=source_reports["stability_supplemental"][
            "sha256"
        ],
        promotion_gate_path=Path(source_reports["promotion_gate"]["path"]),
    )
    if verified_supplemental != supplemental:
        raise ValueError("stability full launch supplemental replay differs")
    full_root = checkpoint_root.resolve() / "stability_full_300k_ema_teacher"
    expected_runs = [
        (full_root / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher").as_posix(),
        (full_root / "dense_rollout_x0_u2_ema_teacher").as_posix(),
    ]
    if report.get("training_run_dirs") != expected_runs:
        raise ValueError("stability full launch run paths differ")
    return {
        "status": "verified",
        "sha256": expected_receipt_sha256,
        "git": expected_git,
        "source_count": len(source_reports),
        "stability_supplemental_sha256": source_reports[
            "stability_supplemental"
        ]["sha256"],
        "training_run_dirs": expected_runs,
    }


def validate_full_monitor(
    report: dict[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if report.get("schema_version") != 2 or report.get("monitor") != FULL_MONITOR_NAME:
        raise ValueError("stability full monitor contract mismatch")
    if report.get("git") != expected_git:
        raise ValueError("stability full monitor Git identity mismatch")
    status = str(report.get("status", ""))
    stage = str(report.get("stage", ""))
    issues = report.get("issues")
    if status in {"failed", "stalled"}:
        raise RuntimeError(f"stability full monitor reached {status}")
    if not isinstance(issues, list):
        raise ValueError("stability full monitor issues are malformed")
    complete = status == "pass" and stage == "complete" and not issues
    if status == "pass" and not complete:
        raise ValueError("stability full monitor pass state is malformed")
    runs = report.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("stability full monitor runs are missing")
    if complete:
        for method in ("cofitok", "dense_identity"):
            run = runs.get(method)
            if (
                not isinstance(run, dict)
                or run.get("complete") is not True
                or int(run.get("expected_steps", -1)) != 300_000
                or int(run.get("last_step", -1)) != 300_000
                or run.get("health_issues") != []
            ):
                raise ValueError(f"stability full monitor run is incomplete: {method}")
    return {
        "status": status,
        "stage": stage,
        "issues": issues,
        "complete": complete,
        "updated_at": report.get("updated_at"),
    }


def validate_gate(
    gate: dict[str, Any],
    *,
    gate_path: Path,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_evaluation_revision: str,
    expected_evaluation_branch: str,
) -> dict[str, Any]:
    authorization = validate_generation_gate_authorization(
        gate,
        expected_stage="full",
    )
    sources = verify_generation_gate_source_reports(gate)
    if sources.get("source_profile") != FINAL_GATE_PROFILE:
        raise ValueError("stability full gate source profile mismatch")
    expected_provenance = {
        "training_revision": expected_training_revision,
        "training_branch": expected_training_branch,
        "evaluation_revision": expected_evaluation_revision,
        "evaluation_branch": expected_evaluation_branch,
    }
    if gate.get("provenance_contract") != expected_provenance:
        raise ValueError("stability full gate provenance contract mismatch")
    return {
        "status": "verified",
        "sha256": file_sha256(gate_path),
        "authorization": authorization,
        "source_profile": FINAL_GATE_PROFILE,
        "provenance_contract": expected_provenance,
        "source_report_count": len(sources["source_reports"]),
    }


def validate_completion(
    report: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    if (
        report.get("profile") != COMPLETION_PROFILE
        or report.get("status") != "pass"
        or report.get("complete") is not True
        or report.get("failed_checks") != []
        or report.get("missing_checks") != []
    ):
        raise ValueError("stability generation completion audit did not pass")
    observed = report.get("expectations")
    if not isinstance(observed, dict) or any(
        observed.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("stability generation completion expectations differ")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("stability generation completion checks are missing")
    return {
        "status": "verified",
        "check_count": len(checks),
    }


def _gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _status(
    *,
    status: str,
    detail: str,
    expected: dict[str, Any],
    launch_receipt: dict[str, Any] | None = None,
    monitor: dict[str, Any] | None = None,
    final_gate: dict[str, Any] | None = None,
    completion: dict[str, Any] | None = None,
    gpu_pids: list[int] | None = None,
    stage: str | None = None,
    attempt: int | None = None,
    child_pid: int | None = None,
    child_exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "stage": stage,
        "attempt": attempt,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "child_pid": child_pid,
        "child_exit_code": child_exit_code,
        "expected": expected,
        "launch_receipt": launch_receipt,
        "monitor": monitor,
        "final_gate": final_gate,
        "completion": completion,
        "gpu_pids": list(gpu_pids or []),
        "training_launch_owned": False,
        "formal_generation_completion_claimed": status == "pass",
        "updated_at": _utc_now(),
    }


def _run_stage(
    *,
    name: str,
    runbook: Path,
    project: Path,
    environment: dict[str, str],
    max_attempts: int,
    retry_seconds: float,
    write_status: Callable[..., None],
    nonretryable_report: Path | None = None,
) -> None:
    for attempt in range(1, max_attempts + 1):
        child = subprocess.Popen(
            ["bash", str(runbook)],
            cwd=project,
            env=environment,
        )
        write_status(
            status="running",
            detail=f"{name}_running",
            stage=name,
            attempt=attempt,
            child_pid=child.pid,
        )
        exit_code = child.wait()
        if exit_code == 0:
            return
        if exit_code == STAGE_REPLAY_ERROR_EXIT_CODE:
            write_status(
                status="failed",
                detail=f"{name}_replay_rejected",
                stage=name,
                attempt=attempt,
                child_pid=child.pid,
                child_exit_code=exit_code,
            )
            raise RuntimeError(f"{name} replay validation failed")
        if nonretryable_report is not None and nonretryable_report.is_file():
            report = _read_object(nonretryable_report)
            if report.get("status") in {"failed", "incomplete"}:
                write_status(
                    status="failed",
                    detail=f"{name}_nonretryable_report",
                    stage=name,
                    attempt=attempt,
                    child_pid=child.pid,
                    child_exit_code=exit_code,
                )
                raise RuntimeError(f"{name} produced a nonretryable report")
        if attempt == max_attempts:
            write_status(
                status="failed",
                detail=f"{name}_exhausted_retries",
                stage=name,
                attempt=attempt,
                child_pid=child.pid,
                child_exit_code=exit_code,
            )
            raise RuntimeError(f"{name} failed after {max_attempts} attempts")
        write_status(
            status="retrying",
            detail=f"{name}_retry_scheduled",
            stage=name,
            attempt=attempt,
            child_pid=child.pid,
            child_exit_code=exit_code,
        )
        time.sleep(retry_seconds * attempt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for an externally authorized stability-full 300K pair, then "
            "run formal post-evaluation, release export, and completion audit."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--full-monitor", type=Path, required=True)
    parser.add_argument("--full-launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-full-launch-receipt-sha256", required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--scaling-gate", type=Path, required=True)
    parser.add_argument("--expected-scaling-gate-sha256", required=True)
    parser.add_argument("--expected-full-readiness-sha256", required=True)
    parser.add_argument("--expected-scaling-training-revision", required=True)
    parser.add_argument("--expected-scaling-training-branch", required=True)
    parser.add_argument("--expected-scaling-evaluation-revision", required=True)
    parser.add_argument("--expected-scaling-evaluation-branch", required=True)
    parser.add_argument("--expected-full-training-revision", required=True)
    parser.add_argument("--expected-full-training-branch", required=True)
    parser.add_argument("--expected-evaluation-revision", required=True)
    parser.add_argument("--expected-evaluation-branch", required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--posteval-runbook", type=Path, required=True)
    parser.add_argument("--export-runbook", type=Path, required=True)
    parser.add_argument("--completion-runbook", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5_184_000.0)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--max-stage-attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (
        args.timeout_seconds <= 0
        or args.poll_seconds <= 0
        or args.retry_seconds <= 0
        or args.max_stage_attempts < 1
    ):
        raise ValueError("post-training supervisor timing values are invalid")
    project = args.project.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    supervisor_git = _git_identity(project)
    if (
        args.expected_evaluation_revision != supervisor_git["revision"]
        or args.expected_evaluation_branch != supervisor_git["branch"]
    ):
        raise ValueError("evaluation identity must match the supervisor checkout")
    verify_supervisor_checkout(
        project,
        expected_revision=args.expected_evaluation_revision,
        expected_branch=args.expected_evaluation_branch,
    )
    for runbook in (
        args.posteval_runbook,
        args.export_runbook,
        args.completion_runbook,
    ):
        if not runbook.resolve().is_file():
            raise FileNotFoundError(runbook)
    if file_sha256(args.scaling_gate) != args.expected_scaling_gate_sha256:
        raise ValueError("stability scaling gate SHA256 differs")
    scaling_gate = _read_object(args.scaling_gate)
    validate_generation_gate_authorization(
        scaling_gate,
        expected_stage="scaling",
    )
    scaling_sources = verify_generation_gate_source_reports(scaling_gate)
    if scaling_sources.get("source_profile") != "stability_scaling":
        raise ValueError("stability scaling gate source profile mismatch")
    if scaling_gate.get("provenance_contract") != {
        "training_revision": args.expected_scaling_training_revision,
        "training_branch": args.expected_scaling_training_branch,
        "evaluation_revision": args.expected_scaling_evaluation_revision,
        "evaluation_branch": args.expected_scaling_evaluation_branch,
    }:
        raise ValueError("stability scaling gate provenance contract mismatch")
    launch_receipt = validate_full_launch_receipt(
        _read_object(args.full_launch_receipt),
        receipt_path=args.full_launch_receipt,
        expected_receipt_sha256=args.expected_full_launch_receipt_sha256,
        expected_readiness_sha256=args.expected_full_readiness_sha256,
        expected_scaling_gate_sha256=args.expected_scaling_gate_sha256,
        expected_training_revision=args.expected_full_training_revision,
        expected_training_branch=args.expected_full_training_branch,
        checkpoint_root=checkpoint_root,
    )
    expected = {
        "decision_sha256": args.expected_decision_sha256,
        "scaling_gate_sha256": args.expected_scaling_gate_sha256,
        "full_readiness_sha256": args.expected_full_readiness_sha256,
        "full_launch_receipt_sha256": args.expected_full_launch_receipt_sha256,
        "scaling_training_revision": args.expected_scaling_training_revision,
        "scaling_training_branch": args.expected_scaling_training_branch,
        "scaling_evaluation_revision": args.expected_scaling_evaluation_revision,
        "scaling_evaluation_branch": args.expected_scaling_evaluation_branch,
        "full_training_revision": args.expected_full_training_revision,
        "full_training_branch": args.expected_full_training_branch,
        "evaluation_revision": args.expected_evaluation_revision,
        "evaluation_branch": args.expected_evaluation_branch,
        "training_launch_owned": False,
    }
    full_root = checkpoint_root / "stability_full_300k_ema_teacher"
    final_gate_path = full_root / "reports/final_generation_gate.json"
    completion_path = full_root / "reports/stability_generation_completion_audit.json"
    observed_monitor: dict[str, Any] | None = None
    observed_gate: dict[str, Any] | None = None
    observed_completion: dict[str, Any] | None = None

    def publish(**overrides: Any) -> None:
        write_json_report(
            args.status_output,
            _status(
                expected=expected,
                launch_receipt=launch_receipt,
                monitor=observed_monitor,
                final_gate=observed_gate,
                completion=observed_completion,
                **overrides,
            ),
        )

    deadline = time.monotonic() + args.timeout_seconds
    while True:
        if args.full_monitor.is_file():
            observed_monitor = validate_full_monitor(
                _read_object(args.full_monitor),
                expected_revision=args.expected_full_training_revision,
                expected_branch=args.expected_full_training_branch,
            )
            if observed_monitor["complete"]:
                gpu_pids = _gpu_compute_pids()
                if not gpu_pids:
                    break
                detail = "full_training_complete_waiting_for_idle_gpu"
            else:
                gpu_pids = _gpu_compute_pids()
                detail = "waiting_for_authorized_full_training_completion"
        else:
            gpu_pids = _gpu_compute_pids()
            detail = "waiting_for_full_training_monitor"
        publish(
            status="waiting",
            detail=detail,
            stage="full_training_wait",
            gpu_pids=gpu_pids,
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for stability-full training completion")
        time.sleep(args.poll_seconds)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": sys.executable,
            "CHECKPOINT_ROOT": str(checkpoint_root),
            "EXPECTED_SCALING_GATE_SHA256": args.expected_scaling_gate_sha256,
            "EXPECTED_TRAINING_REVISION": args.expected_full_training_revision,
            "EXPECTED_TRAINING_BRANCH": args.expected_full_training_branch,
            "EXPECTED_TARGET_REVISION": args.expected_evaluation_revision,
            "EXPECTED_TARGET_BRANCH": args.expected_evaluation_branch,
        }
    )
    _run_stage(
        name="full_postevaluation",
        runbook=args.posteval_runbook.resolve(),
        project=project,
        environment=environment,
        max_attempts=args.max_stage_attempts,
        retry_seconds=args.retry_seconds,
        write_status=publish,
    )
    if not final_gate_path.is_file():
        raise RuntimeError("full post-evaluation completed without a final gate")
    observed_gate = validate_gate(
        _read_object(final_gate_path),
        gate_path=final_gate_path,
        expected_training_revision=args.expected_full_training_revision,
        expected_training_branch=args.expected_full_training_branch,
        expected_evaluation_revision=args.expected_evaluation_revision,
        expected_evaluation_branch=args.expected_evaluation_branch,
    )
    final_gate_sha256 = observed_gate["sha256"]
    export_environment = dict(environment)
    export_environment.update(
        {
            "EXPECTED_FINAL_GATE_SHA256": final_gate_sha256,
            "EXPECTED_TARGET_REVISION": args.expected_evaluation_revision,
            "EXPECTED_TARGET_BRANCH": args.expected_evaluation_branch,
        }
    )
    _run_stage(
        name="inference_export",
        runbook=args.export_runbook.resolve(),
        project=project,
        environment=export_environment,
        max_attempts=args.max_stage_attempts,
        retry_seconds=args.retry_seconds,
        write_status=publish,
    )
    audit_environment = dict(export_environment)
    audit_environment.update(
        {
            "EXPECTED_DECISION_SHA256": args.expected_decision_sha256,
            "EXPECTED_SCALING_TRAINING_REVISION": (
                args.expected_scaling_training_revision
            ),
            "EXPECTED_SCALING_TRAINING_BRANCH": args.expected_scaling_training_branch,
            "EXPECTED_SCALING_EVALUATION_REVISION": (
                args.expected_scaling_evaluation_revision
            ),
            "EXPECTED_SCALING_EVALUATION_BRANCH": (
                args.expected_scaling_evaluation_branch
            ),
            "EXPECTED_FULL_READINESS_SHA256": args.expected_full_readiness_sha256,
            "EXPECTED_FULL_LAUNCH_RECEIPT_SHA256": (
                args.expected_full_launch_receipt_sha256
            ),
            "EXPECTED_FULL_TRAINING_REVISION": args.expected_full_training_revision,
            "EXPECTED_FULL_TRAINING_BRANCH": args.expected_full_training_branch,
            "EXPECTED_FULL_EVALUATION_REVISION": args.expected_evaluation_revision,
            "EXPECTED_FULL_EVALUATION_BRANCH": args.expected_evaluation_branch,
            "EXPECTED_AUDIT_REVISION": args.expected_evaluation_revision,
            "EXPECTED_AUDIT_BRANCH": args.expected_evaluation_branch,
        }
    )
    _run_stage(
        name="completion_audit",
        runbook=args.completion_runbook.resolve(),
        project=project,
        environment=audit_environment,
        max_attempts=args.max_stage_attempts,
        retry_seconds=args.retry_seconds,
        write_status=publish,
        nonretryable_report=completion_path,
    )
    observed_completion = validate_completion(
        _read_object(completion_path),
        expected={
            "decision_sha256": args.expected_decision_sha256,
            "scaling_gate_sha256": args.expected_scaling_gate_sha256,
            "full_readiness_sha256": args.expected_full_readiness_sha256,
            "full_launch_receipt_sha256": args.expected_full_launch_receipt_sha256,
            "final_gate_sha256": final_gate_sha256,
            "scaling_training_revision": args.expected_scaling_training_revision,
            "scaling_training_branch": args.expected_scaling_training_branch,
            "scaling_evaluation_revision": args.expected_scaling_evaluation_revision,
            "scaling_evaluation_branch": args.expected_scaling_evaluation_branch,
            "full_training_revision": args.expected_full_training_revision,
            "full_training_branch": args.expected_full_training_branch,
            "full_evaluation_revision": args.expected_evaluation_revision,
            "full_evaluation_branch": args.expected_evaluation_branch,
            "export_revision": args.expected_evaluation_revision,
            "export_branch": args.expected_evaluation_branch,
        },
    )
    publish(status="pass", detail="stability_generation_system_completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        try:
            parsed = _parse_args()
            write_json_report(
                parsed.status_output,
                _status(
                    status="failed",
                    detail=f"{type(error).__name__}: {error}",
                    expected={
                        "full_launch_receipt_sha256": (
                            parsed.expected_full_launch_receipt_sha256
                        ),
                        "full_training_revision": parsed.expected_full_training_revision,
                        "full_training_branch": parsed.expected_full_training_branch,
                        "evaluation_revision": parsed.expected_evaluation_revision,
                        "evaluation_branch": parsed.expected_evaluation_branch,
                        "training_launch_owned": False,
                    },
                ),
            )
        finally:
            raise
