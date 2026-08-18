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
from typing import Any

from cofitok.generation_gate import validate_generation_gate_authorization
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.process_monitoring import wait_for_child_with_heartbeat
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.build_generation_large_capacity_deployment_receipt import (
        verify_deployment_receipt,
    )
except ModuleNotFoundError:
    from build_generation_large_capacity_deployment_receipt import (
        verify_deployment_receipt,
    )


ROLE = "generation_stability_full_readiness_waiter"
POSTEVAL_ROLE = "generation_stability_50k_posteval_waiter"
SOURCE_PROFILE = "stability_scaling"
FULL_ROOT_ID = "stability_full_300k_ema_teacher"
COFITOK_RUN_ID = "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN_ID = "dense_rollout_x0_u2_ema_teacher"
COFITOK_CONFIG = (
    "configs/generation/"
    "imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
)
DENSE_CONFIG = (
    "configs/generation/"
    "imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json"
)


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


def verify_readiness_checkout(
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
        raise ValueError("stability-full readiness checkout identity mismatch")
    return actual


def validate_posteval_status(
    report: dict[str, Any],
    *,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_evaluation_revision: str,
    expected_evaluation_branch: str,
) -> dict[str, Any]:
    if report.get("schema_version") != 1 or report.get("role") != POSTEVAL_ROLE:
        raise ValueError("stability 50K post-evaluation status contract mismatch")
    status = str(report.get("status", ""))
    if status == "failed":
        raise RuntimeError("stability 50K post-evaluation failed")
    expected = report.get("expected")
    required = {
        "training_revision": expected_training_revision,
        "training_branch": expected_training_branch,
        "evaluation_revision": expected_evaluation_revision,
        "evaluation_branch": expected_evaluation_branch,
        "formal_300k_allowed": False,
    }
    if not isinstance(expected, dict) or any(
        expected.get(name) != value for name, value in required.items()
    ):
        raise ValueError("stability 50K post-evaluation identity mismatch")
    complete = (
        status == "pass"
        and report.get("detail") == "formal_ema_postevaluation_completed"
        and int(report.get("child_exit_code", -1)) == 0
    )
    return {
        "status": status,
        "detail": report.get("detail"),
        "complete": complete,
        "updated_at": report.get("updated_at"),
    }


def validate_scaling_gate(
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
        expected_stage="scaling",
    )
    sources = verify_generation_gate_source_reports(gate)
    if sources.get("source_profile") != SOURCE_PROFILE:
        raise ValueError("stability scaling gate source profile mismatch")
    expected_provenance = {
        "training_revision": expected_training_revision,
        "training_branch": expected_training_branch,
        "evaluation_revision": expected_evaluation_revision,
        "evaluation_branch": expected_evaluation_branch,
    }
    if gate.get("provenance_contract") != expected_provenance:
        raise ValueError("stability scaling gate provenance contract mismatch")
    return {
        "status": "verified",
        "sha256": file_sha256(gate_path),
        "authorization": authorization,
        "provenance_contract": expected_provenance,
        "source_profile": SOURCE_PROFILE,
        "source_report_count": len(sources["source_reports"]),
    }


def validate_deployment(
    *,
    project: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    receipt = verify_deployment_receipt(
        _read_object(receipt_path),
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
        require_current_formal_repository=False,
    )
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if (
        Path(receipt["checkout"]["path"]).resolve() != project.resolve()
        or receipt["checkout"]["git"] != expected_git
        or receipt.get("readiness_execution_allowed") is not True
        or receipt.get("readiness_executed") is not False
        or receipt.get("full_training_launch_allowed") is not False
    ):
        raise ValueError("large-capacity deployment readiness boundary mismatch")
    return {
        "receipt_sha256": expected_receipt_sha256,
        "checkout": receipt["checkout"],
        "readiness_execution_allowed": True,
        "full_training_launch_allowed": False,
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
    return [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _full_paths(project: Path, checkpoint_root: Path) -> dict[str, Path]:
    full_root = checkpoint_root / FULL_ROOT_ID
    reports = full_root / "reports"
    return {
        "readiness": reports / "full_training_readiness.json",
        "config_validation": reports / "config_validation.json",
        "storage_capacity": reports / "storage_capacity.json",
        "runtime_selection": reports / "runtime_selection.json",
        "benchmark_root": full_root / "runtime_preflight/training",
        "cofitok_run": full_root / COFITOK_RUN_ID,
        "dense_run": full_root / DENSE_RUN_ID,
        "cofitok_config": project / COFITOK_CONFIG,
        "dense_config": project / DENSE_CONFIG,
    }


def validate_existing_readiness(
    *,
    project: Path,
    checkpoint_root: Path,
    promotion_gate: Path,
    deployment_receipt: Path,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    paths = _full_paths(project, checkpoint_root)
    readiness = paths["readiness"]
    readiness_sha256 = file_sha256(readiness)
    command = [
        sys.executable,
        str(project / "scripts/validate_generation_full_readiness.py"),
        "--project-root",
        str(project),
        "--readiness",
        str(readiness),
        "--expected-readiness-sha256",
        readiness_sha256,
        "--deployment-receipt",
        str(deployment_receipt),
        "--promotion-gate",
        str(promotion_gate),
        "--cofitok-config",
        str(paths["cofitok_config"]),
        "--dense-config",
        str(paths["dense_config"]),
        "--config-validation",
        str(paths["config_validation"]),
        "--storage-capacity",
        str(paths["storage_capacity"]),
        "--runtime-selection",
        str(paths["runtime_selection"]),
        "--training-run-dir",
        str(paths["cofitok_run"]),
        "--training-run-dir",
        str(paths["dense_run"]),
        "--benchmark-root",
        str(paths["benchmark_root"]),
        "--storage-path",
        str(checkpoint_root),
        "--expected-revision",
        expected_revision,
        "--expected-branch",
        expected_branch,
        "--allow-later-formal-repository",
        "--require-current-runtime-environment",
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
            "existing stability-full readiness failed replay"
            + (f": {diagnostic[-4000:]}" if diagnostic else "")
        )
    return {
        "status": "verified",
        "path": readiness.resolve().as_posix(),
        "sha256": readiness_sha256,
    }


def _status(
    *,
    status: str,
    detail: str,
    expected: dict[str, Any],
    posteval: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    deployment: dict[str, Any] | None = None,
    gpu_pids: list[int] | None = None,
    readiness: dict[str, Any] | None = None,
    child_pid: int | None = None,
    child_exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "child_pid": child_pid,
        "child_exit_code": child_exit_code,
        "expected": expected,
        "posteval": posteval,
        "gate": gate,
        "deployment": deployment,
        "gpu_pids": list(gpu_pids or []),
        "readiness": readiness,
        "full_training_launch_allowed": False,
        "updated_at": _utc_now(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a passing source-bound stability 50K gate and an idle GPU, "
            "then execute only the isolated 250M readiness qualification."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--posteval-status", type=Path, required=True)
    parser.add_argument("--deployment-receipt", type=Path, required=True)
    parser.add_argument("--expected-deployment-receipt-sha256", required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--readiness-runbook", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-evaluation-revision", required=True)
    parser.add_argument("--expected-evaluation-branch", required=True)
    parser.add_argument("--expected-readiness-revision", required=True)
    parser.add_argument("--expected-readiness-branch", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1_209_600.0)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.timeout_seconds <= 0.0 or args.poll_seconds <= 0.0:
        raise ValueError("readiness waiter timing values must be positive")
    project = args.project.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    runbook = args.readiness_runbook.resolve()
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    expected = {
        "training_revision": args.expected_training_revision,
        "training_branch": args.expected_training_branch,
        "evaluation_revision": args.expected_evaluation_revision,
        "evaluation_branch": args.expected_evaluation_branch,
        "readiness_revision": args.expected_readiness_revision,
        "readiness_branch": args.expected_readiness_branch,
        "deployment_receipt_sha256": args.expected_deployment_receipt_sha256,
        "source_profile": SOURCE_PROFILE,
        "readiness_execution_allowed": True,
        "full_training_launch_allowed": False,
    }
    verify_readiness_checkout(
        project,
        expected_revision=args.expected_readiness_revision,
        expected_branch=args.expected_readiness_branch,
    )
    deployment = validate_deployment(
        project=project,
        receipt_path=args.deployment_receipt,
        expected_receipt_sha256=args.expected_deployment_receipt_sha256,
        expected_revision=args.expected_readiness_revision,
        expected_branch=args.expected_readiness_branch,
    )
    paths = _full_paths(project, checkpoint_root)
    if paths["readiness"].is_file():
        readiness = validate_existing_readiness(
            project=project,
            checkpoint_root=checkpoint_root,
            promotion_gate=args.promotion_gate,
            deployment_receipt=args.deployment_receipt,
            expected_revision=args.expected_readiness_revision,
            expected_branch=args.expected_readiness_branch,
        )
        write_json_report(
            args.status_output,
            _status(
                status="pass",
                detail="existing_full_readiness_replayed",
                expected=expected,
                deployment=deployment,
                readiness=readiness,
            ),
        )
        return 0

    deadline = time.monotonic() + args.timeout_seconds
    posteval: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    while True:
        if args.posteval_status.is_file():
            posteval = validate_posteval_status(
                _read_object(args.posteval_status),
                expected_training_revision=args.expected_training_revision,
                expected_training_branch=args.expected_training_branch,
                expected_evaluation_revision=args.expected_evaluation_revision,
                expected_evaluation_branch=args.expected_evaluation_branch,
            )
        if posteval is not None and posteval["complete"]:
            if not args.promotion_gate.is_file():
                raise RuntimeError(
                    "completed stability 50K post-evaluation did not produce a gate"
                )
            gate = validate_scaling_gate(
                _read_object(args.promotion_gate),
                gate_path=args.promotion_gate,
                expected_training_revision=args.expected_training_revision,
                expected_training_branch=args.expected_training_branch,
                expected_evaluation_revision=args.expected_evaluation_revision,
                expected_evaluation_branch=args.expected_evaluation_branch,
            )
            gpu_pids = _gpu_compute_pids()
            if not gpu_pids:
                break
            detail = "passing_gate_waiting_for_idle_gpu"
        else:
            gpu_pids = _gpu_compute_pids()
            detail = "waiting_for_passing_stability_gate"
        write_json_report(
            args.status_output,
            _status(
                status="waiting",
                detail=detail,
                expected=expected,
                posteval=posteval,
                gate=gate,
                deployment=deployment,
                gpu_pids=gpu_pids,
            ),
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for stability-full readiness authorization")
        time.sleep(args.poll_seconds)

    verify_readiness_checkout(
        project,
        expected_revision=args.expected_readiness_revision,
        expected_branch=args.expected_readiness_branch,
    )
    deployment = validate_deployment(
        project=project,
        receipt_path=args.deployment_receipt,
        expected_receipt_sha256=args.expected_deployment_receipt_sha256,
        expected_revision=args.expected_readiness_revision,
        expected_branch=args.expected_readiness_branch,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": sys.executable,
            "CHECKPOINT_ROOT": str(checkpoint_root),
            "EXPECTED_SCALING_GATE_SHA256": str(gate["sha256"]),
            "EXPECTED_DEPLOYMENT_RECEIPT_SHA256": (
                args.expected_deployment_receipt_sha256
            ),
            "EXPECTED_TARGET_REVISION": args.expected_readiness_revision,
            "EXPECTED_TARGET_BRANCH": args.expected_readiness_branch,
        }
    )
    child = subprocess.Popen(
        ["bash", str(runbook)],
        cwd=project,
        env=environment,
    )
    write_json_report(
        args.status_output,
        _status(
            status="running",
            detail="stability_full_readiness_running",
            expected=expected,
            posteval=posteval,
            gate=gate,
            deployment=deployment,
            child_pid=child.pid,
        ),
    )
    exit_code = wait_for_child_with_heartbeat(
        child,
        poll_seconds=args.poll_seconds,
        heartbeat=lambda: write_json_report(
            args.status_output,
            _status(
                status="running",
                detail="stability_full_readiness_running",
                expected=expected,
                posteval=posteval,
                gate=gate,
                deployment=deployment,
                child_pid=child.pid,
            ),
        ),
    )
    if exit_code != 0:
        write_json_report(
            args.status_output,
            _status(
                status="failed",
                detail="stability_full_readiness_failed",
                expected=expected,
                posteval=posteval,
                gate=gate,
                deployment=deployment,
                child_pid=child.pid,
                child_exit_code=exit_code,
            ),
        )
        return exit_code
    readiness = validate_existing_readiness(
        project=project,
        checkpoint_root=checkpoint_root,
        promotion_gate=args.promotion_gate,
        deployment_receipt=args.deployment_receipt,
        expected_revision=args.expected_readiness_revision,
        expected_branch=args.expected_readiness_branch,
    )
    write_json_report(
        args.status_output,
        _status(
            status="pass",
            detail="stability_full_readiness_completed",
            expected=expected,
            posteval=posteval,
            gate=gate,
            deployment=deployment,
            readiness=readiness,
            child_pid=child.pid,
            child_exit_code=0,
        ),
    )
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
                        "training_revision": parsed.expected_training_revision,
                        "training_branch": parsed.expected_training_branch,
                        "evaluation_revision": parsed.expected_evaluation_revision,
                        "evaluation_branch": parsed.expected_evaluation_branch,
                        "readiness_revision": parsed.expected_readiness_revision,
                        "readiness_branch": parsed.expected_readiness_branch,
                        "deployment_receipt_sha256": (
                            parsed.expected_deployment_receipt_sha256
                        ),
                        "source_profile": SOURCE_PROFILE,
                        "readiness_execution_allowed": True,
                        "full_training_launch_allowed": False,
                    },
                ),
            )
        finally:
            raise
