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

from cofitok.generation.frozen_supplemental import (
    FROZEN_SUPPLEMENTAL_ROLE,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.run_generation_stability_full_readiness_waiter import (
        validate_posteval_status,
    )
    from scripts.verify_generation_stability_frozen_supplemental import (
        verify_frozen_supplemental_report,
    )
except ModuleNotFoundError:
    from run_generation_stability_full_readiness_waiter import (
        validate_posteval_status,
    )
    from verify_generation_stability_frozen_supplemental import (
        verify_frozen_supplemental_report,
    )


ROLE = "generation_stability_frozen_supplemental_waiter"
READINESS_ROLE = "generation_stability_full_readiness_waiter"
TRANSIENT_RUNBOOK_EXIT_CODES = {9, 75}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


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


def verify_supplemental_checkout(
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
        raise ValueError("frozen supplemental checkout identity mismatch")
    return actual


def _age_seconds(timestamp: Any, *, now: datetime | None = None) -> float:
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("coordination status timestamp is missing")
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("coordination status timestamp has no timezone")
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - parsed.astimezone(timezone.utc)).total_seconds())


def observe_posteval_status(
    report: dict[str, Any],
    *,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_evaluation_revision: str,
    expected_evaluation_branch: str,
    silence_seconds: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    observation = validate_posteval_status(
        report,
        expected_training_revision=expected_training_revision,
        expected_training_branch=expected_training_branch,
        expected_evaluation_revision=expected_evaluation_revision,
        expected_evaluation_branch=expected_evaluation_branch,
    )
    if not observation["complete"]:
        age = _age_seconds(observation.get("updated_at"), now=now)
        if age > silence_seconds:
            raise RuntimeError("stability 50K post-evaluation status is stale")
        observation["age_seconds"] = age
    return observation


def observe_readiness_status(
    report: dict[str, Any],
    *,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_evaluation_revision: str,
    expected_evaluation_branch: str,
    expected_readiness_revision: str,
    expected_readiness_branch: str,
    silence_seconds: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    expected = report.get("expected")
    required = {
        "training_revision": expected_training_revision,
        "training_branch": expected_training_branch,
        "evaluation_revision": expected_evaluation_revision,
        "evaluation_branch": expected_evaluation_branch,
        "readiness_revision": expected_readiness_revision,
        "readiness_branch": expected_readiness_branch,
        "full_training_launch_allowed": False,
    }
    status = str(report.get("status", ""))
    if (
        report.get("schema_version") != 1
        or report.get("role") != READINESS_ROLE
        or status not in {"waiting", "running", "pass", "failed"}
        or not isinstance(expected, dict)
        or any(expected.get(name) != value for name, value in required.items())
        or report.get("full_training_launch_allowed") is not False
    ):
        raise ValueError("stability-full readiness coordination contract mismatch")
    terminal = status in {"pass", "failed"}
    observation = {
        "status": status,
        "detail": report.get("detail"),
        "terminal": terminal,
        "child_pid": report.get("child_pid"),
        "child_exit_code": report.get("child_exit_code"),
        "updated_at": report.get("updated_at"),
        "full_training_launch_allowed": False,
    }
    if not terminal:
        age = _age_seconds(observation["updated_at"], now=now)
        if age > silence_seconds:
            raise RuntimeError("stability-full readiness coordination status is stale")
        observation["age_seconds"] = age
    return observation


def validate_supplemental_result(
    report: dict[str, Any],
    *,
    report_path: Path,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_frozen_evaluation_revision: str,
    expected_frozen_evaluation_branch: str,
    expected_supplemental_revision: str,
    expected_supplemental_branch: str,
) -> dict[str, Any]:
    status = str(report.get("status", ""))
    checks = report.get("checks")
    expected_provenance = {
        "training_revision": expected_training_revision,
        "training_branch": expected_training_branch,
        "evaluation_revision": expected_frozen_evaluation_revision,
        "evaluation_branch": expected_frozen_evaluation_branch,
        "supplemental_revision": expected_supplemental_revision,
        "supplemental_branch": expected_supplemental_branch,
    }
    expected_builder = {
        "revision": expected_supplemental_revision,
        "branch": expected_supplemental_branch,
        "tracked_dirty": False,
    }
    boundary = {
        "supplemental_non_authorizing": True,
        "replaces_generation_gate": False,
        "replaces_readiness": False,
        "scaling_authorization_evaluated": False,
        "full_training_launch_allowed": False,
    }
    expected_checks = {
        "base_gate_passed",
        "distribution_support_passed",
        "ema_rollout_stability_passed",
        "all_supplemental_quality_checks_passed",
    }
    component_checks = (
        "base_gate_passed",
        "distribution_support_passed",
        "ema_rollout_stability_passed",
    )
    checks_coherent = isinstance(checks, dict) and (
        bool(checks.get("all_supplemental_quality_checks_passed"))
        == all(bool(checks.get(name)) for name in component_checks)
    )
    decision = str(report.get("decision", ""))
    if (
        report.get("schema_version") != 1
        or report.get("role") != FROZEN_SUPPLEMENTAL_ROLE
        or status not in {"pass", "hold"}
        or report.get("builder_git") != expected_builder
        or report.get("provenance_contract") != expected_provenance
        or report.get("claim_boundary") != boundary
        or not isinstance(checks, dict)
        or set(checks) != expected_checks
        or not all(isinstance(value, bool) for value in checks.values())
        or not checks_coherent
        or (status == "pass")
        != bool(checks["all_supplemental_quality_checks_passed"])
        or (status == "pass" and decision != "supplemental_quality_complete")
        or (status == "hold" and decision != "hold")
    ):
        raise ValueError("frozen supplemental terminal result contract mismatch")
    source = _source(report_path)
    verification = None
    if status == "pass":
        verification = verify_frozen_supplemental_report(
            report,
            report_path=report_path,
            expected_report_sha256=source["sha256"],
        )
    return {
        "status": status,
        "decision": decision,
        "source": source,
        "checks": checks,
        "pass_replay": verification,
        "supplemental_non_authorizing": True,
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


def _status(
    *,
    status: str,
    detail: str,
    expected: dict[str, Any],
    checkout: dict[str, Any] | None = None,
    posteval: dict[str, Any] | None = None,
    readiness: dict[str, Any] | None = None,
    gpu_pids: list[int] | None = None,
    supplemental: dict[str, Any] | None = None,
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
        "checkout": checkout,
        "posteval": posteval,
        "readiness_coordination": readiness,
        "gpu_pids": list(gpu_pids or []),
        "supplemental": supplemental,
        "supplemental_non_authorizing": True,
        "full_training_launch_allowed": False,
        "updated_at": _utc_now(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for frozen stability post-evaluation and the already queued "
            "readiness GPU stage to finish, then execute only the source-bound "
            "non-authorizing frozen quality supplemental."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--posteval-status", type=Path, required=True)
    parser.add_argument("--readiness-status", type=Path, required=True)
    parser.add_argument("--supplemental-report", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--supplemental-runbook", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-frozen-evaluation-revision", required=True)
    parser.add_argument("--expected-frozen-evaluation-branch", required=True)
    parser.add_argument("--expected-readiness-revision", required=True)
    parser.add_argument("--expected-readiness-branch", required=True)
    parser.add_argument("--expected-supplemental-revision", required=True)
    parser.add_argument("--expected-supplemental-branch", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1_209_600.0)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--status-silence-seconds", type=float, default=900.0)
    return parser.parse_args()


def _expected(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_revision": args.expected_training_revision,
        "training_branch": args.expected_training_branch,
        "frozen_evaluation_revision": args.expected_frozen_evaluation_revision,
        "frozen_evaluation_branch": args.expected_frozen_evaluation_branch,
        "readiness_revision": args.expected_readiness_revision,
        "readiness_branch": args.expected_readiness_branch,
        "supplemental_revision": args.expected_supplemental_revision,
        "supplemental_branch": args.expected_supplemental_branch,
        "supplemental_non_authorizing": True,
        "full_training_launch_allowed": False,
    }


def _run_locked(args: argparse.Namespace) -> int:
    if (
        args.timeout_seconds <= 0.0
        or args.poll_seconds <= 0.0
        or args.status_silence_seconds <= 0.0
    ):
        raise ValueError("supplemental waiter timing values must be positive")
    project = args.project.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    runbook = args.supplemental_runbook.resolve()
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    expected = _expected(args)
    checkout = verify_supplemental_checkout(
        project,
        expected_revision=args.expected_supplemental_revision,
        expected_branch=args.expected_supplemental_branch,
    )
    deadline = time.monotonic() + args.timeout_seconds
    child_pid: int | None = None
    child_exit_code: int | None = None
    while True:
        checkout = verify_supplemental_checkout(
            project,
            expected_revision=args.expected_supplemental_revision,
            expected_branch=args.expected_supplemental_branch,
        )
        posteval = None
        if args.posteval_status.is_file():
            posteval = observe_posteval_status(
                _read_object(args.posteval_status),
                expected_training_revision=args.expected_training_revision,
                expected_training_branch=args.expected_training_branch,
                expected_evaluation_revision=(
                    args.expected_frozen_evaluation_revision
                ),
                expected_evaluation_branch=args.expected_frozen_evaluation_branch,
                silence_seconds=args.status_silence_seconds,
            )
        readiness = None
        if args.readiness_status.is_file():
            readiness = observe_readiness_status(
                _read_object(args.readiness_status),
                expected_training_revision=args.expected_training_revision,
                expected_training_branch=args.expected_training_branch,
                expected_evaluation_revision=(
                    args.expected_frozen_evaluation_revision
                ),
                expected_evaluation_branch=args.expected_frozen_evaluation_branch,
                expected_readiness_revision=args.expected_readiness_revision,
                expected_readiness_branch=args.expected_readiness_branch,
                silence_seconds=args.status_silence_seconds,
            )
        gpu_pids = _gpu_compute_pids()
        ready = (
            posteval is not None
            and posteval["complete"]
            and readiness is not None
            and readiness["terminal"]
            and not gpu_pids
        )
        if ready:
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON": sys.executable,
                    "CHECKPOINT_ROOT": str(checkpoint_root),
                    "EXPECTED_TRAINING_REVISION": args.expected_training_revision,
                    "EXPECTED_TRAINING_BRANCH": args.expected_training_branch,
                    "EXPECTED_FROZEN_EVALUATION_REVISION": (
                        args.expected_frozen_evaluation_revision
                    ),
                    "EXPECTED_FROZEN_EVALUATION_BRANCH": (
                        args.expected_frozen_evaluation_branch
                    ),
                    "EXPECTED_SUPPLEMENTAL_REVISION": (
                        args.expected_supplemental_revision
                    ),
                    "EXPECTED_SUPPLEMENTAL_BRANCH": (
                        args.expected_supplemental_branch
                    ),
                }
            )
            child = subprocess.Popen(
                ["bash", str(runbook)],
                cwd=project,
                env=environment,
            )
            child_pid = child.pid
            write_json_report(
                args.status_output,
                _status(
                    status="running",
                    detail="frozen_supplemental_running",
                    expected=expected,
                    checkout=checkout,
                    posteval=posteval,
                    readiness=readiness,
                    child_pid=child_pid,
                ),
            )
            child_exit_code = child.wait()
            if child_exit_code in TRANSIENT_RUNBOOK_EXIT_CODES:
                detail = "supplemental_runbook_deferred_by_concurrent_gpu_or_lock"
            elif child_exit_code != 0:
                write_json_report(
                    args.status_output,
                    _status(
                        status="failed",
                        detail="frozen_supplemental_failed",
                        expected=expected,
                        checkout=checkout,
                        posteval=posteval,
                        readiness=readiness,
                        child_pid=child_pid,
                        child_exit_code=child_exit_code,
                    ),
                )
                return child_exit_code
            else:
                if not args.supplemental_report.is_file():
                    raise RuntimeError(
                        "successful supplemental runbook did not produce its report"
                    )
                supplemental = validate_supplemental_result(
                    _read_object(args.supplemental_report),
                    report_path=args.supplemental_report,
                    expected_training_revision=args.expected_training_revision,
                    expected_training_branch=args.expected_training_branch,
                    expected_frozen_evaluation_revision=(
                        args.expected_frozen_evaluation_revision
                    ),
                    expected_frozen_evaluation_branch=(
                        args.expected_frozen_evaluation_branch
                    ),
                    expected_supplemental_revision=(
                        args.expected_supplemental_revision
                    ),
                    expected_supplemental_branch=args.expected_supplemental_branch,
                )
                write_json_report(
                    args.status_output,
                    _status(
                        status="pass",
                        detail="frozen_supplemental_execution_completed",
                        expected=expected,
                        checkout=checkout,
                        posteval=posteval,
                        readiness=readiness,
                        supplemental=supplemental,
                        child_pid=child_pid,
                        child_exit_code=0,
                    ),
                )
                return 0
        elif posteval is None or not posteval["complete"]:
            detail = "waiting_for_frozen_postevaluation"
        elif readiness is None or not readiness["terminal"]:
            detail = "waiting_for_readiness_gpu_stage_to_finish"
        else:
            detail = "waiting_for_idle_gpu"
        write_json_report(
            args.status_output,
            _status(
                status="waiting",
                detail=detail,
                expected=expected,
                checkout=checkout,
                posteval=posteval,
                readiness=readiness,
                gpu_pids=gpu_pids,
                child_pid=child_pid,
                child_exit_code=child_exit_code,
            ),
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting to execute frozen supplemental")
        time.sleep(args.poll_seconds)


def run_waiter(args: argparse.Namespace) -> int:
    expected = _expected(args)
    try:
        with exclusive_output_lock(args.status_output, role=ROLE):
            try:
                return _run_locked(args)
            except Exception as error:
                write_json_report(
                    args.status_output,
                    _status(
                        status="failed",
                        detail=f"{type(error).__name__}: {error}",
                        expected=expected,
                    ),
                )
                print(f"{type(error).__name__}: {error}", file=sys.stderr)
                return 1
    except OutputLockError as error:
        print(str(error), file=sys.stderr)
        return 75


def main() -> int:
    return run_waiter(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
