from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROLE = "generation_quality_bridge_exposure_aware_followup_waiter"
SCHEMA_VERSION = 1
RUNBOOK_NAME = "generation_quality_bridge_followup_decision_after_result.sh"
DECISION_NAME = "followup_experiment_decision_exposure_aware_v2.json"
EXPOSURE_RELATIVE_PATH = Path(
    "reports/training_exposure_terminal_100k/training_exposure_report.json"
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact terminal quality and exposure evidence, then build "
            "a CPU-only exposure-aware follow-up decision."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-bridge-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-self-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=0)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    if args.max_polls < 0:
        parser.error("--max-polls must be nonnegative")
    return args


def _base_status(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "project": str(args.project),
        "quality_bridge_root": str(args.quality_bridge_root),
        "poll_seconds": args.poll_seconds,
        "scope": {
            "source_observation_allowed": True,
            "decision_build_allowed": True,
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
            "evaluation_launch_allowed": False,
            "promotion_allowed": False,
            "release_allowed": False,
            "full_300k_launch_allowed": False,
            "process_signals_allowed": False,
        },
    }


def _write_status(
    args: argparse.Namespace,
    base: dict[str, Any],
    *,
    status: str,
    detail: str,
    **extra: Any,
) -> None:
    _atomic_json(
        args.status,
        {
            **base,
            "status": status,
            "detail": detail,
            "updated_at": _utc_now(),
            **extra,
        },
    )


def _validate_checkout(args: argparse.Namespace) -> dict[str, Any]:
    if not args.project.is_dir():
        raise ValueError(f"project is missing: {args.project}")
    observed = {
        "revision": _run(["git", "rev-parse", "HEAD"], cwd=args.project).stdout.strip(),
        "tree": _run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=args.project
        ).stdout.strip(),
        "branch": _run(
            ["git", "branch", "--show-current"], cwd=args.project
        ).stdout.strip(),
        "tracked_dirty": bool(
            _run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=args.project,
            ).stdout.strip()
        ),
    }
    expected = {
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if observed != expected:
        raise ValueError(
            "exposure-aware decision checkout identity differs: "
            + json.dumps(observed, sort_keys=True)
        )
    runbook = args.project / "artifacts" / "runbooks" / RUNBOOK_NAME
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    return {**observed, "runbook_sha256": _sha256(runbook)}


def _postcondition(decision: Path) -> dict[str, Any]:
    payload = json.loads(decision.read_text(encoding="utf-8"))
    boundary = payload.get("authorization_boundary")
    recommendation = payload.get("recommended_next_stage")
    exposure = payload.get("training_exposure")
    if (
        int(payload.get("schema_version", -1)) != 2
        or payload.get("status") != "completed"
        or not isinstance(boundary, dict)
        or not isinstance(recommendation, dict)
        or not isinstance(exposure, dict)
        or exposure.get("terminal_result_content_bound") is not True
        or exposure.get("terminal_checkpoint_binding_verified") is not True
        or any(
            boundary.get(key) is not False
            for key in (
                "recommended_stage_execution_allowed",
                "quality_bridge_execution_allowed",
                "full_training_launch_allowed",
                "full_300k_launch_allowed",
                "report_is_promotion_gate",
                "release_authorization_allowed",
            )
        )
    ):
        raise ValueError("exposure-aware decision postcondition differs")
    return {
        "id": recommendation.get("id"),
        "category": recommendation.get("category"),
        "execution_ready": recommendation.get("execution_ready"),
        "full_data_equivalent_epochs": exposure.get("full_data_equivalent_epochs"),
        "insufficient_exposure_is_live_hypothesis": exposure.get(
            "insufficient_exposure_is_live_hypothesis"
        ),
    }


def main() -> int:
    args = parse_args()
    base = _base_status(args)
    script = Path(__file__).resolve()
    observed_self_sha256 = _sha256(script)
    if observed_self_sha256 != args.expected_self_sha256:
        _write_status(
            args,
            base,
            status="failed",
            detail="waiter_self_sha256_mismatch",
            expected_self_sha256=args.expected_self_sha256,
            observed_self_sha256=observed_self_sha256,
        )
        return 81
    try:
        checkout = _validate_checkout(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        _write_status(
            args,
            base,
            status="failed",
            detail="decision_checkout_validation_failed",
            error=str(error),
            waiter_sha256=observed_self_sha256,
        )
        return 82
    base.update({"waiter_sha256": observed_self_sha256, "git": checkout})

    reports = args.quality_bridge_root / "reports"
    result = reports / "quality_bridge_result.json"
    exposure = args.quality_bridge_root / EXPOSURE_RELATIVE_PATH
    decision = reports / DECISION_NAME
    runbook = args.project / "artifacts" / "runbooks" / RUNBOOK_NAME
    polls = 0
    while not result.is_file() or not exposure.is_file():
        polls += 1
        if not result.is_file():
            detail = "waiting_for_quality_bridge_result"
        else:
            detail = "waiting_for_terminal_training_exposure_report"
        _write_status(
            args,
            base,
            status="waiting",
            detail=detail,
            polls=polls,
            result_exists=result.is_file(),
            exposure_exists=exposure.is_file(),
            decision_exists=decision.is_file(),
        )
        if args.max_polls and polls >= args.max_polls:
            _write_status(
                args,
                base,
                status="stopped",
                detail="bounded_wait_completed_without_terminal_sources",
                polls=polls,
                result_exists=result.is_file(),
                exposure_exists=exposure.is_file(),
                decision_exists=decision.is_file(),
            )
            return 78
        time.sleep(args.poll_seconds)

    result_sha256 = _sha256(result)
    exposure_sha256 = _sha256(exposure)
    existing_decision_sha256 = _sha256(decision) if decision.is_file() else ""
    _write_status(
        args,
        base,
        status="running",
        detail="replaying_terminal_quality_and_exposure_sources",
        polls=polls,
        result_sha256=result_sha256,
        exposure_sha256=exposure_sha256,
        existing_decision_sha256=existing_decision_sha256 or None,
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT": str(args.project),
            "PYTHON": str(args.python),
            "QUALITY_BRIDGE_ROOT": str(args.quality_bridge_root),
            "EXPECTED_DECISION_REVISION": args.expected_revision,
            "EXPECTED_DECISION_BRANCH": args.expected_branch,
            "EXPECTED_DECISION_SHA256": existing_decision_sha256,
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{_utc_now()}] result_sha256={result_sha256} "
            f"exposure_sha256={exposure_sha256}\n"
        )
        handle.flush()
        completed = subprocess.run(
            ["bash", str(runbook)],
            cwd=str(args.project),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"[{_utc_now()}] exit_code={completed.returncode}\n")

    if completed.returncode != 0:
        _write_status(
            args,
            base,
            status="failed",
            detail="exposure_aware_followup_replay_failed",
            polls=polls,
            result_sha256=result_sha256,
            exposure_sha256=exposure_sha256,
            exit_code=completed.returncode,
            log=str(args.log),
        )
        return completed.returncode
    if not decision.is_file():
        _write_status(
            args,
            base,
            status="failed",
            detail="exposure_aware_decision_missing_after_successful_runbook",
            polls=polls,
            exit_code=0,
            log=str(args.log),
        )
        return 83
    try:
        recommendation = _postcondition(decision)
    except (OSError, ValueError, KeyError, TypeError) as error:
        _write_status(
            args,
            base,
            status="failed",
            detail="exposure_aware_decision_postcondition_failed",
            polls=polls,
            decision_sha256=_sha256(decision),
            error=str(error),
            log=str(args.log),
        )
        return 84
    _write_status(
        args,
        base,
        status="completed",
        detail="exposure_aware_followup_decision_verified",
        polls=polls,
        result_sha256=result_sha256,
        exposure_sha256=exposure_sha256,
        decision_path=str(decision),
        decision_sha256=_sha256(decision),
        recommended_next_stage=recommendation,
        log=str(args.log),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
