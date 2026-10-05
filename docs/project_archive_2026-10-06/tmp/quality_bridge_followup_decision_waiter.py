#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


ROLE = "cofitok_quality_bridge_followup_decision_waiter"
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        check=check,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "expected_revision": args.expected_revision,
        "expected_tree": args.expected_tree,
        "expected_branch": args.expected_branch,
        "poll_seconds": args.poll_seconds,
        "scope": {
            "decision_build_allowed": True,
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "release_allowed": False,
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
        raise RuntimeError(f"project is missing: {args.project}")
    revision = _run(["git", "rev-parse", "HEAD"], cwd=args.project).stdout.strip()
    tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=args.project).stdout.strip()
    branch = _run(
        ["git", "branch", "--show-current"], cwd=args.project
    ).stdout.strip()
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=args.project,
    ).stdout.strip()
    observed = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": bool(dirty),
    }
    expected = (args.expected_revision, args.expected_tree, args.expected_branch)
    if (revision, tree, branch) != expected or dirty:
        raise RuntimeError(
            "decision checkout identity mismatch: "
            + json.dumps(observed, sort_keys=True)
        )
    runbook = (
        args.project
        / "artifacts/runbooks/generation_quality_bridge_followup_decision_after_result.sh"
    )
    if not runbook.is_file():
        raise RuntimeError(f"decision runbook is missing: {runbook}")
    if not args.python.is_file():
        raise RuntimeError(f"runtime Python is missing: {args.python}")
    return {**observed, "runbook_sha256": _sha256(runbook)}


def main() -> int:
    args = _parse_args()
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
        git_identity = _validate_checkout(args)
    except Exception as exc:
        _write_status(
            args,
            base,
            status="failed",
            detail="decision_checkout_validation_failed",
            error=str(exc),
            waiter_sha256=observed_self_sha256,
        )
        return 82

    base.update(
        {
            "waiter_sha256": observed_self_sha256,
            "git": git_identity,
        }
    )
    result = args.quality_bridge_root / "reports/quality_bridge_result.json"
    decision = (
        args.quality_bridge_root / "reports/followup_experiment_decision.json"
    )
    runbook = (
        args.project
        / "artifacts/runbooks/generation_quality_bridge_followup_decision_after_result.sh"
    )

    polls = 0
    while not result.is_file():
        polls += 1
        _write_status(
            args,
            base,
            status="waiting",
            detail="waiting_for_quality_bridge_result",
            polls=polls,
            result_exists=False,
            decision_exists=decision.is_file(),
        )
        if args.max_polls and polls >= args.max_polls:
            _write_status(
                args,
                base,
                status="stopped",
                detail="bounded_wait_completed_without_result",
                polls=polls,
                result_exists=False,
                decision_exists=decision.is_file(),
            )
            return 78
        time.sleep(args.poll_seconds)

    result_sha256 = _sha256(result)
    existing_decision_sha256 = _sha256(decision) if decision.is_file() else ""
    _write_status(
        args,
        base,
        status="running",
        detail="replaying_sources_and_building_followup_decision",
        polls=polls,
        result_exists=True,
        result_sha256=result_sha256,
        decision_exists=decision.is_file(),
        existing_decision_sha256=existing_decision_sha256 or None,
    )

    env = os.environ.copy()
    env.update(
        {
            "PROJECT": str(args.project),
            "PYTHON": str(args.python),
            "QUALITY_BRIDGE_ROOT": str(args.quality_bridge_root),
            "EXPECTED_DECISION_REVISION": args.expected_revision,
            "EXPECTED_DECISION_BRANCH": args.expected_branch,
            "EXPECTED_DECISION_SHA256": existing_decision_sha256,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{_utc_now()}] result={result} sha256={result_sha256}\n"
        )
        handle.flush()
        completed = subprocess.run(
            ["bash", str(runbook)],
            cwd=str(args.project),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(f"[{_utc_now()}] exit_code={completed.returncode}\n")

    if completed.returncode != 0:
        _write_status(
            args,
            base,
            status="failed",
            detail="followup_decision_replay_failed",
            polls=polls,
            result_sha256=result_sha256,
            decision_exists=decision.is_file(),
            exit_code=completed.returncode,
            log=str(args.log),
        )
        return completed.returncode

    if not decision.is_file():
        _write_status(
            args,
            base,
            status="failed",
            detail="decision_missing_after_successful_runbook",
            polls=polls,
            result_sha256=result_sha256,
            exit_code=0,
            log=str(args.log),
        )
        return 83

    decision_sha256 = _sha256(decision)
    try:
        decision_payload = json.loads(decision.read_text(encoding="utf-8"))
        recommendation = decision_payload["recommended_next_stage"]
        boundary = decision_payload["authorization_boundary"]
        if any(
            boundary.get(key) is not False
            for key in (
                "recommended_stage_execution_allowed",
                "quality_bridge_execution_allowed",
                "full_training_launch_allowed",
                "full_300k_launch_allowed",
                "report_is_promotion_gate",
                "release_authorization_allowed",
            )
        ):
            raise ValueError("decision authorization boundary is not fail-closed")
    except Exception as exc:
        _write_status(
            args,
            base,
            status="failed",
            detail="decision_postcondition_failed",
            polls=polls,
            result_sha256=result_sha256,
            decision_sha256=decision_sha256,
            error=str(exc),
            log=str(args.log),
        )
        return 84

    _write_status(
        args,
        base,
        status="completed",
        detail="followup_experiment_decision_verified",
        polls=polls,
        result_sha256=result_sha256,
        decision_path=str(decision),
        decision_sha256=decision_sha256,
        recommended_next_stage={
            "id": recommendation.get("id"),
            "category": recommendation.get("category"),
            "execution_ready": recommendation.get("execution_ready"),
        },
        log=str(args.log),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
