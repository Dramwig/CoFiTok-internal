from __future__ import annotations

import argparse
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.reporting import write_json_report


SUPERVISOR_NAME = "generation_completion_supervisor"


def build_status(
    *,
    status: str,
    detail: str,
    attempt: int,
    max_attempts: int,
    pipeline_stage: str,
    pipeline_status: str,
    child_exit_code: int | None,
    next_retry_seconds: int | None,
    git_commit: str,
    hostname: str,
    updated_at: str,
) -> dict[str, Any]:
    if status not in {"running", "retrying", "pass", "failed"}:
        raise ValueError("unsupported supervisor status")
    if attempt < 1 or max_attempts < attempt:
        raise ValueError("supervisor attempt counters are invalid")
    if status == "retrying" and (next_retry_seconds is None or next_retry_seconds < 1):
        raise ValueError("retrying supervisor status requires a positive delay")
    if status != "retrying" and next_retry_seconds is not None:
        raise ValueError("next retry delay is valid only for retrying status")
    if status in {"retrying", "failed"} and child_exit_code is None:
        raise ValueError("retrying/failed supervisor status requires child exit code")
    return {
        "schema_version": 1,
        "supervisor": SUPERVISOR_NAME,
        "status": status,
        "detail": detail,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "pipeline_stage": pipeline_stage,
        "pipeline_status": pipeline_status,
        "child_exit_code": child_exit_code,
        "next_retry_seconds": next_retry_seconds,
        "git_commit": git_commit,
        "hostname": hostname,
        "updated_at": updated_at,
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically update the generation completion supervisor status."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    parser.add_argument("--pipeline-stage", default="unknown")
    parser.add_argument("--pipeline-status", default="unknown")
    parser.add_argument("--child-exit-code", type=int)
    parser.add_argument("--next-retry-seconds", type=int)
    args = parser.parse_args()
    report = build_status(
        status=args.status,
        detail=args.detail,
        attempt=args.attempt,
        max_attempts=args.max_attempts,
        pipeline_stage=args.pipeline_stage,
        pipeline_status=args.pipeline_status,
        child_exit_code=args.child_exit_code,
        next_retry_seconds=args.next_retry_seconds,
        git_commit=_git_commit(),
        hostname=socket.gethostname(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_report(Path(args.output), report)
    print(args.output)


if __name__ == "__main__":
    main()
