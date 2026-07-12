from __future__ import annotations

import argparse
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.reporting import write_json_report


PIPELINE_NAME = "generation_complete_after_10pct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically update the large-scale generation pipeline status."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--status",
        required=True,
        choices=("running", "pass", "failed"),
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--exit-code", type=int)
    return parser.parse_args()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_status(
    *,
    status: str,
    stage: str,
    detail: str,
    exit_code: int | None,
    git_commit: str,
    hostname: str,
    updated_at: str,
) -> dict[str, Any]:
    if status not in {"running", "pass", "failed"}:
        raise ValueError(f"unsupported pipeline status: {status}")
    if status == "failed" and exit_code is None:
        raise ValueError("failed pipeline status requires an exit code")
    if status != "failed" and exit_code is not None:
        raise ValueError("exit code is only valid for failed pipeline status")
    return {
        "schema_version": 1,
        "pipeline": PIPELINE_NAME,
        "status": status,
        "stage": stage,
        "detail": detail,
        "exit_code": exit_code,
        "git_commit": git_commit,
        "hostname": hostname,
        "updated_at": updated_at,
    }


def main() -> None:
    args = parse_args()
    report = build_status(
        status=args.status,
        stage=args.stage,
        detail=args.detail,
        exit_code=args.exit_code,
        git_commit=_git_commit(),
        hostname=socket.gethostname(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
