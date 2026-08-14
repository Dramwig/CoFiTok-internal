from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation_control_process_relaunch import execute_process_relaunch
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relaunch only the approval-selected CPU control processes in detached "
            "sessions, with append-only logs and owned-process rollback."
        )
    )
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--rollback-grace-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"process relaunch execution output exists: {args.output}")
    if args.status_timeout_seconds <= 0 or args.rollback_grace_seconds <= 0:
        raise ValueError("process relaunch timeouts must be positive")

    def publish(payload: object) -> None:
        write_json_report(args.output, dict(payload))

    report = execute_process_relaunch(
        approval_path=args.approval,
        expected_approval_sha256=args.expected_approval_sha256,
        publish=publish,
        status_timeout_seconds=args.status_timeout_seconds,
        rollback_grace_seconds=args.rollback_grace_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
