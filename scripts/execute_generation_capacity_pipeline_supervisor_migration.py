from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cofitok.generation_control_process_migration import (
    EXECUTION_ROLE,
    EXECUTION_SCHEMA_VERSION,
    execute_supervisor_migration,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate exactly six source-bound waiting capacity supervisors, "
            "with partial-launch rollback and old-command restoration."
        )
    )
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-grace-seconds", type=float, default=10.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"migration execution output exists: {args.output}")
    if args.stop_grace_seconds <= 0 or args.status_timeout_seconds <= 0:
        raise ValueError("migration timeouts must be positive")

    with exclusive_output_lock(args.output, role=EXECUTION_ROLE):
        try:
            report = execute_supervisor_migration(
                approval_path=args.approval,
                expected_approval_sha256=args.expected_approval_sha256,
                publish=lambda value: write_json_report(args.output, dict(value)),
                stop_grace_seconds=args.stop_grace_seconds,
                status_timeout_seconds=args.status_timeout_seconds,
            )
        except Exception as error:
            report = {
                "schema_version": EXECUTION_SCHEMA_VERSION,
                "role": EXECUTION_ROLE,
                "status": "failed_before_signal",
                "failed_at": _timestamp(),
                "error_type": type(error).__name__,
                "detail": str(error),
                "effects": {
                    "processes_signaled": False,
                    "processes_launched": False,
                    "gpu_queried_or_allocated": False,
                    "formal_checkout_modified": False,
                },
            }
            write_json_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
