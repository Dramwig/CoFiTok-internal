from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from cofitok.generation_capacity_supervisor_repair import (
    REPAIR_EXECUTION_ROLE,
    REPAIR_EXECUTION_SCHEMA_VERSION,
    execute_supervisor_receipt_repair,
)
from cofitok.reporting import write_json_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute one exact capacity supervisor receipt repair."
    )
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    def publish(value: dict) -> None:
        write_json_report(args.output, value)

    try:
        report = execute_supervisor_receipt_repair(
            approval_path=args.approval,
            expected_approval_sha256=args.expected_approval_sha256,
            publish=publish,
            status_timeout_seconds=args.status_timeout_seconds,
        )
    except Exception as error:
        write_json_report(
            args.output,
            {
                "schema_version": REPAIR_EXECUTION_SCHEMA_VERSION,
                "role": REPAIR_EXECUTION_ROLE,
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "detail": str(error),
            },
        )
        raise
    if report.get("status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
