from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation_capacity_supervisor_repair import (
    build_supervisor_receipt_repair_approval,
)
from cofitok.reporting import write_json_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approve one exact capacity supervisor receipt repair plan."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_supervisor_receipt_repair_approval(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    write_json_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
