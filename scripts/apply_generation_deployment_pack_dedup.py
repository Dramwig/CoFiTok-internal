from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.deployment_pack_dedup import apply_deployment_pack_dedup_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an exact, receipt-bound deployment Git-pack hardlink plan."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--allow-checkout", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = apply_deployment_pack_dedup_plan(
        plan=plan,
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        output_path=args.output,
        allow_checkouts=set(args.allow_checkout),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
