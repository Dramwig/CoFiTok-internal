from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )
except ModuleNotFoundError:
    from build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a milestone report and its content-addressed source reports."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument(
        "--source-profile",
        choices=("full", "stability_full", "quality_bridge", "capacity_scaling"),
        default="full",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    source_verification = verify_milestone_source_reports(
        report,
        source_profile=args.source_profile,
    )
    evidence, warnings = validate_milestone_report(
        report,
        expected_step=args.expected_step,
        source_verification=source_verification,
        expected_source_profile=args.source_profile,
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "milestone_step": args.expected_step,
                "evidence": evidence,
                "warnings": warnings,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
