from __future__ import annotations

import argparse
import json

from cofitok.generation.capacity_scaling_training import (
    validate_capacity_scaling_partial_training,
)
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one exact-resumed 250M capacity-scaling run at the "
            "intentional step-50K stop."
        )
    )
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-parameter-count", type=int, required=True)
    parser.add_argument("--expected-micro-batch-size", type=int, required=True)
    parser.add_argument(
        "--expected-gradient-accumulation-steps",
        type=int,
        required=True,
    )
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_capacity_scaling_partial_training(
        report_path=args.training_report,
        config_path=args.config,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_parameter_count=args.expected_parameter_count,
        expected_micro_batch_size=args.expected_micro_batch_size,
        expected_gradient_accumulation_steps=(
            args.expected_gradient_accumulation_steps
        ),
    )
    if args.output:
        write_json_report(args.output, report)
        print(args.output)
    else:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "completed_steps": report["completed_steps"],
                    "checkpoint": report["checkpoint"],
                    "authorization_boundary": report["authorization_boundary"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
