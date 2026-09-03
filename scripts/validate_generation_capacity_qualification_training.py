from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation.capacity_qualification_training import (
    CAPACITY_QUALIFICATION_STAGES,
    validate_capacity_qualification_partial_training,
)
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a clean 10K stop of one 250M capacity-qualification arm."
    )
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-parameter-count", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument(
        "--expected-stage",
        choices=tuple(sorted(CAPACITY_QUALIFICATION_STAGES)),
        default="stability_capacity_qualification",
    )
    parser.add_argument("--expected-base-channels", type=int, default=256)
    parser.add_argument(
        "--allow-exact-resume",
        action="store_true",
        help=(
            "Accept only an integrity-bound same-run checkpoint resume below the "
            "intentional 10K stop."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_capacity_qualification_partial_training(
        report_path=args.training_report,
        config_path=args.config,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_parameter_count=args.expected_parameter_count,
        expected_micro_batch_size=args.micro_batch_size,
        expected_gradient_accumulation_steps=args.gradient_accumulation_steps,
        expected_stage=args.expected_stage,
        expected_base_channels=args.expected_base_channels,
        allow_exact_resume=args.allow_exact_resume,
    )
    write_json_report(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
