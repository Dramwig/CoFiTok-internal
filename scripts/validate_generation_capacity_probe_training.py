from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation.capacity_probe_training import (
    validate_capacity_probe_partial_training,
)
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an exact intentional 10K capacity-probe training stop."
    )
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-stop-step", type=int, default=10_000)
    parser.add_argument("--expected-configured-steps", type=int, default=100_000)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-parameter-count", type=int, required=True)
    parser.add_argument("--expected-micro-batch-size", type=int, required=True)
    parser.add_argument(
        "--expected-gradient-accumulation-steps",
        type=int,
        required=True,
    )
    parser.add_argument("--expected-dataset", default="imagenet_256")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_capacity_probe_partial_training(
        report_path=args.training_report,
        config_path=args.config,
        expected_stop_step=args.expected_stop_step,
        expected_configured_steps=args.expected_configured_steps,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_parameter_count=args.expected_parameter_count,
        expected_micro_batch_size=args.expected_micro_batch_size,
        expected_gradient_accumulation_steps=(
            args.expected_gradient_accumulation_steps
        ),
        expected_dataset=args.expected_dataset,
    )
    write_json_report(Path(args.output), report)
    print(args.output)


if __name__ == "__main__":
    main()
