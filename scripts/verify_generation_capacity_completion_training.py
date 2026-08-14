from __future__ import annotations

import argparse
import json

from cofitok.generation.capacity_completion_training import (
    validate_capacity_completion_training,
)
from cofitok.inference_replay import file_identity, read_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one exact-resumed 250M step-100K training validation."
    )
    parser.add_argument("--validation", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-parameter-count", type=int, required=True)
    parser.add_argument("--expected-micro-batch-size", type=int, required=True)
    parser.add_argument(
        "--expected-gradient-accumulation-steps", type=int, required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.validation)
    if identity["sha256"] != args.expected_validation_sha256:
        raise ValueError("capacity completion training validation SHA256 differs")
    actual = read_json_object(
        args.validation,
        name="capacity completion training validation",
    )
    expected = validate_capacity_completion_training(
        report_path=args.training_report,
        config_path=args.config,
        source_archive_path=args.source_archive,
        expected_source_archive_sha256=args.expected_source_archive_sha256,
        method=args.method,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_parameter_count=args.expected_parameter_count,
        expected_micro_batch_size=args.expected_micro_batch_size,
        expected_gradient_accumulation_steps=(
            args.expected_gradient_accumulation_steps
        ),
    )
    if actual != expected:
        raise ValueError("capacity completion training validation is not reproducible")
    print(json.dumps({"status": "verified", "validation": identity}, sort_keys=True))


if __name__ == "__main__":
    main()
