from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.capacity_scaling_execution import (
    validate_capacity_scaling_launch_receipt,
)
from cofitok.inference_replay import file_identity, read_json_object

try:
    from scripts.build_generation_capacity_scaling_launch_receipt import (
        _common_arguments,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_capacity_scaling_launch_receipt import (
        _common_arguments,
        build_from_sources,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and verify an exact matched 250M step-10K to step-50K "
            "capacity-scaling launch receipt."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.receipt)
    if identity["sha256"] != args.expected_receipt_sha256:
        raise ValueError("capacity scaling launch receipt SHA256 differs")
    actual = read_json_object(args.receipt, name="capacity scaling launch receipt")
    expected = build_from_sources(
        args,
        require_initial_state=False,
        require_idle_launch=False,
    )
    if actual != expected:
        raise ValueError("capacity scaling launch receipt is not reproducible")
    evidence = validate_capacity_scaling_launch_receipt(
        actual,
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_output_root=args.output_root,
    )
    print(json.dumps({"status": "verified", "receipt": identity, **evidence}, sort_keys=True))


if __name__ == "__main__":
    main()
