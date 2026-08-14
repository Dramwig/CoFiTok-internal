from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from scripts.build_generation_capacity_probe_launch_receipt import (
    _common_arguments,
    build_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the immutable 250M/10K capacity-probe launch receipt."
    )
    _common_arguments(parser)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.receipt)
    if identity["sha256"] != args.expected_receipt_sha256:
        raise ValueError("capacity probe launch receipt SHA256 differs")
    actual = read_json_object(args.receipt, name="capacity probe launch receipt")
    expected = build_from_args(
        args,
        require_training_state_absent=False,
        require_current_git=True,
    )
    if actual != expected:
        raise ValueError("capacity probe launch receipt is not reproducible")
    print(args.receipt)


if __name__ == "__main__":
    main()
