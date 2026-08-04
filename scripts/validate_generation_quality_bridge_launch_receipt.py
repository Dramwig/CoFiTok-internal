from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from scripts.build_generation_quality_bridge_launch_receipt import (
    _common_arguments,
    build_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the immutable launch receipt before resuming the full-data "
            "100K quality bridge."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--allow-later-git-revision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt_identity = file_identity(args.receipt)
    if receipt_identity["sha256"] != args.expected_receipt_sha256:
        raise ValueError("quality bridge launch receipt SHA256 differs")
    actual = read_json_object(args.receipt, name="quality bridge launch receipt")
    expected = build_from_args(
        args,
        require_training_state_absent=False,
        require_current_git=not args.allow_later_git_revision,
    )
    if actual != expected:
        raise ValueError("quality bridge launch receipt is not reproducible")
    print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
