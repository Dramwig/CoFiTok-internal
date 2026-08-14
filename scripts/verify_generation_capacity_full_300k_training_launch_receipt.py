from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object

try:
    from scripts.build_generation_capacity_full_300k_training_launch_receipt import (
        _common_arguments,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_capacity_full_300k_training_launch_receipt import (
        _common_arguments,
        build_from_sources,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay an immutable experimental capacity-full fresh "
            "matched 300K training authorization receipt."
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
        raise ValueError("capacity-full training receipt SHA256 differs")
    actual = read_json_object(args.receipt, name="capacity-full training receipt")
    expected = build_from_sources(args, require_training_state_absent=False)
    if actual != expected:
        raise ValueError("capacity-full training receipt is not reproducible")
    print(json.dumps({"receipt": identity, "status": "authorized"}, sort_keys=True))


if __name__ == "__main__":
    main()
