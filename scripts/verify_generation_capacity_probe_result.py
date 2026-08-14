from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from scripts.build_generation_capacity_probe_result import (
    _common_arguments,
    build_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the physical evidence for a four-arm capacity-probe result."
    )
    _common_arguments(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.result)
    if identity["sha256"] != args.expected_result_sha256:
        raise ValueError("capacity probe result SHA256 differs")
    actual = read_json_object(args.result, name="capacity probe result")
    expected = build_from_args(args)
    if actual != expected:
        raise ValueError("capacity probe result is not reproducible")
    print(args.result)


if __name__ == "__main__":
    main()
