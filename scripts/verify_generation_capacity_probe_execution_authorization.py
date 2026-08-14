from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from scripts.build_generation_capacity_probe_execution_authorization import (
    _common_arguments,
    build_from_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an exact capacity-probe-only execution authorization."
    )
    _common_arguments(parser)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.authorization)
    if identity["sha256"] != args.expected_authorization_sha256:
        raise ValueError("capacity probe execution authorization SHA256 differs")
    actual = read_json_object(
        args.authorization,
        name="capacity probe execution authorization",
    )
    expected = build_from_paths(args, require_current_git=True)
    if actual != expected:
        raise ValueError("capacity probe execution authorization is not reproducible")
    print(args.authorization)


if __name__ == "__main__":
    main()
