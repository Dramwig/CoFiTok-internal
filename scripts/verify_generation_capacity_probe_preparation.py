from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from scripts.build_generation_capacity_probe_preparation import (
    _common_arguments,
    build_from_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently replay a matched 250M/10K capacity preparation."
    )
    _common_arguments(parser)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.preparation)
    if identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("capacity probe preparation SHA256 differs")
    actual = read_json_object(
        args.preparation,
        name="capacity probe preparation",
    )
    expected = build_from_paths(args)
    if actual != expected:
        raise ValueError("capacity probe preparation is not reproducible")
    print(args.preparation)


if __name__ == "__main__":
    main()
