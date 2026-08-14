from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.capacity_full_readiness import (
    validate_capacity_full_readiness,
)
from cofitok.inference_replay import file_identity, read_json_object

try:
    from scripts.build_generation_capacity_full_300k_readiness import (
        _common_arguments,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_capacity_full_300k_readiness import (
        _common_arguments,
        build_from_sources,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically replay a capacity-full fresh-300K readiness artifact."
    )
    _common_arguments(parser)
    parser.add_argument("--readiness", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual = read_json_object(args.readiness, name="capacity-full readiness")
    expected = build_from_sources(args)
    if actual != expected:
        raise ValueError("capacity-full readiness is not reproducible")
    evidence = validate_capacity_full_readiness(
        actual,
        expected_readiness_revision=args.expected_readiness_revision,
        expected_readiness_tree=args.expected_readiness_tree,
        expected_readiness_branch=args.expected_readiness_branch,
    )
    print(
        json.dumps(
            {"readiness": file_identity(args.readiness), **evidence},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
