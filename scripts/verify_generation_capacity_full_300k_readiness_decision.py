from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.capacity_full_readiness_decision import (
    validate_capacity_full_readiness_decision,
)
from cofitok.inference_replay import file_identity, read_json_object

try:
    from scripts.build_generation_capacity_full_300k_readiness_decision import (
        _common_arguments,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_capacity_full_300k_readiness_decision import (
        _common_arguments,
        build_from_sources,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay and verify a capacity-full 300K readiness decision."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--decision", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual = read_json_object(
        args.decision,
        name="capacity-full 300K readiness decision",
    )
    expected = build_from_sources(args)
    if actual != expected:
        raise ValueError("capacity-full readiness decision is not reproducible")
    evidence = validate_capacity_full_readiness_decision(
        actual,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "decision": file_identity(args.decision),
                **evidence,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
