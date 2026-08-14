from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.capacity_completion_decision import (
    validate_capacity_completion_decision,
)
from cofitok.inference_replay import file_identity, read_json_object

try:
    from scripts.build_generation_capacity_completion_decision import (
        _common_arguments,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_capacity_completion_decision import (
        _common_arguments,
        build_from_sources,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and verify the exact matched 250M step-50K to step-100K "
            "capacity-completion decision."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.decision)
    if identity["sha256"] != args.expected_decision_sha256:
        raise ValueError("capacity completion decision SHA256 differs")
    actual = read_json_object(args.decision, name="capacity completion decision")
    expected = build_from_sources(args)
    if actual != expected:
        raise ValueError("capacity completion decision is not reproducible")
    evidence = validate_capacity_completion_decision(
        actual,
        expected_decision_revision=args.expected_completion_decision_revision,
        expected_decision_tree=args.expected_completion_decision_tree,
        expected_decision_branch=args.expected_completion_decision_branch,
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "decision": identity,
                **evidence,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
