from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.capacity_scaling_decision import (
    validate_capacity_scaling_decision,
)
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)

try:
    from scripts.build_generation_capacity_scaling_decision import (
        PROJECT_ROOT,
        _common_arguments,
        _expected_git,
        build_from_sources,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from build_generation_capacity_scaling_decision import (
        PROJECT_ROOT,
        _common_arguments,
        _expected_git,
        build_from_sources,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and verify the non-authorizing matched base256 10K-to-50K "
            "capacity-scaling preparation decision."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    decision_identity = identity(args.decision)
    if decision_identity["sha256"] != args.expected_decision_sha256:
        raise ValueError("capacity scaling decision SHA256 differs")
    decision_git = checkout_identity(PROJECT_ROOT)
    expected_decision_git = _expected_git(
        revision=args.expected_decision_revision,
        tree=args.expected_decision_tree,
        branch=args.expected_decision_branch,
    )
    if decision_git != expected_decision_git:
        raise ValueError("capacity scaling verifier checkout identity differs")
    confirmation_git = _expected_git(
        revision=args.expected_confirmation_revision,
        tree=args.expected_confirmation_tree,
        branch=args.expected_confirmation_branch,
    )
    actual = read_object(args.decision, name="capacity scaling decision")
    expected = build_from_sources(
        capacity_confirmation_result_path=(
            args.capacity_confirmation_result.resolve()
        ),
        expected_capacity_confirmation_result_sha256=(
            args.expected_capacity_confirmation_result_sha256
        ),
        confirmation_checkout=confirmation_git,
        decision_git=decision_git,
    )
    if actual != expected:
        raise ValueError("capacity scaling decision is not reproducible")
    evidence = validate_capacity_scaling_decision(
        actual,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "decision": decision_identity,
                **evidence,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
