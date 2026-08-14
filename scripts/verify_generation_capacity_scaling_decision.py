from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.capacity_scaling_decision import (
    validate_capacity_scaling_decision,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance

try:
    from scripts.build_generation_capacity_scaling_decision import (
        PROJECT_ROOT,
        _common_arguments,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_capacity_scaling_decision import (
        PROJECT_ROOT,
        _common_arguments,
        build_from_sources,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and verify a matched 250M step-10K to step-50K "
            "capacity-scaling decision."
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
        raise ValueError("capacity scaling decision SHA256 differs")
    decision_git = git_provenance(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_decision_revision,
        "branch": args.expected_decision_branch,
        "tracked_dirty": False,
    }
    if decision_git != expected_git:
        raise ValueError("capacity scaling verifier checkout identity differs")
    actual = read_json_object(args.decision, name="capacity scaling decision")
    expected = build_from_sources(
        capacity_probe_result_path=args.capacity_probe_result.resolve(),
        expected_capacity_probe_result_sha256=(
            args.expected_capacity_probe_result_sha256
        ),
        standing_authorization_path=args.standing_authorization.resolve(),
        expected_standing_authorization_sha256=(
            args.expected_standing_authorization_sha256
        ),
        decision_git=decision_git,
        expected_capacity_revision=args.expected_capacity_revision,
        expected_capacity_branch=args.expected_capacity_branch,
    )
    if actual != expected:
        raise ValueError("capacity scaling decision is not reproducible")
    evidence = validate_capacity_scaling_decision(
        actual,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_branch=args.expected_decision_branch,
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
