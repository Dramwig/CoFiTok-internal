from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_completion_decision import (
    build_capacity_completion_decision,
)
from cofitok.generation.capacity_scaling_result import (
    validate_capacity_scaling_50k_result,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report

try:
    from scripts.build_generation_capacity_scaling_50k_result import (
        _common_arguments as _result_arguments,
        _git_identity,
        build_from_sources as rebuild_capacity_scaling_result,
    )
except ModuleNotFoundError:
    from build_generation_capacity_scaling_50k_result import (
        _common_arguments as _result_arguments,
        _git_identity,
        build_from_sources as rebuild_capacity_scaling_result,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    _result_arguments(parser)
    parser.add_argument("--capacity-scaling-50k-result", type=Path, required=True)
    parser.add_argument(
        "--expected-capacity-scaling-50k-result-sha256",
        required=True,
    )
    parser.add_argument("--expected-completion-decision-revision", required=True)
    parser.add_argument("--expected-completion-decision-tree", required=True)
    parser.add_argument("--expected-completion-decision-branch", required=True)


def replay_capacity_scaling_result(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(args.capacity_scaling_50k_result)
    if identity["sha256"] != args.expected_capacity_scaling_50k_result_sha256:
        raise ValueError("capacity scaling 50K result SHA256 differs")
    actual = read_json_object(
        args.capacity_scaling_50k_result,
        name="capacity scaling 50K result",
    )
    expected = rebuild_capacity_scaling_result(args)
    if actual != expected:
        raise ValueError("capacity scaling 50K result is not reproducible")
    validate_capacity_scaling_50k_result(
        actual,
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
    )
    return actual, identity


def build_from_sources(args: argparse.Namespace) -> dict[str, Any]:
    decision_git = _git_identity(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_completion_decision_revision,
        "tree": args.expected_completion_decision_tree,
        "branch": args.expected_completion_decision_branch,
        "tracked_dirty": False,
    }
    if decision_git != expected_git:
        raise ValueError("capacity completion decision checkout identity differs")
    result, result_identity = replay_capacity_scaling_result(args)
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    standing = read_json_object(
        args.standing_authorization,
        name="standing experiment authorization",
    )
    return build_capacity_completion_decision(
        capacity_scaling_result=result,
        capacity_scaling_result_identity=result_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        decision_git=decision_git,
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
        expected_training_revision=args.expected_capacity_revision,
        expected_training_tree=args.expected_capacity_tree,
        expected_training_branch=args.expected_capacity_branch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-replayed decision for the exact matched 250M "
            "step-50K to configured step-100K completion segment."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity completion decision exists: {args.output}")
    decision = build_from_sources(args)
    write_json_report(args.output, decision)
    print(decision["recommended_next_stage"]["id"])


if __name__ == "__main__":
    main()
