from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object, reject_symlink_chain

try:
    from build_generation_factorization_quality_regression_execution_authorization import (
        build_from_paths as build_authorization_from_paths,
    )
    from build_generation_factorization_quality_regression_source_binding import (
        build_from_paths as build_source_binding_from_paths,
    )
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts.build_generation_factorization_quality_regression_execution_authorization import (
        build_from_paths as build_authorization_from_paths,
    )
    from scripts.build_generation_factorization_quality_regression_source_binding import (
        build_from_paths as build_source_binding_from_paths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the source binding and execution authorization before the matched "
            "factorization quality-regression diagnostic loads either checkpoint."
        )
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--expected-source-binding-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--terminal-system-guard", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization_path = reject_symlink_chain(
        args.authorization,
        name="factorization-regression execution authorization",
    ).resolve()
    authorization_identity = file_identity(authorization_path)
    if authorization_identity["sha256"] != args.expected_authorization_sha256:
        raise ValueError("factorization-regression authorization SHA256 differs")
    authorization = read_json_object(
        authorization_path,
        name="factorization-regression execution authorization",
    )
    expected_binding = build_source_binding_from_paths(
        preparation_path=args.preparation,
        expected_preparation_sha256=args.expected_preparation_sha256,
        followup_decision_path=args.followup_decision,
        terminal_system_guard_path=args.terminal_system_guard,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )
    binding_path = reject_symlink_chain(
        args.source_binding,
        name="factorization-regression source binding",
    ).resolve()
    binding_identity = file_identity(binding_path)
    if binding_identity["sha256"] != args.expected_source_binding_sha256:
        raise ValueError("factorization-regression source-binding SHA256 differs")
    binding = read_json_object(binding_path, name="factorization-regression source binding")
    if binding != expected_binding:
        raise ValueError("factorization-regression source binding is not reproducible")
    expected_authorization = build_authorization_from_paths(
        preparation_path=args.preparation,
        expected_preparation_sha256=args.expected_preparation_sha256,
        source_binding_path=binding_path,
        expected_source_binding_sha256=args.expected_source_binding_sha256,
        standing_authorization_path=args.standing_authorization,
        expected_standing_authorization_sha256=(
            args.expected_standing_authorization_sha256
        ),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )
    if authorization != expected_authorization:
        raise ValueError("factorization-regression authorization is not reproducible")
    print(authorization_identity["sha256"])


if __name__ == "__main__":
    main()
