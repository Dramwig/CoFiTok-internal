from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.reporting import file_sha256
from scripts.build_generation_epsilon_stability_execution_authorization import (
    build_from_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay and validate an epsilon-stability execution gate."
    )
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument(
        "--expected-execution-authorization-sha256",
        required=True,
    )
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--user-authorization", required=True)
    parser.add_argument("--expected-user-authorization-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.execution_authorization)
    if file_sha256(path) != args.expected_execution_authorization_sha256:
        raise ValueError("epsilon-stability execution authorization SHA256 differs")
    with path.open("r", encoding="utf-8") as handle:
        actual = json.load(handle)
    expected = build_from_paths(
        preparation_path=args.preparation,
        expected_preparation_sha256=args.expected_preparation_sha256,
        design_path=args.design,
        expected_design_sha256=args.expected_design_sha256,
        user_authorization_path=args.user_authorization,
        expected_user_authorization_sha256=(
            args.expected_user_authorization_sha256
        ),
    )
    if actual != expected:
        raise ValueError("epsilon-stability execution authorization does not replay")
    print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
