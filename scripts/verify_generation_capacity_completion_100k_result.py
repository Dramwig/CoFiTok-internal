from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.capacity_completion_result import (
    validate_capacity_completion_100k_result,
)
from cofitok.inference_replay import file_identity, read_json_object

try:
    from scripts.build_generation_capacity_completion_100k_result import (
        _common_arguments,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_capacity_completion_100k_result import (
        _common_arguments,
        build_from_sources,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild and verify the exact capacity-completion 100K result."
    )
    _common_arguments(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.result)
    if identity["sha256"] != args.expected_result_sha256:
        raise ValueError("capacity completion 100K result SHA256 differs")
    actual = read_json_object(
        args.result,
        name="capacity completion 100K result",
    )
    expected = build_from_sources(args)
    if actual != expected:
        raise ValueError("capacity completion 100K result is not reproducible")
    evidence = validate_capacity_completion_100k_result(
        actual,
        expected_execution_revision=args.expected_completion_execution_revision,
        expected_execution_tree=args.expected_completion_execution_tree,
        expected_execution_branch=args.expected_completion_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
    )
    print(json.dumps({"status": "verified", "result": identity, **evidence}, sort_keys=True))


if __name__ == "__main__":
    main()
