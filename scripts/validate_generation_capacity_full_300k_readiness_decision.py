from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.capacity_full_readiness_decision import (
    validate_capacity_full_readiness_decision,
)
from cofitok.inference_replay import file_identity, read_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a capacity-full readiness decision and every bound source "
            "identity without rebuilding or authorizing training."
        )
    )
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--expected-result-revision", required=True)
    parser.add_argument("--expected-result-tree", required=True)
    parser.add_argument("--expected-result-branch", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.decision)
    if identity["sha256"] != args.expected_decision_sha256:
        raise ValueError("capacity-full readiness decision SHA256 differs")
    report = read_json_object(
        args.decision,
        name="capacity-full readiness decision",
    )
    evidence = validate_capacity_full_readiness_decision(
        report,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
    )
    for name, expected in report["source_reports"].items():
        if file_identity(expected["path"]) != expected:
            raise ValueError(f"capacity-full readiness decision source changed: {name}")
    print(json.dumps({"decision": identity, **evidence}, sort_keys=True))


if __name__ == "__main__":
    main()
