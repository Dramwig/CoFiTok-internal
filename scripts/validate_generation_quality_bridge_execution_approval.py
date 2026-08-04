from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.quality_bridge import (
    validate_quality_bridge_execution_approval,
)
from cofitok.inference_replay import file_identity, read_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a user-created, execution-only approval sentinel for the "
            "full-data 100K quality bridge."
        )
    )
    parser.add_argument("--approval", required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    approval_path = Path(args.approval)
    preparation_path = Path(args.preparation)
    approval_identity = file_identity(approval_path)
    preparation_identity = file_identity(preparation_path)
    if approval_identity["sha256"] != args.expected_approval_sha256:
        raise ValueError("quality bridge execution approval SHA256 differs")
    if preparation_identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("quality bridge preparation SHA256 differs")
    evidence = validate_quality_bridge_execution_approval(
        read_json_object(approval_path, name="quality bridge execution approval"),
        preparation_identity=preparation_identity,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_output_root=Path(args.expected_output_root).resolve().as_posix(),
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "approval": approval_identity,
                "preparation": preparation_identity,
                "evidence": evidence,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
