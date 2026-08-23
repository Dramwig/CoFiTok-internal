from __future__ import annotations

import argparse
import copy
from pathlib import Path

from cofitok.generation.post_reconciliation_decision import (
    AUTHORIZATION_BOUNDARY,
    POST_RECONCILIATION_DECISION_ROLE,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import write_json_report

try:
    from scripts.build_generation_100k_post_reconciliation_decision import (
        add_source_arguments,
        build_from_paths,
    )
except ModuleNotFoundError:
    from build_generation_100k_post_reconciliation_decision import (
        add_source_arguments,
        build_from_paths,
    )


VERIFICATION_ROLE = "generation_100k_post_reconciliation_decision_verification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically replay a 100K post-reconciliation decision."
    )
    add_source_arguments(parser)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_verification(args: argparse.Namespace) -> dict:
    decision_identity = file_identity(args.decision)
    if decision_identity["sha256"] != args.expected_decision_sha256:
        raise ValueError("post-reconciliation decision SHA256 differs")
    actual = read_json_object(args.decision, name="post-reconciliation decision")
    expected = build_from_paths(args)
    if actual != expected:
        raise ValueError("post-reconciliation decision is not reproducible")
    if (
        actual.get("role") != POST_RECONCILIATION_DECISION_ROLE
        or actual.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or actual.get("terminal_status") != "hold"
        or actual.get("generation_advantage_proven") is not False
    ):
        raise ValueError("post-reconciliation decision boundary differs")
    return {
        "schema_version": 1,
        "role": VERIFICATION_ROLE,
        "status": "verified",
        "decision": decision_identity,
        "source_replay": copy.deepcopy(actual["source_replay"]),
        "scientific_resolution": copy.deepcopy(actual["scientific_resolution"]),
        "recommended_next_stage": copy.deepcopy(actual["recommended_next_stage"]),
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def main() -> None:
    args = parse_args()
    with exclusive_output_lock(args.output, role=VERIFICATION_ROLE):
        report = build_verification(args)
        if args.output.is_file():
            if not args.resume:
                raise FileExistsError(
                    "decision verification exists; pass --resume to replay it"
                )
            existing = read_json_object(
                args.output,
                name="post-reconciliation decision verification",
            )
            if existing != report:
                raise ValueError("existing decision verification differs")
            print(f"reused {args.output}")
            return
        write_json_report(args.output, report)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
