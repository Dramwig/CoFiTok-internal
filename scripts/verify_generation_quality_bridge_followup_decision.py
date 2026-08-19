from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance

try:
    from scripts.build_generation_quality_bridge_followup_decision import (
        PROJECT_ROOT,
        _common_arguments,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_quality_bridge_followup_decision import (
        PROJECT_ROOT,
        _common_arguments,
        build_from_sources,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently replay a non-authorizing 100K quality-bridge "
            "next-experiment decision."
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
        raise ValueError("quality bridge follow-up decision SHA256 differs")
    decision_git = git_provenance(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_decision_revision,
        "branch": args.expected_decision_branch,
        "tracked_dirty": False,
    }
    if decision_git != expected_git:
        raise ValueError("quality bridge follow-up verifier checkout identity differs")
    actual = read_json_object(args.decision, name="quality bridge follow-up decision")
    expected = build_from_sources(
        quality_bridge_result_path=args.quality_bridge_result,
        expected_quality_bridge_result_sha256=(
            args.expected_quality_bridge_result_sha256
        ),
        training_exposure_report_path=args.training_exposure_report,
        expected_training_exposure_report_sha256=(
            args.expected_training_exposure_report_sha256
        ),
        decision_git=decision_git,
    )
    if actual != expected:
        raise ValueError("quality bridge follow-up decision is not reproducible")
    print(
        json.dumps(
            {
                "status": "verified",
                "decision": identity,
                "recommended_next_stage": actual["recommended_next_stage"],
                "authorization_boundary": actual["authorization_boundary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
