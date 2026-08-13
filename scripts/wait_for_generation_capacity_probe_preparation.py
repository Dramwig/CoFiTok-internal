from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report
from scripts.build_generation_capacity_probe_preparation import build_from_paths


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_capacity_probe_preparation_waiter"
CAPACITY_RECOMMENDATION = "prepare_matched_250m_capacity_qualification_probe"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the quality-bridge decision and prepare the matched capacity "
            "probe only when that exact evidence branch is selected."
        )
    )
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--expected-self-revision", required=True)
    parser.add_argument("--expected-self-tree", required=True)
    parser.add_argument("--expected-self-branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--build-argument", action="append", default=[])
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    return args


def _parse_build_arguments(values: list[str]) -> argparse.Namespace:
    parsed: dict[str, Any] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name.strip() or not raw:
            raise ValueError("build argument must use NAME=VALUE")
        key = name.strip().replace("-", "_")
        if key in parsed:
            raise ValueError(f"duplicate build argument: {key}")
        parsed[key] = raw
    required = {
        "base_cofitok_config",
        "base_dense_config",
        "capacity_cofitok_config",
        "capacity_dense_config",
        "cofitok_reference_receipt",
        "dense_reference_receipt",
        "expected_preparation_revision",
        "expected_preparation_branch",
        "output_root",
    }
    if set(parsed) != required:
        raise ValueError(
            "capacity preparation waiter build-argument set differs: "
            f"{sorted(set(parsed) ^ required)}"
        )
    for name in (
        "base_cofitok_config",
        "base_dense_config",
        "capacity_cofitok_config",
        "capacity_dense_config",
        "cofitok_reference_receipt",
        "dense_reference_receipt",
    ):
        parsed[name] = Path(parsed[name])
    return argparse.Namespace(**parsed)


def _status(
    args: argparse.Namespace,
    *,
    status: str,
    detail: str,
    polls: int,
    recommendation: str | None,
    decision_identity: dict[str, Any] | None,
    preparation_identity: dict[str, Any] | None = None,
) -> None:
    write_json_report(
        args.status,
        {
            "schema_version": WAITER_SCHEMA_VERSION,
            "role": WAITER_ROLE,
            "status": status,
            "detail": detail,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "polls": polls,
            "poll_seconds": args.poll_seconds,
            "updated_at": _utc_now(),
            "self_git": {
                "revision": args.expected_self_revision,
                "tree": args.expected_self_tree,
                "branch": args.expected_self_branch,
                "tracked_dirty": False,
            },
            "followup_decision_path": args.followup_decision.resolve().as_posix(),
            "followup_decision": decision_identity,
            "observed_recommendation": recommendation,
            "preparation_path": args.preparation.resolve().as_posix(),
            "preparation": preparation_identity,
            "authorization_boundary": {
                "preparation_build_allowed": True,
                "capacity_probe_execution_allowed": False,
                "gpu_use_allowed": False,
                "training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "release_allowed": False,
            },
        },
    )


def main() -> None:
    args = parse_args()
    build_args = _parse_build_arguments(args.build_argument)
    polls = 0
    while True:
        polls += 1
        if not args.followup_decision.is_file():
            _status(
                args,
                status="waiting",
                detail="waiting_for_quality_bridge_followup_decision",
                polls=polls,
                recommendation=None,
                decision_identity=None,
            )
            time.sleep(args.poll_seconds)
            continue
        decision_identity = file_identity(args.followup_decision)
        decision = read_json_object(
            args.followup_decision,
            name="quality bridge follow-up decision",
        )
        recommendation = decision.get("recommended_next_stage", {}).get("id")
        if recommendation != CAPACITY_RECOMMENDATION:
            _status(
                args,
                status="not_selected",
                detail="followup_decision_selected_another_evidence_branch",
                polls=polls,
                recommendation=(
                    str(recommendation) if recommendation is not None else None
                ),
                decision_identity=decision_identity,
            )
            return
        build_args.followup_decision = args.followup_decision
        build_args.expected_followup_decision_sha256 = decision_identity["sha256"]
        expected = build_from_paths(build_args)
        if args.preparation.is_file():
            actual = read_json_object(
                args.preparation,
                name="capacity probe preparation",
            )
            if actual != expected:
                raise ValueError("existing capacity probe preparation is not reproducible")
        else:
            write_json_report(args.preparation, expected)
        preparation_identity = file_identity(args.preparation)
        _status(
            args,
            status="completed",
            detail="matched_capacity_probe_preparation_verified",
            polls=polls,
            recommendation=str(recommendation),
            decision_identity=decision_identity,
            preparation_identity=preparation_identity,
        )
        return


if __name__ == "__main__":
    main()
