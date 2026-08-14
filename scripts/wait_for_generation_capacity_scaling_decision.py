from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

from cofitok.generation.capacity_scaling_decision import (
    validate_capacity_scaling_decision,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report
from scripts.build_generation_capacity_scaling_decision import build_from_sources


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_capacity_scaling_decision_waiter"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _git_identity(project: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--porcelain")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact 250M/10K capacity result and build only its "
            "source-replayed step-50K scaling decision."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--capacity-probe-result", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--expected-self-revision", required=True)
    parser.add_argument("--expected-self-tree", required=True)
    parser.add_argument("--expected-self-branch", required=True)
    parser.add_argument("--expected-capacity-revision", required=True)
    parser.add_argument("--expected-capacity-branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    return args


def _status(
    args: argparse.Namespace,
    *,
    state: str,
    detail: str,
    polls: int,
    self_git: dict[str, Any],
    result_identity: dict[str, Any] | None,
    decision_identity: dict[str, Any] | None,
    recommendation: dict[str, Any] | None,
) -> None:
    write_json_report(
        args.status,
        {
            "schema_version": WAITER_SCHEMA_VERSION,
            "role": WAITER_ROLE,
            "status": state,
            "detail": detail,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "polls": polls,
            "poll_seconds": args.poll_seconds,
            "updated_at": _utc_now(),
            "self_git": self_git,
            "capacity_probe_result_path": (
                args.capacity_probe_result.resolve().as_posix()
            ),
            "capacity_probe_result": result_identity,
            "decision_path": args.decision.resolve().as_posix(),
            "decision": decision_identity,
            "recommended_next_stage": recommendation,
            "authorization_boundary": {
                "cpu_only_decision_build_allowed": True,
                "gpu_use_allowed": False,
                "training_launch_allowed": False,
                "configured_100k_completion_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_or_release_allowed": False,
            },
        },
    )


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    expected_git = {
        "revision": args.expected_self_revision,
        "tree": args.expected_self_tree,
        "branch": args.expected_self_branch,
        "tracked_dirty": False,
    }
    self_git = _git_identity(project)
    if self_git != expected_git:
        raise ValueError("capacity scaling waiter checkout identity differs")
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    polls = 0
    while True:
        polls += 1
        if not args.capacity_probe_result.is_file():
            _status(
                args,
                state="waiting",
                detail="waiting_for_exact_capacity_probe_result",
                polls=polls,
                self_git=self_git,
                result_identity=None,
                decision_identity=None,
                recommendation=None,
            )
            time.sleep(args.poll_seconds)
            continue
        result_identity = file_identity(args.capacity_probe_result)
        expected = build_from_sources(
            capacity_probe_result_path=args.capacity_probe_result.resolve(),
            expected_capacity_probe_result_sha256=result_identity["sha256"],
            standing_authorization_path=args.standing_authorization.resolve(),
            expected_standing_authorization_sha256=(
                args.expected_standing_authorization_sha256
            ),
            decision_git={
                key: self_git[key]
                for key in ("revision", "branch", "tracked_dirty")
            },
            expected_capacity_revision=args.expected_capacity_revision,
            expected_capacity_branch=args.expected_capacity_branch,
        )
        if args.decision.is_file():
            actual = read_json_object(
                args.decision,
                name="capacity scaling decision",
            )
            if actual != expected:
                raise ValueError(
                    "existing capacity scaling decision is not reproducible"
                )
        else:
            write_json_report(args.decision, expected)
        validate_capacity_scaling_decision(
            expected,
            expected_decision_revision=args.expected_self_revision,
            expected_decision_branch=args.expected_self_branch,
        )
        decision_identity = file_identity(args.decision)
        recommendation = expected["recommended_next_stage"]
        authorized = expected["execution_authorization"][
            "matched_250m_resume_allowed"
        ]
        _status(
            args,
            state="completed" if authorized else "not_selected",
            detail=(
                "matched_250m_step_10k_to_50k_decision_verified"
                if authorized
                else "capacity_result_did_not_authorize_additional_training"
            ),
            polls=polls,
            self_git=self_git,
            result_identity=result_identity,
            decision_identity=decision_identity,
            recommendation=recommendation,
        )
        return


if __name__ == "__main__":
    main()
