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
from cofitok.generation.exposure_capacity_authorization import identity, read_object
from cofitok.reporting import write_json_report
from scripts.build_generation_capacity_scaling_decision import build_from_sources


WAITER_SCHEMA = "cofitok_generation_capacity_scaling_decision_waiter_v2"
WAITER_ROLE = "source_bound_capacity_scaling_decision_waiter"
WAITER_BOUNDARY = {
    "cpu_only_decision_build_allowed": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_use_allowed": False,
    "training_launch_allowed": False,
    "configured_100k_completion_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}


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
        "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact four-arm capacity confirmation result and build "
            "only its non-authorizing base256 10K-to-50K preparation decision."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--capacity-confirmation-result", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--expected-self-revision", required=True)
    parser.add_argument("--expected-self-tree", required=True)
    parser.add_argument("--expected-self-branch", required=True)
    parser.add_argument("--expected-confirmation-revision", required=True)
    parser.add_argument("--expected-confirmation-tree", required=True)
    parser.add_argument("--expected-confirmation-branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args(argv)
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
    next_stage: dict[str, Any] | None,
    error: str | None = None,
) -> None:
    write_json_report(
        args.status,
        {
            "schema_version": WAITER_SCHEMA,
            "role": WAITER_ROLE,
            "status": state,
            "detail": detail,
            "error": error,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "polls": polls,
            "poll_seconds": args.poll_seconds,
            "updated_at": _utc_now(),
            "self_git": self_git,
            "capacity_confirmation_result_path": (
                args.capacity_confirmation_result.resolve().as_posix()
            ),
            "capacity_confirmation_result": result_identity,
            "decision_path": args.decision.resolve().as_posix(),
            "decision": decision_identity,
            "next_stage": next_stage,
            "authorization_boundary": dict(WAITER_BOUNDARY),
        },
    )


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    self_git = _git_identity(project)
    expected_git = {
        "revision": args.expected_self_revision,
        "tree": args.expected_self_tree,
        "branch": args.expected_self_branch,
        "tracked_dirty": False,
    }
    if self_git != expected_git:
        raise ValueError("capacity scaling waiter checkout identity differs")
    confirmation_git = {
        "revision": args.expected_confirmation_revision,
        "tree": args.expected_confirmation_tree,
        "branch": args.expected_confirmation_branch,
        "tracked_dirty": False,
    }
    polls = 0
    while True:
        polls += 1
        if not args.capacity_confirmation_result.is_file():
            _status(
                args,
                state="waiting",
                detail="waiting_for_exact_capacity_confirmation_result",
                polls=polls,
                self_git=self_git,
                result_identity=None,
                decision_identity=None,
                next_stage=None,
            )
            time.sleep(args.poll_seconds)
            continue
        result_identity = identity(args.capacity_confirmation_result)
        try:
            expected = build_from_sources(
                capacity_confirmation_result_path=(
                    args.capacity_confirmation_result.resolve()
                ),
                expected_capacity_confirmation_result_sha256=(
                    result_identity["sha256"]
                ),
                confirmation_checkout=confirmation_git,
                decision_git=self_git,
            )
            if args.decision.is_file():
                actual = read_object(args.decision, name="capacity scaling decision")
                if actual != expected:
                    raise ValueError(
                        "existing capacity scaling decision is not reproducible"
                    )
            else:
                write_json_report(args.decision, expected)
            validate_capacity_scaling_decision(
                expected,
                expected_decision_revision=args.expected_self_revision,
                expected_decision_tree=args.expected_self_tree,
                expected_decision_branch=args.expected_self_branch,
            )
        except Exception as error:
            _status(
                args,
                state="failed",
                detail="capacity_confirmation_replay_or_decision_failed",
                polls=polls,
                self_git=self_git,
                result_identity=result_identity,
                decision_identity=(
                    identity(args.decision) if args.decision.is_file() else None
                ),
                next_stage=None,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        decision_identity = identity(args.decision)
        next_stage = expected["next_stage"]
        selected = next_stage["capacity_scaling_preparation_allowed"]
        _status(
            args,
            state="completed" if selected else "not_selected",
            detail=(
                "base256_10k_to_50k_preparation_decision_verified"
                if selected
                else "capacity_confirmation_did_not_select_scaling_preparation"
            ),
            polls=polls,
            self_git=self_git,
            result_identity=result_identity,
            decision_identity=decision_identity,
            next_stage=next_stage,
        )
        return


if __name__ == "__main__":
    main()
