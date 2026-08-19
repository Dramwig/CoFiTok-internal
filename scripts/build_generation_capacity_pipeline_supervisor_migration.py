from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation_control_process_migration import (
    SUPERVISOR_NAMES,
    build_supervisor_migration_approval,
    build_supervisor_migration_plan,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound, no-effect plan for migrating exactly six "
            "waiting capacity supervisors to a heartbeat-hardened checkout."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--expected-formal-revision", required=True)
    parser.add_argument("--expected-formal-tree", required=True)
    parser.add_argument("--expected-formal-branch", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument(
        "--process",
        action="append",
        required=True,
        metavar="NAME=PID",
        help="Exact live supervisor name and PID; provide all six once.",
    )
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--approval-output", type=Path, required=True)
    return parser.parse_args()


def _processes(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        name, separator, raw_pid = value.partition("=")
        if not separator or name in result:
            raise ValueError(f"invalid or duplicate --process value: {value}")
        try:
            pid = int(raw_pid)
        except ValueError as error:
            raise ValueError(f"process PID is not an integer: {value}") from error
        result[name] = pid
    if set(result) != set(SUPERVISOR_NAMES):
        raise ValueError("--process must name exactly the six capacity supervisors")
    return result


def main() -> int:
    args = parse_args()
    if args.plan_output.exists() or args.approval_output.exists():
        raise FileExistsError("migration plan or approval output already exists")
    target = args.project.resolve()
    formal = args.formal_project.resolve()
    plan = build_supervisor_migration_plan(
        selected_pids=_processes(args.process),
        target_project=target,
        expected_target_git={
            "path": target.as_posix(),
            "revision": args.expected_revision,
            "tree": args.expected_tree,
            "branch": args.expected_branch,
            "tracked_dirty": False,
        },
        formal_project=formal,
        expected_formal_git={
            "path": formal.as_posix(),
            "revision": args.expected_formal_revision,
            "tree": args.expected_formal_tree,
            "branch": args.expected_formal_branch,
            "tracked_dirty": False,
        },
        standing_authorization_path=args.standing_authorization,
        expected_standing_authorization_sha256=(
            args.expected_standing_authorization_sha256
        ),
    )
    write_json_report(args.plan_output, plan)
    plan_identity = file_identity(args.plan_output)
    approval = build_supervisor_migration_approval(
        plan_path=args.plan_output,
        expected_plan_sha256=plan_identity["sha256"],
    )
    write_json_report(args.approval_output, approval)
    print(
        json.dumps(
            {
                "status": "ready",
                "selected_process_names": list(SUPERVISOR_NAMES),
                "plan": plan_identity,
                "approval": file_identity(args.approval_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
