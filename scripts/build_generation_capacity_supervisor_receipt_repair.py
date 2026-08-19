from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation_capacity_supervisor_repair import (
    build_supervisor_receipt_repair_plan,
)
from cofitok.reporting import write_json_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the source-bound capacity supervisor receipt repair plan."
    )
    parser.add_argument("--migration-plan", type=Path, required=True)
    parser.add_argument("--expected-migration-plan-sha256", required=True)
    parser.add_argument("--migration-approval", type=Path, required=True)
    parser.add_argument("--expected-migration-approval-sha256", required=True)
    parser.add_argument("--migration-execution", type=Path, required=True)
    parser.add_argument("--expected-migration-execution-sha256", required=True)
    parser.add_argument("--target-project", type=Path, required=True)
    parser.add_argument("--expected-target-revision", required=True)
    parser.add_argument("--expected-target-tree", required=True)
    parser.add_argument("--expected-target-branch", required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--expected-formal-revision", required=True)
    parser.add_argument("--expected-formal-tree", required=True)
    parser.add_argument("--expected-formal-branch", required=True)
    parser.add_argument("--live-training-pid", type=int, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--posteval-receipt", type=Path, required=True)
    parser.add_argument("--expected-posteval-receipt-sha256", required=True)
    parser.add_argument("--finalization-receipt", type=Path, required=True)
    parser.add_argument("--expected-finalization-receipt-sha256", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(path: Path, revision: str, tree: str, branch: str) -> dict:
    return {
        "path": path.resolve().as_posix(),
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def main() -> int:
    args = _parse_args()
    report = build_supervisor_receipt_repair_plan(
        migration_plan_path=args.migration_plan,
        expected_migration_plan_sha256=args.expected_migration_plan_sha256,
        migration_approval_path=args.migration_approval,
        expected_migration_approval_sha256=args.expected_migration_approval_sha256,
        migration_execution_path=args.migration_execution,
        expected_migration_execution_sha256=args.expected_migration_execution_sha256,
        target_project=args.target_project,
        expected_target_git=_git(
            args.target_project,
            args.expected_target_revision,
            args.expected_target_tree,
            args.expected_target_branch,
        ),
        formal_project=args.formal_project,
        expected_formal_git=_git(
            args.formal_project,
            args.expected_formal_revision,
            args.expected_formal_tree,
            args.expected_formal_branch,
        ),
        live_training_pid=args.live_training_pid,
        training_receipt_path=args.training_receipt,
        expected_training_receipt_sha256=args.expected_training_receipt_sha256,
        posteval_receipt_path=args.posteval_receipt,
        expected_posteval_receipt_sha256=args.expected_posteval_receipt_sha256,
        finalization_receipt_path=args.finalization_receipt,
        expected_finalization_receipt_sha256=(
            args.expected_finalization_receipt_sha256
        ),
        archive_root=args.archive_root,
    )
    write_json_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
