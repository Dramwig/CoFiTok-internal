from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from cofitok.generation.capacity_completion_execution import (
    build_capacity_completion_launch_receipt,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--capacity-scaling-result", type=Path, required=True)
    parser.add_argument("--expected-capacity-scaling-result-sha256", required=True)
    parser.add_argument(
        "--capacity-scaling-launch-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--expected-capacity-scaling-launch-receipt-sha256", required=True
    )
    parser.add_argument("--source-checkpoint-archive", type=Path, required=True)
    parser.add_argument("--expected-source-checkpoint-archive-sha256", required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--expected-storage-capacity-sha256", required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--storage-path", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--expected-scaling-execution-revision", required=True)
    parser.add_argument("--expected-scaling-execution-tree", required=True)
    parser.add_argument("--expected-scaling-execution-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)


def _read_exact(path: Path, *, digest: str, name: str) -> tuple[dict, dict]:
    identity = file_identity(path)
    if identity["sha256"] != digest:
        raise ValueError(f"{name} SHA256 differs")
    return read_json_object(path, name=name), identity


def build_from_sources(args: argparse.Namespace) -> dict[str, Any]:
    execution_git = _git_identity(PROJECT_ROOT)
    if execution_git != {
        "revision": args.expected_execution_revision,
        "tree": args.expected_execution_tree,
        "branch": args.expected_execution_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity completion execution checkout identity differs")
    training_git = _git_identity(args.training_project.resolve())
    if training_git != {
        "revision": args.expected_training_revision,
        "tree": args.expected_training_tree,
        "branch": args.expected_training_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity completion training checkout identity differs")
    decision, decision_identity = _read_exact(
        args.decision.resolve(),
        digest=args.expected_decision_sha256,
        name="capacity completion decision",
    )
    result, result_identity = _read_exact(
        args.capacity_scaling_result.resolve(),
        digest=args.expected_capacity_scaling_result_sha256,
        name="capacity scaling 50K result",
    )
    scaling_launch, scaling_launch_identity = _read_exact(
        args.capacity_scaling_launch_receipt.resolve(),
        digest=args.expected_capacity_scaling_launch_receipt_sha256,
        name="capacity scaling 50K launch receipt",
    )
    archive, archive_identity = _read_exact(
        args.source_checkpoint_archive.resolve(),
        digest=args.expected_source_checkpoint_archive_sha256,
        name="capacity completion source archive",
    )
    storage, storage_identity = _read_exact(
        args.storage_capacity.resolve(),
        digest=args.expected_storage_capacity_sha256,
        name="capacity completion storage capacity",
    )
    cofitok_config = file_identity(
        scaling_launch["source_reports"]["cofitok_config"]["path"]
    )
    dense_config = file_identity(
        scaling_launch["source_reports"]["dense_config"]["path"]
    )
    return build_capacity_completion_launch_receipt(
        decision=decision,
        capacity_scaling_result=result,
        capacity_scaling_launch_receipt=scaling_launch,
        source_checkpoint_archive=archive,
        storage_capacity=storage,
        source_identities={
            "capacity_completion_decision": decision_identity,
            "capacity_scaling_50k_result": result_identity,
            "capacity_scaling_50k_launch_receipt": scaling_launch_identity,
            "source_checkpoint_archive": archive_identity,
            "cofitok_config": cofitok_config,
            "dense_config": dense_config,
            "storage_capacity": storage_identity,
        },
        execution_git=execution_git,
        training_git=training_git,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        expected_scaling_execution_revision=(
            args.expected_scaling_execution_revision
        ),
        expected_scaling_execution_tree=args.expected_scaling_execution_tree,
        expected_scaling_execution_branch=args.expected_scaling_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        output_root=args.output_root,
        storage_path=args.storage_path,
        training_project=args.training_project.resolve().as_posix(),
        gpu_idle_at_launch=True,
        relevant_processes_absent_at_launch=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound launch receipt for the exact matched 250M "
            "step-50K to step-100K capacity-completion segment."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity completion launch receipt exists: {args.output}")
    report = build_from_sources(args)
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
