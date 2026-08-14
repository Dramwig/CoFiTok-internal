from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_full_training_launch import (
    build_capacity_full_training_launch,
    validate_capacity_full_training_launch,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git(project: Path) -> dict[str, Any]:
    def run(*arguments: str, binary: bool = False) -> str | bytes:
        result = subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
        )
        return result.stdout

    status = run("status", "--porcelain=v1", "--untracked-files=no")
    return {
        "revision": str(run("rev-parse", "HEAD")).strip(),
        "tree": str(run("rev-parse", "HEAD^{tree}")).strip(),
        "branch": str(run("branch", "--show-current")).strip(),
        "tracked_dirty": bool(str(status).strip()),
    }


def _identity_matches(identity: dict[str, Any], *, label: str) -> None:
    if file_identity(identity["path"]) != identity:
        raise ValueError(f"capacity-full training source changed: {label}")


def _require_source_replay(report: dict[str, Any], *, label: str) -> None:
    sources = report.get("source_reports")
    if not isinstance(sources, dict):
        raise ValueError(f"{label} source report set is missing")
    for name, identity in sources.items():
        if not isinstance(identity, dict):
            raise ValueError(f"{label} source identity is malformed: {name}")
        _identity_matches(identity, label=f"{label}.{name}")


def _formal_snapshot(project: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "-C", str(project), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "path": project.resolve().as_posix(),
        "head": head,
        "porcelain_count": len(porcelain.splitlines()),
        "porcelain_sha256": hashlib.sha256(porcelain).hexdigest(),
    }


def _transition(
    training_project: Path,
    *,
    source_revision: str,
    target_revision: str,
) -> dict[str, Any]:
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(training_project),
            "merge-base",
            "--is-ancestor",
            source_revision,
            target_revision,
        ],
        check=False,
    ).returncode == 0
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(training_project),
            "diff",
            "--name-only",
            source_revision,
            target_revision,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "source_revision": source_revision,
        "target_revision": target_revision,
        "source_is_ancestor": ancestor,
        "changed_paths": sorted(changed),
        "target_configs_byte_identical": True,
        "model_data_diffusion_compute_paths_changed": [],
        "runtime_requalification_required": False,
        "authorization_metadata_only_runtime_change": True,
    }


def _require_training_state_absent(paths: dict[str, Path]) -> None:
    for method, path in paths.items():
        if path.exists() and not path.is_dir():
            raise ValueError(f"capacity-full {method} run path is not a directory")
        if path.is_dir():
            entries = sorted(path.iterdir())
            if entries:
                preview = ", ".join(entry.name for entry in entries[:20])
                raise ValueError(
                    f"capacity-full {method} training state exists: {preview}"
                )


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution-project", type=Path, required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--readiness-supervisor-status", type=Path, required=True)
    parser.add_argument("--readiness-deployment-receipt", type=Path, required=True)
    parser.add_argument("--expected-readiness-deployment-sha256", required=True)
    parser.add_argument("--readiness-deployment-clarification", type=Path, required=True)
    parser.add_argument("--expected-readiness-clarification-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--launch-storage-capacity", type=Path, required=True)
    parser.add_argument("--cofitok-run-dir", type=Path, required=True)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-readiness-revision", required=True)
    parser.add_argument("--expected-readiness-tree", required=True)
    parser.add_argument("--expected-readiness-branch", required=True)


def build_from_sources(
    args: argparse.Namespace,
    *,
    require_training_state_absent: bool,
) -> dict[str, Any]:
    execution_project = args.execution_project.resolve()
    training_project = args.training_project.resolve()
    formal_project = args.formal_project.resolve()
    execution_git = _git(execution_project)
    training_git = _git(training_project)
    expected_execution_git = {
        "revision": args.expected_execution_revision,
        "tree": args.expected_execution_tree,
        "branch": args.expected_execution_branch,
        "tracked_dirty": False,
    }
    expected_training_git = {
        "revision": args.expected_training_revision,
        "tree": args.expected_training_tree,
        "branch": args.expected_training_branch,
        "tracked_dirty": False,
    }
    if execution_git != expected_execution_git:
        raise ValueError("capacity-full execution checkout identity differs")
    if training_git != expected_training_git:
        raise ValueError("capacity-full training checkout identity differs")

    readiness = read_json_object(args.readiness, name="capacity-full readiness")
    _require_source_replay(readiness, label="capacity-full readiness")
    status = read_json_object(
        args.readiness_supervisor_status,
        name="capacity-full readiness supervisor status",
    )
    deployment_identity = file_identity(args.readiness_deployment_receipt)
    if deployment_identity["sha256"] != args.expected_readiness_deployment_sha256:
        raise ValueError("capacity-full readiness deployment SHA256 differs")
    deployment = read_json_object(
        args.readiness_deployment_receipt,
        name="capacity-full readiness deployment receipt",
    )
    clarification_identity = file_identity(args.readiness_deployment_clarification)
    if clarification_identity["sha256"] != args.expected_readiness_clarification_sha256:
        raise ValueError("capacity-full readiness clarification SHA256 differs")
    clarification = read_json_object(
        args.readiness_deployment_clarification,
        name="capacity-full readiness deployment clarification",
    )
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    standing = read_json_object(
        args.standing_authorization,
        name="standing experiment authorization",
    )
    if _formal_snapshot(formal_project) != deployment.get("formal_checkout_snapshot"):
        raise ValueError("formal remote checkout changed since readiness deployment")

    run_dirs = {
        "cofitok": args.cofitok_run_dir.resolve(),
        "dense_identity": args.dense_run_dir.resolve(),
    }
    if require_training_state_absent:
        _require_training_state_absent(run_dirs)
    report = build_capacity_full_training_launch(
        readiness=readiness,
        readiness_identity=file_identity(args.readiness),
        readiness_supervisor_status=status,
        readiness_supervisor_status_identity=file_identity(
            args.readiness_supervisor_status
        ),
        readiness_deployment_receipt=deployment,
        readiness_deployment_receipt_identity=deployment_identity,
        readiness_deployment_clarification=clarification,
        readiness_deployment_clarification_identity=clarification_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        target_config_identities={
            "cofitok": file_identity(args.cofitok_config),
            "dense_identity": file_identity(args.dense_config),
        },
        launch_storage=read_json_object(
            args.launch_storage_capacity,
            name="capacity-full launch storage capacity",
        ),
        launch_storage_identity=file_identity(args.launch_storage_capacity),
        readiness_to_training_transition=_transition(
            training_project,
            source_revision=args.expected_readiness_revision,
            target_revision=args.expected_training_revision,
        ),
        execution_git=execution_git,
        training_git=training_git,
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_readiness_revision=args.expected_readiness_revision,
        expected_readiness_tree=args.expected_readiness_tree,
        expected_readiness_branch=args.expected_readiness_branch,
    )
    validate_capacity_full_training_launch(
        report,
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_readiness_revision=args.expected_readiness_revision,
        expected_readiness_tree=args.expected_readiness_tree,
        expected_readiness_branch=args.expected_readiness_branch,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the immutable experimental capacity-full fresh matched 300K "
            "training authorization receipt."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity-full training receipt exists: {args.output}")
    report = build_from_sources(args, require_training_state_absent=True)
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
