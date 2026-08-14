from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from cofitok.configs import load_config
from cofitok.generation.capacity_full_readiness_decision import (
    build_capacity_full_readiness_decision,
    validate_capacity_full_readiness_decision,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import to_jsonable, write_json_report
from scripts.validate_generation_configs import validate_pair


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
    parser.add_argument("--capacity-completion-result", type=Path, required=True)
    parser.add_argument("--result-waiter-status", type=Path, required=True)
    parser.add_argument(
        "--result-waiter-deployment-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-result-waiter-deployment-receipt-sha256",
        required=True,
    )
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--target-cofitok-config", type=Path, required=True)
    parser.add_argument("--target-dense-config", type=Path, required=True)
    parser.add_argument("--full-output-root", type=Path, required=True)
    parser.add_argument("--cofitok-run-dir", type=Path, required=True)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--expected-result-execution-revision", required=True)
    parser.add_argument("--expected-result-execution-tree", required=True)
    parser.add_argument("--expected-result-execution-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-result-revision", required=True)
    parser.add_argument("--expected-result-tree", required=True)
    parser.add_argument("--expected-result-branch", required=True)


def _verify_identity(path: Path, expected: dict[str, Any], *, label: str) -> None:
    if file_identity(path) != expected:
        raise ValueError(f"capacity-full source changed: {label}")


def _require_training_state_absent(paths: dict[str, Path]) -> None:
    for method, path in paths.items():
        if path.exists() and not path.is_dir():
            raise ValueError(f"capacity-full {method} run path is not a directory")
        if path.is_dir():
            entries = sorted(path.iterdir())
            if entries:
                preview = ", ".join(item.name for item in entries[:20])
                raise ValueError(
                    f"capacity-full {method} training state exists: {preview}"
                )


def _pair_validation(
    cofitok_path: Path,
    dense_path: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    return to_jsonable(
        validate_pair(
            load_config(cofitok_path),
            load_config(dense_path),
            max_parameter_gap=0.02,
            stage=stage,
        )
    )


def build_from_sources(args: argparse.Namespace) -> dict[str, Any]:
    decision_git = _git_identity(PROJECT_ROOT)
    if decision_git != {
        "revision": args.expected_decision_revision,
        "tree": args.expected_decision_tree,
        "branch": args.expected_decision_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity-full decision checkout identity differs")
    result_identity = file_identity(args.capacity_completion_result)
    result = read_json_object(
        args.capacity_completion_result,
        name="capacity completion 100K result",
    )
    result_sources = result.get("source_reports")
    if not isinstance(result_sources, dict):
        raise ValueError("capacity completion result source set is missing")
    for name, identity in result_sources.items():
        if not isinstance(identity, dict):
            raise ValueError(f"capacity completion result source is malformed: {name}")
        _verify_identity(Path(identity["path"]), identity, label=name)

    waiter_status_identity = file_identity(args.result_waiter_status)
    waiter_status = read_json_object(
        args.result_waiter_status,
        name="capacity completion result waiter status",
    )
    deployment_identity = file_identity(args.result_waiter_deployment_receipt)
    if (
        deployment_identity["sha256"]
        != args.expected_result_waiter_deployment_receipt_sha256
    ):
        raise ValueError("capacity completion result waiter deployment SHA256 differs")
    deployment = read_json_object(
        args.result_waiter_deployment_receipt,
        name="capacity completion result waiter deployment",
    )
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    standing = read_json_object(
        args.standing_authorization,
        name="standing experiment authorization",
    )

    launch_source = result_sources.get("capacity_completion_launch_receipt")
    if not isinstance(launch_source, dict):
        raise ValueError("capacity completion launch source is missing")
    launch_path = Path(launch_source["path"])
    _verify_identity(
        launch_path,
        launch_source,
        label="capacity_completion_launch_receipt",
    )
    launch = read_json_object(
        launch_path,
        name="capacity completion launch receipt",
    )
    launch_sources = launch.get("source_reports")
    if not isinstance(launch_sources, dict):
        raise ValueError("capacity completion launch source reports are missing")
    source_config_paths = {
        "cofitok": Path(launch_sources["cofitok_config"]["path"]),
        "dense_identity": Path(launch_sources["dense_config"]["path"]),
    }
    source_config_identities = {
        "cofitok": file_identity(source_config_paths["cofitok"]),
        "dense_identity": file_identity(source_config_paths["dense_identity"]),
    }
    if (
        source_config_identities["cofitok"] != launch_sources["cofitok_config"]
        or source_config_identities["dense_identity"]
        != launch_sources["dense_config"]
    ):
        raise ValueError("capacity completion source config identity differs")
    target_config_paths = {
        "cofitok": args.target_cofitok_config.resolve(),
        "dense_identity": args.target_dense_config.resolve(),
    }
    target_config_identities = {
        method: file_identity(path)
        for method, path in target_config_paths.items()
    }
    source_configs = {
        method: read_json_object(path, name=f"source {method} config")
        for method, path in source_config_paths.items()
    }
    target_configs = {
        method: read_json_object(path, name=f"target {method} config")
        for method, path in target_config_paths.items()
    }
    source_validation = _pair_validation(
        source_config_paths["cofitok"],
        source_config_paths["dense_identity"],
        stage="stability_capacity_probe",
    )
    target_validation = _pair_validation(
        target_config_paths["cofitok"],
        target_config_paths["dense_identity"],
        stage="stability_full",
    )
    run_dirs = {
        "cofitok": args.cofitok_run_dir.resolve(),
        "dense_identity": args.dense_run_dir.resolve(),
    }
    _require_training_state_absent(run_dirs)
    report = build_capacity_full_readiness_decision(
        capacity_completion_result=result,
        capacity_completion_result_identity=result_identity,
        result_waiter_status=waiter_status,
        result_waiter_status_identity=waiter_status_identity,
        result_waiter_deployment_receipt=deployment,
        result_waiter_deployment_receipt_identity=deployment_identity,
        capacity_completion_launch_receipt=launch,
        capacity_completion_launch_receipt_identity=launch_source,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        source_configs=source_configs,
        source_config_identities=source_config_identities,
        target_configs=target_configs,
        target_config_identities=target_config_identities,
        source_pair_validation=source_validation,
        target_pair_validation=target_validation,
        training_run_dirs={
            method: path.as_posix() for method, path in run_dirs.items()
        },
        training_state_absent=True,
        full_output_root=args.full_output_root.resolve().as_posix(),
        benchmark_root=args.benchmark_root.resolve().as_posix(),
        storage_path=args.storage_path.resolve().as_posix(),
        decision_builder_git=decision_git,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        expected_result_execution_revision=(
            args.expected_result_execution_revision
        ),
        expected_result_execution_tree=args.expected_result_execution_tree,
        expected_result_execution_branch=args.expected_result_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
    )
    validate_capacity_full_readiness_decision(
        report,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-replayed decision that can authorize only the "
            "fresh 250M full-300K runtime/readiness benchmark."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"capacity-full readiness decision exists: {args.output}"
        )
    report = build_from_sources(args)
    write_json_report(args.output, report)
    print(report["execution_authorization"])


if __name__ == "__main__":
    main()
