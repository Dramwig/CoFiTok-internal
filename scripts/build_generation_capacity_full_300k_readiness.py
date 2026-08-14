from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from cofitok.generation.capacity_full_readiness import (
    build_capacity_full_readiness,
    validate_capacity_full_readiness,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report

try:
    from scripts.build_generation_full_readiness import (
        validate_full_config_contract,
        validate_full_runtime_selection,
    )
except ModuleNotFoundError:
    from build_generation_full_readiness import (
        validate_full_config_contract,
        validate_full_runtime_selection,
    )


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
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--config-validation", type=Path, required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--cofitok-run-dir", type=Path, required=True)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--expected-readiness-revision", required=True)
    parser.add_argument("--expected-readiness-tree", required=True)
    parser.add_argument("--expected-readiness-branch", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--expected-result-revision", required=True)
    parser.add_argument("--expected-result-tree", required=True)
    parser.add_argument("--expected-result-branch", required=True)


def _require_absent(paths: dict[str, Path]) -> None:
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


def build_from_sources(args: argparse.Namespace) -> dict[str, Any]:
    readiness_git = _git_identity(PROJECT_ROOT)
    if readiness_git != {
        "revision": args.expected_readiness_revision,
        "tree": args.expected_readiness_tree,
        "branch": args.expected_readiness_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity-full readiness checkout identity differs")
    decision_identity = file_identity(args.decision)
    if decision_identity["sha256"] != args.expected_decision_sha256:
        raise ValueError("capacity-full readiness decision SHA256 differs")
    decision = read_json_object(args.decision, name="capacity-full readiness decision")
    for name, expected in decision.get("source_reports", {}).items():
        if file_identity(expected["path"]) != expected:
            raise ValueError(f"capacity-full readiness decision source changed: {name}")
    config_paths = {
        "cofitok": args.cofitok_config.resolve(),
        "dense_identity": args.dense_config.resolve(),
    }
    config_identities = {
        method: file_identity(path) for method, path in config_paths.items()
    }
    config_validation_identity = file_identity(args.config_validation)
    config_validation = read_json_object(
        args.config_validation,
        name="capacity-full config validation",
    )
    validate_full_config_contract(
        config_validation,
        cofitok_config_path=config_paths["cofitok"],
        dense_config_path=config_paths["dense_identity"],
    )
    storage_identity = file_identity(args.storage_capacity)
    storage = read_json_object(
        args.storage_capacity,
        name="capacity-full storage capacity",
    )
    runtime_identity = file_identity(args.runtime_selection)
    runtime = read_json_object(
        args.runtime_selection,
        name="capacity-full runtime selection",
    )
    run_dirs = {
        "cofitok": args.cofitok_run_dir.resolve(),
        "dense_identity": args.dense_run_dir.resolve(),
    }
    _require_absent(run_dirs)
    runtime_evidence = validate_full_runtime_selection(
        runtime,
        cofitok_config_path=config_paths["cofitok"],
        dense_config_path=config_paths["dense_identity"],
        training_run_dirs=list(run_dirs.values()),
        benchmark_root=args.benchmark_root.resolve(),
        expected_revision=args.expected_readiness_revision,
        expected_branch=args.expected_readiness_branch,
        project_root=PROJECT_ROOT,
        require_current_runtime_environment=True,
    )
    report = build_capacity_full_readiness(
        readiness_decision=decision,
        readiness_decision_identity=decision_identity,
        target_config_identities=config_identities,
        config_validation=config_validation,
        config_validation_identity=config_validation_identity,
        storage_report=storage,
        storage_report_identity=storage_identity,
        runtime_selection=runtime,
        runtime_selection_identity=runtime_identity,
        runtime_evidence=runtime_evidence,
        readiness_git=readiness_git,
        expected_readiness_revision=args.expected_readiness_revision,
        expected_readiness_tree=args.expected_readiness_tree,
        expected_readiness_branch=args.expected_readiness_branch,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
    )
    validate_capacity_full_readiness(
        report,
        expected_readiness_revision=args.expected_readiness_revision,
        expected_readiness_tree=args.expected_readiness_tree,
        expected_readiness_branch=args.expected_readiness_branch,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact fresh-300K readiness artifact after the selected "
            "capacity result and bounded 250M runtime benchmark."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity-full readiness exists: {args.output}")
    report = build_from_sources(args)
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
