from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.configs import load_config
from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_CONFIGURED_STEPS,
    CAPACITY_PROBE_EFFECTIVE_BATCH,
)
from cofitok.generation.capacity_probe_execution import (
    build_capacity_probe_launch_receipt,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance, write_json_report
from scripts.select_generation_training_runtime import (
    _current_runtime_environment_sha,
    _selection_contract,
    validate_frozen_runtime_selection,
)
from scripts.validate_generation_configs import validate_pair


RUNTIME_CANDIDATES = ((1, 64), (2, 32), (4, 16), (8, 8), (16, 4))
RUNTIME_BASELINE = (1, 64)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--expected-execution-authorization-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--config-validation", type=Path, required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--training-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--require-current-runtime-environment", action="store_true")


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "preparation": args.preparation.resolve(),
        "execution_authorization": args.execution_authorization.resolve(),
        "cofitok_config": args.cofitok_config.resolve(),
        "dense_config": args.dense_config.resolve(),
        "config_validation": args.config_validation.resolve(),
        "storage_capacity": args.storage_capacity.resolve(),
        "runtime_selection": args.runtime_selection.resolve(),
    }


def _require_absent_training_state(run_dirs: list[Path]) -> None:
    for run_dir in run_dirs:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(
                f"refusing unreceipted capacity probe training state: {run_dir}"
            )


def build_from_args(
    args: argparse.Namespace,
    *,
    require_training_state_absent: bool,
    require_current_git: bool,
) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    expected_git = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if require_current_git and git_provenance(project_root) != expected_git:
        raise ValueError("capacity probe launch Git identity differs")
    sources = _source_paths(args)
    identities = {name: file_identity(path) for name, path in sources.items()}
    if identities["preparation"]["sha256"] != args.expected_preparation_sha256:
        raise ValueError("capacity probe preparation SHA256 differs")
    if (
        identities["execution_authorization"]["sha256"]
        != args.expected_execution_authorization_sha256
    ):
        raise ValueError("capacity probe execution authorization SHA256 differs")
    config_validation = read_json_object(
        sources["config_validation"],
        name="capacity probe config validation",
    )
    recomputed = validate_pair(
        load_config(sources["cofitok_config"]),
        load_config(sources["dense_config"]),
        max_parameter_gap=0.02,
        stage="stability_capacity_probe",
    )
    if config_validation != recomputed:
        raise ValueError("capacity probe config validation is not reproducible")

    run_dirs = [path.resolve() for path in args.training_run_dir]
    if len(run_dirs) != 2 or len(set(run_dirs)) != 2:
        raise ValueError("capacity probe requires two distinct training run directories")
    if require_training_state_absent:
        _require_absent_training_state(run_dirs)
    benchmark_root = args.benchmark_root.resolve()
    config_sha256 = {
        "cofitok": identities["cofitok_config"]["sha256"],
        "dense_identity": identities["dense_config"]["sha256"],
    }
    selection = read_json_object(
        sources["runtime_selection"],
        name="capacity probe runtime selection",
    )
    contract = _selection_contract(
        run_dirs=run_dirs,
        candidates=list(RUNTIME_CANDIDATES),
        baseline_candidate=RUNTIME_BASELINE,
        expected_effective_batch=CAPACITY_PROBE_EFFECTIVE_BATCH,
        benchmark_steps=8,
        warmup_steps=2,
        max_memory_fraction=0.9,
        target_steps=CAPACITY_PROBE_CONFIGURED_STEPS,
        revision=args.expected_revision,
        branch=args.expected_branch,
        config_sha256=config_sha256,
        benchmark_root=benchmark_root,
    )
    current_environment_sha = str(selection.get("runtime_environment_sha256", ""))
    if args.require_current_runtime_environment:
        current_environment_sha = _current_runtime_environment_sha(
            sources["cofitok_config"],
            project_root=project_root,
        )
    validate_frozen_runtime_selection(
        selection,
        expected_contract=contract,
        cofitok_config=sources["cofitok_config"],
        dense_config=sources["dense_config"],
        current_runtime_environment_sha256=current_environment_sha,
    )
    return build_capacity_probe_launch_receipt(
        preparation=read_json_object(
            sources["preparation"],
            name="capacity probe preparation",
        ),
        execution_authorization=read_json_object(
            sources["execution_authorization"],
            name="capacity probe execution authorization",
        ),
        config_validation=config_validation,
        storage_capacity=read_json_object(
            sources["storage_capacity"],
            name="capacity probe storage capacity",
        ),
        runtime_selection=selection,
        source_identities=identities,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        output_root=args.output_root.resolve().as_posix(),
        storage_path=args.storage_path.resolve().as_posix(),
        training_run_dirs=[path.as_posix() for path in run_dirs],
        benchmark_root=benchmark_root.as_posix(),
        training_state_absent_at_launch=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable 250M/10K capacity-probe launch receipt."
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity probe launch receipt exists: {args.output}")
    report = build_from_args(
        args,
        require_training_state_absent=True,
        require_current_git=True,
    )
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
