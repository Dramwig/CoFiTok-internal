from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.reporting import file_sha256

try:
    from scripts.build_generation_full_readiness import (
        _read_json,
        verify_readiness_report,
    )
except ModuleNotFoundError:
    from build_generation_full_readiness import _read_json, verify_readiness_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revalidate the immutable 250M stability-full readiness artifact."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--deployment-receipt", type=Path, required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--config-validation", type=Path, required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--training-run-dir", action="append", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--allow-later-formal-repository", action="store_true")
    parser.add_argument("--require-current-runtime-environment", action="store_true")
    parser.add_argument("--print-selected-runtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if file_sha256(args.readiness) != args.expected_readiness_sha256:
        raise ValueError("stability full readiness SHA256 differs")
    if len(args.training_run_dir) != 2 or len(set(args.training_run_dir)) != 2:
        raise ValueError("exactly two distinct matched training run directories are required")
    source_paths = {
        "deployment_receipt": args.deployment_receipt,
        "promotion_gate": args.promotion_gate,
        "cofitok_config": args.cofitok_config,
        "dense_config": args.dense_config,
        "config_validation": args.config_validation,
        "storage_capacity": args.storage_capacity,
        "runtime_selection": args.runtime_selection,
    }
    report = verify_readiness_report(
        _read_json(args.readiness),
        source_paths=source_paths,
        training_run_dirs=args.training_run_dir,
        benchmark_root=args.benchmark_root,
        storage_path=args.storage_path,
        project_root=Path(args.project_root),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        require_current_runtime_environment=(
            args.require_current_runtime_environment
        ),
        require_current_formal_repository=(
            not args.allow_later_formal_repository
        ),
        require_current_git=True,
    )
    selected = report["runtime_selection"]
    if args.print_selected_runtime:
        print(
            f"{selected['micro_batch_size']} "
            f"{selected['gradient_accumulation_steps']}"
        )
    else:
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
