from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.build_generation_full_readiness import (
        _read_json,
        FULL_COMPLETION_SAMPLE_RESERVE,
        require_absent_training_state,
        source_identities,
        validate_full_storage_capacity,
    )
    from scripts.build_generation_full_readiness_bridge import (
        verify_readiness_bridge,
    )
    from scripts.verify_generation_stability_frozen_supplemental import (
        verify_frozen_supplemental_report,
    )
except ModuleNotFoundError:
    from build_generation_full_readiness import (
        _read_json,
        FULL_COMPLETION_SAMPLE_RESERVE,
        require_absent_training_state,
        source_identities,
        validate_full_storage_capacity,
    )
    from build_generation_full_readiness_bridge import verify_readiness_bridge
    from verify_generation_stability_frozen_supplemental import (
        verify_frozen_supplemental_report,
    )


SCHEMA_VERSION = 3
ROLE = "stability_full_training_launch_receipt"


def build_full_launch_receipt(
    *,
    source_paths: dict[str, Path],
    training_run_dirs: list[Path],
    benchmark_root: Path,
    storage_path: Path,
    project_root: Path,
    expected_revision: str,
    expected_branch: str,
    expected_readiness_sha256: str,
    expected_stability_supplemental_sha256: str,
    require_current_runtime_environment: bool,
    require_current_formal_repository: bool = True,
    require_current_git: bool = True,
    require_training_state_absent: bool = False,
) -> dict[str, Any]:
    required_sources = {
        "deployment_receipt",
        "promotion_gate",
        "stability_supplemental",
        "full_readiness",
        "readiness_bridge",
        "cofitok_config",
        "dense_config",
        "config_validation",
        "storage_capacity",
        "runtime_selection",
        "launch_storage_capacity",
    }
    if set(source_paths) != required_sources:
        raise ValueError("stability full launch receipt source set differs")
    if len(training_run_dirs) != 2 or len(set(training_run_dirs)) != 2:
        raise ValueError("exactly two distinct matched training run directories are required")
    if file_sha256(source_paths["full_readiness"]) != expected_readiness_sha256:
        raise ValueError("stability full readiness SHA256 differs")
    if require_training_state_absent:
        require_absent_training_state(training_run_dirs)

    stability_supplemental = verify_frozen_supplemental_report(
        _read_json(source_paths["stability_supplemental"]),
        report_path=source_paths["stability_supplemental"],
        expected_report_sha256=expected_stability_supplemental_sha256,
        promotion_gate_path=source_paths["promotion_gate"],
    )

    bridge_report = _read_json(source_paths["readiness_bridge"])
    source_git = bridge_report.get("source_git", {})
    target_git = bridge_report.get("target_git", {})
    source_deployment = bridge_report.get("source_deployment_receipt", {})
    bridge = verify_readiness_bridge(
        bridge_report,
        bridge_path=source_paths["readiness_bridge"],
        expected_bridge_sha256=file_sha256(source_paths["readiness_bridge"]),
        project_root=project_root,
        readiness_path=source_paths["full_readiness"],
        expected_readiness_sha256=expected_readiness_sha256,
        source_deployment_receipt=Path(str(source_deployment.get("path", ""))),
        expected_source_deployment_receipt_sha256=str(
            source_deployment.get("sha256", "")
        ),
        target_deployment_receipt=source_paths["deployment_receipt"],
        expected_target_deployment_receipt_sha256=file_sha256(
            source_paths["deployment_receipt"]
        ),
        expected_source_revision=str(source_git.get("revision", "")),
        expected_source_branch=str(source_git.get("branch", "")),
        expected_target_revision=expected_revision,
        expected_target_branch=expected_branch,
        require_current_target_git=require_current_git,
    )
    launch_storage = validate_full_storage_capacity(
        _read_json(source_paths["launch_storage_capacity"]),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_path=storage_path,
        minimum_sample_count=FULL_COMPLETION_SAMPLE_RESERVE,
    )
    sources = source_identities(source_paths)
    if (
        stability_supplemental.get("report")
        != sources["stability_supplemental"]
    ):
        raise ValueError("stability supplemental launch binding differs")
    selected_runtime = bridge["runtime_selection"]
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if target_git != expected_git:
        raise ValueError("stability full readiness bridge target identity differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "role": ROLE,
        "stage": "stability_full",
        "git": expected_git,
        "source_reports": sources,
        "readiness_sha256": expected_readiness_sha256,
        "quality_prerequisites": {
            "frozen_stability_supplemental": stability_supplemental,
        },
        "readiness_bridge": {
            "sha256": sources["readiness_bridge"]["sha256"],
            "source_git": bridge["source_git"],
            "target_git": bridge["target_git"],
            "training_semantics_identical": bridge[
                "training_semantics_identical"
            ],
            "controlled_preamble_upgrade": bridge["full_training_runbook"][
                "controlled_preamble_upgrade"
            ],
            "source_preamble_sha256": bridge["full_training_runbook"][
                "source_preamble_sha256"
            ],
            "normalized_target_preamble_sha256": bridge[
                "full_training_runbook"
            ]["normalized_target_preamble_sha256"],
            "training_execution_sha256": bridge["full_training_runbook"][
                "training_execution_sha256"
            ],
        },
        "promotion_authorization": bridge["promotion_authorization"],
        "deployment": {
            "receipt_sha256": sources["deployment_receipt"]["sha256"],
            "checkout_git": expected_git,
        },
        "runtime_selection": selected_runtime,
        "launch_storage_capacity": launch_storage,
        "training_run_dirs": [
            path.resolve().as_posix() for path in training_run_dirs
        ],
        "benchmark_root": benchmark_root.resolve().as_posix(),
        "storage_path": storage_path.resolve().as_posix(),
        "training_state_absent_at_launch": True,
        "full_training_launch_authorized": True,
        "formal_generation_completion_claimed": False,
    }


def verify_full_launch_receipt(
    report: dict[str, Any],
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    if file_sha256(receipt_path) != expected_receipt_sha256:
        raise ValueError("stability full launch receipt SHA256 differs")
    expected = build_full_launch_receipt(**kwargs)
    if report != expected:
        raise ValueError("stability full launch receipt is not reproducible")
    return expected


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--deployment-receipt", type=Path, required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--stability-supplemental", type=Path, required=True)
    parser.add_argument(
        "--expected-stability-supplemental-sha256",
        required=True,
    )
    parser.add_argument("--full-readiness", type=Path, required=True)
    parser.add_argument("--readiness-bridge", type=Path, required=True)
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--config-validation", type=Path, required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--launch-storage-capacity", type=Path, required=True)
    parser.add_argument("--training-run-dir", action="append", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--require-current-runtime-environment", action="store_true")


def source_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "deployment_receipt": args.deployment_receipt,
        "promotion_gate": args.promotion_gate,
        "stability_supplemental": args.stability_supplemental,
        "full_readiness": args.full_readiness,
        "readiness_bridge": args.readiness_bridge,
        "cofitok_config": args.cofitok_config,
        "dense_config": args.dense_config,
        "config_validation": args.config_validation,
        "storage_capacity": args.storage_capacity,
        "runtime_selection": args.runtime_selection,
        "launch_storage_capacity": args.launch_storage_capacity,
    }


def build_kwargs_from_args(
    args: argparse.Namespace,
    *,
    require_current_git: bool,
    require_current_formal_repository: bool,
    require_training_state_absent: bool,
) -> dict[str, Any]:
    return {
        "source_paths": source_paths_from_args(args),
        "training_run_dirs": args.training_run_dir,
        "benchmark_root": args.benchmark_root,
        "storage_path": args.storage_path,
        "project_root": Path(args.project_root),
        "expected_revision": args.expected_revision,
        "expected_branch": args.expected_branch,
        "expected_readiness_sha256": args.expected_readiness_sha256,
        "expected_stability_supplemental_sha256": (
            args.expected_stability_supplemental_sha256
        ),
        "require_current_runtime_environment": (
            args.require_current_runtime_environment
        ),
        "require_current_formal_repository": require_current_formal_repository,
        "require_current_git": require_current_git,
        "require_training_state_absent": require_training_state_absent,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an immutable stability-full 300K launch receipt."
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"full launch receipt already exists: {args.output}")
    report = build_full_launch_receipt(
        **build_kwargs_from_args(
            args,
            require_current_git=True,
            require_current_formal_repository=True,
            require_training_state_absent=True,
        )
    )
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
