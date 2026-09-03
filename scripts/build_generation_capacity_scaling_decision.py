from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_confirmation_result import (
    build_capacity_confirmation_result,
    validate_capacity_confirmation_result_contract,
)
from cofitok.generation.capacity_scaling_decision import (
    build_capacity_scaling_decision,
    validate_capacity_scaling_decision,
)
from cofitok.generation.capacity_screen import ARM_NAMES
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.reporting import file_sha256, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capacity-confirmation-result", type=Path, required=True)
    parser.add_argument(
        "--expected-capacity-confirmation-result-sha256", required=True
    )
    parser.add_argument("--expected-confirmation-revision", required=True)
    parser.add_argument("--expected-confirmation-tree", required=True)
    parser.add_argument("--expected-confirmation-branch", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)


def _expected_git(
    *, revision: str, tree: str, branch: str
) -> dict[str, Any]:
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def _read_identity_source(value: Any, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    path = Path(str(value["path"]))
    observed = identity(path)
    if observed != value:
        raise ValueError(f"{name} physical identity differs")
    return read_object(path, name=name), observed


def replay_capacity_confirmation_result(
    path: Path,
    *,
    expected_sha256: str,
    expected_confirmation_checkout: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    result_identity = identity(path)
    if result_identity["sha256"] != expected_sha256:
        raise ValueError("capacity confirmation result SHA256 differs")
    actual = read_object(path, name="capacity confirmation result")
    validate_capacity_confirmation_result_contract(actual)
    if actual.get("result_git") != expected_confirmation_checkout:
        raise ValueError("capacity confirmation result Git identity differs")
    sources = actual.get("source_evidence")
    if not isinstance(sources, dict):
        raise ValueError("capacity confirmation source evidence is missing")
    preparation, preparation_identity = _read_identity_source(
        sources.get("preparation"), "capacity confirmation preparation"
    )
    launch, launch_identity = _read_identity_source(
        sources.get("launch_receipt"), "capacity confirmation launch receipt"
    )
    arm_sources = sources.get("arm_validations")
    if not isinstance(arm_sources, dict) or set(arm_sources) != set(ARM_NAMES):
        raise ValueError("capacity confirmation arm source set differs")
    arm_reports: dict[str, dict[str, Any]] = {}
    arm_identities: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        arm_reports[arm], arm_identities[arm] = _read_identity_source(
            arm_sources[arm], f"{arm} capacity confirmation validation"
        )
    recomputed = build_capacity_confirmation_result(
        preparation=preparation,
        preparation_identity=preparation_identity,
        launch_receipt=launch,
        launch_receipt_identity=launch_identity,
        arm_validations=arm_reports,
        arm_validation_identities=arm_identities,
        result_git=expected_confirmation_checkout,
    )
    if actual != recomputed:
        raise ValueError("capacity confirmation result is not reproducible")
    return actual, result_identity, arm_reports, arm_identities


def build_from_sources(
    *,
    capacity_confirmation_result_path: Path,
    expected_capacity_confirmation_result_sha256: str,
    confirmation_checkout: dict[str, Any],
    decision_git: dict[str, Any],
) -> dict[str, Any]:
    result, result_identity, arm_reports, arm_identities = (
        replay_capacity_confirmation_result(
            capacity_confirmation_result_path,
            expected_sha256=expected_capacity_confirmation_result_sha256,
            expected_confirmation_checkout=confirmation_checkout,
        )
    )
    report = build_capacity_scaling_decision(
        capacity_confirmation_result=result,
        capacity_confirmation_result_identity=result_identity,
        arm_validations=arm_reports,
        arm_validation_identities=arm_identities,
        confirmation_checkout=confirmation_checkout,
        decision_git=decision_git,
    )
    validate_capacity_scaling_decision(
        report,
        expected_decision_revision=str(decision_git["revision"]),
        expected_decision_tree=str(decision_git["tree"]),
        expected_decision_branch=str(decision_git["branch"]),
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a physical-source-replayed, non-authorizing decision for "
            "preparing only the matched base256 step-10K to step-50K stage."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity scaling decision exists: {args.output}")
    decision_git = checkout_identity(PROJECT_ROOT)
    expected_decision_git = _expected_git(
        revision=args.expected_decision_revision,
        tree=args.expected_decision_tree,
        branch=args.expected_decision_branch,
    )
    if decision_git != expected_decision_git:
        raise ValueError("capacity scaling decision checkout identity differs")
    confirmation_git = _expected_git(
        revision=args.expected_confirmation_revision,
        tree=args.expected_confirmation_tree,
        branch=args.expected_confirmation_branch,
    )
    report = build_from_sources(
        capacity_confirmation_result_path=(
            args.capacity_confirmation_result.resolve()
        ),
        expected_capacity_confirmation_result_sha256=(
            args.expected_capacity_confirmation_result_sha256
        ),
        confirmation_checkout=confirmation_git,
        decision_git=decision_git,
    )
    write_json_report(args.output, report)
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": report["scientific_status"],
                "capacity_scaling_preparation_allowed": report["next_stage"][
                    "capacity_scaling_preparation_allowed"
                ],
                "decision_sha256": file_sha256(args.output),
                "training_launch_allowed": False,
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
