from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_ARM_NAMES,
    CAPACITY_PROBE_RESULT_SOURCE_NAMES,
)
from cofitok.generation.capacity_scaling_decision import (
    build_capacity_scaling_decision,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance, write_json_report

try:
    from scripts.build_generation_capacity_probe_result import build_from_args
except ModuleNotFoundError:
    from build_generation_capacity_probe_result import build_from_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capacity-probe-result", type=Path, required=True)
    parser.add_argument("--expected-capacity-probe-result-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--expected-capacity-revision", required=True)
    parser.add_argument("--expected-capacity-branch", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-branch", required=True)


def _result_replay_args(
    result: dict[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> SimpleNamespace:
    sources = result.get("source_reports")
    if not isinstance(sources, dict) or set(sources) != (
        CAPACITY_PROBE_RESULT_SOURCE_NAMES
    ):
        raise ValueError("capacity probe result source set differs")
    values: dict[str, Any] = {
        "preparation": Path(str(sources["preparation"]["path"])),
        "launch_receipt": Path(str(sources["launch_receipt"]["path"])),
        "base256_cofitok_training_validation": Path(
            str(sources["base256_cofitok_training_validation"]["path"])
        ),
        "base256_dense_training_validation": Path(
            str(sources["base256_dense_identity_training_validation"]["path"])
        ),
        "expected_preparation_sha256": sources["preparation"]["sha256"],
        "expected_launch_receipt_sha256": sources["launch_receipt"]["sha256"],
        "expected_revision": expected_revision,
        "expected_branch": expected_branch,
    }
    for arm in CAPACITY_PROBE_ARM_NAMES:
        for kind in ("sampling_preflight", "generation", "checkpoint_eval"):
            name = f"{arm}_{kind}"
            values[name] = Path(str(sources[name]["path"]))
    return SimpleNamespace(**values)


def replay_capacity_probe_result(
    path: Path,
    *,
    expected_sha256: str,
    expected_revision: str,
    expected_branch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("capacity probe result SHA256 differs")
    actual = read_json_object(path, name="capacity probe result")
    replay_args = _result_replay_args(
        actual,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    recomputed = build_from_args(replay_args)
    if actual != recomputed:
        raise ValueError("capacity probe result is not reproducible")
    return actual, identity


def build_from_sources(
    *,
    capacity_probe_result_path: Path,
    expected_capacity_probe_result_sha256: str,
    standing_authorization_path: Path,
    expected_standing_authorization_sha256: str,
    decision_git: dict[str, Any],
    expected_capacity_revision: str,
    expected_capacity_branch: str,
) -> dict[str, Any]:
    result, result_identity = replay_capacity_probe_result(
        capacity_probe_result_path,
        expected_sha256=expected_capacity_probe_result_sha256,
        expected_revision=expected_capacity_revision,
        expected_branch=expected_capacity_branch,
    )
    standing_identity = file_identity(standing_authorization_path)
    if standing_identity["sha256"] != expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    standing = read_json_object(
        standing_authorization_path,
        name="standing experiment authorization",
    )
    return build_capacity_scaling_decision(
        capacity_probe_result=result,
        capacity_probe_result_identity=result_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        decision_git=decision_git,
        expected_capacity_revision=expected_capacity_revision,
        expected_capacity_branch=expected_capacity_branch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a physical-source-replayed decision for only the matched "
            "250M step-10K to step-50K capacity-scaling segment."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity scaling decision exists: {args.output}")
    decision_git = git_provenance(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_decision_revision,
        "branch": args.expected_decision_branch,
        "tracked_dirty": False,
    }
    if decision_git != expected_git:
        raise ValueError("capacity scaling decision checkout identity differs")
    report = build_from_sources(
        capacity_probe_result_path=args.capacity_probe_result.resolve(),
        expected_capacity_probe_result_sha256=(
            args.expected_capacity_probe_result_sha256
        ),
        standing_authorization_path=args.standing_authorization.resolve(),
        expected_standing_authorization_sha256=(
            args.expected_standing_authorization_sha256
        ),
        decision_git=decision_git,
        expected_capacity_revision=args.expected_capacity_revision,
        expected_capacity_branch=args.expected_capacity_branch,
    )
    write_json_report(args.output, report)
    print(report["recommended_next_stage"]["id"])


if __name__ == "__main__":
    main()
