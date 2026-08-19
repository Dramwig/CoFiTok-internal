from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cofitok.generation.quality_bridge_followup import (
    QUALITY_BRIDGE_EXECUTION_BRANCH,
    QUALITY_BRIDGE_EXECUTION_REVISION,
    build_quality_bridge_followup_decision,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance, write_json_report

try:
    from scripts.build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )
    from scripts.build_generation_quality_bridge_result import build_from_args
    from scripts.build_generation_training_exposure_audit import (
        build_report as build_training_exposure_report,
    )
except ModuleNotFoundError:
    from build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )
    from build_generation_quality_bridge_result import build_from_args
    from build_generation_training_exposure_audit import (
        build_report as build_training_exposure_report,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RESULT_SOURCE_ARGUMENTS = {
    "preparation",
    "launch_receipt",
    "cofitok_training",
    "dense_training",
    "training_pair_validation",
    "cofitok_training_audit",
    "dense_training_audit",
    "milestone_50000",
    "milestone_100000",
    "cofitok_sampling_preflight",
    "dense_sampling_preflight",
    "cofitok_generation",
    "dense_generation",
    "cofitok_checkpoint_eval",
    "dense_checkpoint_eval",
    "class_fidelity_qualification",
    "cofitok_class_fidelity",
    "dense_class_fidelity",
}


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quality-bridge-result", type=Path, required=True)
    parser.add_argument("--expected-quality-bridge-result-sha256", required=True)
    parser.add_argument("--training-exposure-report", type=Path, required=True)
    parser.add_argument("--expected-training-exposure-report-sha256", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-branch", required=True)


def _result_replay_args(result: dict[str, Any]) -> SimpleNamespace:
    sources = result.get("source_reports")
    if not isinstance(sources, dict) or set(sources) != _RESULT_SOURCE_ARGUMENTS:
        raise ValueError("quality bridge result source set differs")
    values: dict[str, Any] = {
        name: Path(str(sources[name]["path"])) for name in _RESULT_SOURCE_ARGUMENTS
    }
    values.update(
        {
            "expected_preparation_sha256": sources["preparation"]["sha256"],
            "expected_launch_receipt_sha256": sources["launch_receipt"]["sha256"],
            "expected_revision": QUALITY_BRIDGE_EXECUTION_REVISION,
            "expected_branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
        }
    )
    return SimpleNamespace(**values)


def replay_quality_bridge_result(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("quality bridge result SHA256 differs")
    actual = read_json_object(path, name="quality bridge terminal result")
    recomputed = build_from_args(_result_replay_args(actual))
    if actual != recomputed:
        raise ValueError("quality bridge terminal result is not reproducible")
    return actual, identity


def _bound_json_source(
    value: object,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} identity is missing")
    path = Path(str(value.get("path", "")))
    identity = file_identity(path)
    if identity != value:
        raise ValueError(f"{label} identity differs")
    return read_json_object(path, name=label), identity


def replay_training_exposure_report(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("terminal training exposure report SHA256 differs")
    actual = read_json_object(path, name="terminal training exposure report")
    sources = actual.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "cofitok",
        "dense_identity",
    }:
        raise ValueError("terminal training exposure source set differs")
    training_reports = {
        method: _bound_json_source(
            sources[method],
            label=f"terminal training exposure {method} training report",
        )
        for method in ("cofitok", "dense_identity")
    }
    quality_bridge_plan = actual.get("quality_bridge_plan")
    if not isinstance(quality_bridge_plan, dict):
        raise TypeError("terminal training exposure quality bridge plan is missing")
    preparation = _bound_json_source(
        quality_bridge_plan.get("source"),
        label="terminal training exposure quality bridge preparation",
    )
    milestone_binding = actual.get("milestone_binding")
    if not isinstance(milestone_binding, dict):
        raise TypeError("terminal training exposure milestone binding is missing")
    milestone = _bound_json_source(
        milestone_binding.get("source"),
        label="terminal training exposure 100K milestone",
    )
    terminal_binding = actual.get("terminal_binding")
    if not isinstance(terminal_binding, dict):
        raise TypeError("terminal training exposure terminal binding is missing")
    terminal_result = _bound_json_source(
        terminal_binding.get("terminal_result"),
        label="terminal training exposure quality bridge result",
    )
    execution_status = _bound_json_source(
        terminal_binding.get("verified_execution_status"),
        label="terminal training exposure execution status",
    )
    recomputed = build_training_exposure_report(
        training_reports,
        quality_bridge_preparation=preparation,
        milestone_report=milestone,
        expected_milestone_step=int(milestone_binding.get("expected_step", -1)),
        milestone_source_profile=str(milestone_binding.get("source_profile", "")),
        quality_bridge_terminal_result=terminal_result,
        quality_bridge_execution_status=execution_status,
    )
    if actual != recomputed:
        raise ValueError("terminal training exposure report is not reproducible")
    return actual, identity


def _milestone(
    result: dict[str, Any],
    *,
    step: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_name = f"milestone_{step}"
    source = result["source_reports"][source_name]
    path = Path(str(source["path"]))
    identity = file_identity(path)
    if identity != source:
        raise ValueError(f"quality bridge milestone {step} identity differs")
    report = read_json_object(path, name=f"quality bridge milestone {step}")
    source_verification = verify_milestone_source_reports(
        report,
        source_profile="quality_bridge",
    )
    evidence, warnings = validate_milestone_report(
        report,
        expected_step=step,
        source_verification=source_verification,
        expected_source_profile="quality_bridge",
    )
    expected_wrapper = {
        "status": "verified",
        "report": identity,
        "evidence": evidence,
        "warnings": warnings,
    }
    if result.get("milestones", {}).get(str(step)) != expected_wrapper:
        raise ValueError(f"quality bridge result milestone {step} replay differs")
    verification = {
        **source_verification,
        "evidence": evidence,
        "warnings": warnings,
    }
    return report, identity, verification


def build_from_sources(
    *,
    quality_bridge_result_path: Path,
    expected_quality_bridge_result_sha256: str,
    training_exposure_report_path: Path,
    expected_training_exposure_report_sha256: str,
    decision_git: dict[str, Any],
) -> dict[str, Any]:
    result, result_identity = replay_quality_bridge_result(
        quality_bridge_result_path,
        expected_sha256=expected_quality_bridge_result_sha256,
    )
    exposure, exposure_identity = replay_training_exposure_report(
        training_exposure_report_path,
        expected_sha256=expected_training_exposure_report_sha256,
    )
    milestones: dict[int, dict[str, Any]] = {}
    identities: dict[int, dict[str, Any]] = {}
    verifications: dict[int, dict[str, Any]] = {}
    for step in (50_000, 100_000):
        report, identity, verification = _milestone(result, step=step)
        milestones[step] = report
        identities[step] = identity
        verifications[step] = verification
    return build_quality_bridge_followup_decision(
        quality_bridge_result=result,
        quality_bridge_result_identity=result_identity,
        milestones=milestones,
        milestone_identities=identities,
        milestone_verifications=verifications,
        training_exposure_report=exposure,
        training_exposure_report_identity=exposure_identity,
        decision_git=decision_git,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-replayed, non-authorizing next-experiment decision "
            "from the completed full-data 100K quality bridge."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"quality bridge follow-up decision exists: {args.output}"
        )
    decision_git = git_provenance(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_decision_revision,
        "branch": args.expected_decision_branch,
        "tracked_dirty": False,
    }
    if decision_git != expected_git:
        raise ValueError("quality bridge follow-up decision checkout identity differs")
    report = build_from_sources(
        quality_bridge_result_path=args.quality_bridge_result,
        expected_quality_bridge_result_sha256=(
            args.expected_quality_bridge_result_sha256
        ),
        training_exposure_report_path=args.training_exposure_report,
        expected_training_exposure_report_sha256=(
            args.expected_training_exposure_report_sha256
        ),
        decision_git=decision_git,
    )
    write_json_report(args.output, report)
    print(report["recommended_next_stage"]["id"])


if __name__ == "__main__":
    main()
