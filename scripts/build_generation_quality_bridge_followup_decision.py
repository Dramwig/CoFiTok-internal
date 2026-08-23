from __future__ import annotations

import argparse
import copy
import subprocess
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
    authoritative_terminal_verification: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("quality bridge result SHA256 differs")
    actual = read_json_object(path, name="quality bridge terminal result")
    if authoritative_terminal_verification is None:
        recomputed = build_from_args(_result_replay_args(actual))
        if actual != recomputed:
            raise ValueError("quality bridge terminal result is not reproducible")
    else:
        _validate_authoritative_terminal_verification(
            actual,
            identity,
            authoritative_terminal_verification,
        )
        sources = actual.get("source_reports")
        if not isinstance(sources, dict) or set(sources) != _RESULT_SOURCE_ARGUMENTS:
            raise ValueError("quality bridge result source set differs")
        for name, source in sources.items():
            _bound_json_source(source, label=f"quality bridge terminal {name}")
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


def _same_content_identity(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: str,
) -> None:
    if {
        "bytes": left.get("bytes"),
        "sha256": left.get("sha256"),
    } != {
        "bytes": right.get("bytes"),
        "sha256": right.get("sha256"),
    }:
        raise ValueError(f"{label} content identity differs")


def _validate_authoritative_terminal_verification(
    terminal: dict[str, Any],
    terminal_identity: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    if (
        verification.get("schema_version") != 1
        or verification.get("role")
        != "generation_quality_bridge_authoritative_terminal_verification"
        or verification.get("status") != "verified"
    ):
        raise ValueError("authoritative quality bridge verification contract differs")
    terminal_git = terminal.get("git")
    quality_project = verification.get("quality_project")
    if (
        not isinstance(terminal_git, dict)
        or not isinstance(quality_project, dict)
        or quality_project.get("revision") != terminal_git.get("revision")
        or quality_project.get("branch") != terminal_git.get("branch")
        or quality_project.get("tracked_dirty") is not False
        or not isinstance(quality_project.get("tree"), str)
        or len(quality_project["tree"]) != 40
        or not isinstance(quality_project.get("path"), str)
        or not quality_project["path"]
    ):
        raise ValueError("authoritative quality bridge project identity differs")
    expected_quality_project = {
        "revision": quality_project["revision"],
        "tree": quality_project["tree"],
        "branch": quality_project["branch"],
        "tracked_dirty": False,
        "path": quality_project["path"],
    }
    if _authoritative_quality_project_identity(
        Path(quality_project["path"])
    ) != expected_quality_project:
        raise ValueError("authoritative quality bridge checkout identity differs")
    for label in ("verifier_source", "builder_source", "python"):
        source = verification.get(label)
        if not isinstance(source, dict):
            raise ValueError(f"authoritative {label} identity is missing")
        path = Path(str(source.get("path", "")))
        if file_identity(path) != source:
            raise ValueError(f"authoritative {label} identity differs")
    verified_result = verification.get("terminal_result")
    if not isinstance(verified_result, dict):
        raise ValueError("authoritatively verified terminal result is missing")
    _same_content_identity(
        verified_result,
        terminal_identity,
        label="authoritatively verified terminal result",
    )
    verifier_output = verification.get("verifier_output")
    if not isinstance(verifier_output, dict):
        raise ValueError("authoritative quality bridge verifier output is missing")
    output_result = verifier_output.get("result")
    if not isinstance(output_result, dict):
        raise ValueError("authoritative verifier result identity is missing")
    _same_content_identity(
        output_result,
        terminal_identity,
        label="authoritative verifier result",
    )
    if (
        verifier_output.get("status") != "verified"
        or verifier_output.get("quality_screen") != terminal.get("quality_screen")
        or verifier_output.get("authorization_boundary")
        != terminal.get("authorization_boundary")
        or verification.get("execution_policy")
        != {
            "cuda_visible_devices": "-1",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
        }
    ):
        raise ValueError("authoritative quality bridge verification differs")
    return copy.deepcopy(verification)


def _authoritative_quality_project_identity(project: Path) -> dict[str, Any]:
    identity = git_provenance(project)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=str(project),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        **identity,
        "tree": tree,
        "path": project.resolve().as_posix(),
    }


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
    authoritative_terminal_verification = terminal_binding.get(
        "authoritative_terminal_verification"
    )
    if authoritative_terminal_verification is not None and not isinstance(
        authoritative_terminal_verification, dict
    ):
        raise ValueError(
            "terminal training exposure authoritative verification is malformed"
        )
    recomputed = build_training_exposure_report(
        training_reports,
        quality_bridge_preparation=preparation,
        milestone_report=milestone,
        expected_milestone_step=int(milestone_binding.get("expected_step", -1)),
        milestone_source_profile=str(milestone_binding.get("source_profile", "")),
        quality_bridge_terminal_result=terminal_result,
        quality_bridge_execution_status=execution_status,
        quality_bridge_terminal_verification=authoritative_terminal_verification,
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
    exposure, exposure_identity = replay_training_exposure_report(
        training_exposure_report_path,
        expected_sha256=expected_training_exposure_report_sha256,
    )
    terminal_binding = exposure.get("terminal_binding")
    if not isinstance(terminal_binding, dict):
        raise ValueError("terminal training exposure binding is missing")
    verification = terminal_binding.get("authoritative_terminal_verification")
    if not isinstance(verification, dict):
        raise ValueError(
            "terminal training exposure authoritative verification is missing"
        )
    result, result_identity = replay_quality_bridge_result(
        quality_bridge_result_path,
        expected_sha256=expected_quality_bridge_result_sha256,
        authoritative_terminal_verification=verification,
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
