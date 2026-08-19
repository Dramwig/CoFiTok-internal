from __future__ import annotations

import argparse
import copy
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_full_data_bridge import (
    BASE_CONFIG_RELATIVE_PATHS,
    EXPECTED_OUTPUT_ROOT,
    POSTTRAINING_DECISION,
    RANKED_CONFIG_RELATIVE_PATHS,
    build_full_data_ranked_bridge_preparation,
    validate_full_data_ranked_bridge_preparation,
    validate_posttraining_sampling_confirmation,
)
from cofitok.generation.conditioning_ranking_probe import (
    FOLLOWUP_DECISION_CATEGORY,
    FOLLOWUP_DECISION_ID,
    FOLLOWUP_DECISION_ROLE,
    QUALITY_BRIDGE_EXECUTION_GIT,
    validate_class_conditioning_followup_decision,
    validate_standing_experiment_authorization,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report
from scripts.build_generation_conditioning_ranking_posttraining_sampling_confirmation import (
    replay_posttraining_sampling_confirmation,
)
from scripts.prepare_generation_conditioning_ranking_full_data_bridge import (
    _parameter_count,
)


ROLE = "generation_conditioning_ranking_full_data_bridge_preparation_supervisor"
FOLLOWUP_DECISION_BUILDER_GIT = {
    "revision": "9b02fa83d20b1459a2706d6d82371caf5c023f54",
    "branch": "scale/generation-quality-bridge-followup-decision-v1",
    "tracked_dirty": False,
}
FOLLOWUP_AUTHORIZATION_BOUNDARY = {
    "decision_evidence_complete": True,
    "recommended_stage_execution_allowed": False,
    "quality_bridge_execution_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
    "new_source_compatible_gate_required": True,
}
EXPECTED_FOLLOWUP_CHECKS = {
    "cofitok_absolute_fid",
    "matched_fid_tolerance",
    "cofitok_precision_floor",
    "cofitok_recall_floor",
    "matched_precision_tolerance",
    "matched_recall_tolerance",
    "matched_endpoint_tolerance",
    "ordered_prefix_rank",
    "coarse_token_utilization",
    "restricted_synthesis_zero_token",
    "shuffle_mismatch",
    "class_fidelity",
}
KNOWN_FOLLOWUP_ROUTES = {
    "reconcile_100k_cross_protocol_evidence": "evidence_conflict",
    "build_source_compatible_formal_quality_gate": "formal_gate_preparation",
    "run_matched_factorization_mechanism_recovery_probe": (
        "cofitok_mechanism_recovery"
    ),
    "run_matched_factorization_quality_regression_probe": (
        "matched_quality_regression"
    ),
    FOLLOWUP_DECISION_ID: FOLLOWUP_DECISION_CATEGORY,
    "prepare_matched_250m_capacity_qualification_probe": "capacity_qualification",
    "diagnose_terminal_distribution_support_then_recipe_probe": (
        "recipe_or_objective_intervention"
    ),
    "extend_followup_policy_before_execution": "unclassified_fail_closed",
}
SUPERVISOR_BOUNDARY = {
    "cpu_only": True,
    "gpu_query_allowed": False,
    "training_allowed": False,
    "sampling_allowed": False,
    "experiment_child_process_launch_allowed": False,
    "process_signaling_allowed": False,
    "future_output_root_creation_allowed": False,
    "preparation_write_count_maximum": 1,
    "checkpoint_promotion_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
}


def _bound_json(
    path: Path,
    *,
    expected_sha256: str | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _full_git_status(project: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_exact_clean_git(
    project: Path,
    *,
    expected_git: Mapping[str, Any],
) -> None:
    if git_provenance(project) != dict(expected_git) or _full_git_status(project):
        raise ValueError(
            "full-data ranking preparation supervisor requires the exact fully "
            "clean checkout"
        )


def _valid_identity(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    return (
        isinstance(path, str)
        and bool(path)
        and type(size) is int
        and size > 0
        and isinstance(digest, str)
        and len(digest) == 64
        and digest == digest.lower()
        and all(character in "0123456789abcdef" for character in digest)
    )


def followup_route(report: Mapping[str, Any]) -> str:
    recommendation = report.get("recommended_next_stage")
    terminal = report.get("terminal_quality")
    sources = report.get("source_reports")
    policy = report.get("claim_policy")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "completed"
        or report.get("role") != FOLLOWUP_DECISION_ROLE
        or report.get("decision_builder_git") != FOLLOWUP_DECISION_BUILDER_GIT
        or report.get("quality_bridge_execution_git")
        != QUALITY_BRIDGE_EXECUTION_GIT
        or report.get("authorization_boundary")
        != FOLLOWUP_AUTHORIZATION_BOUNDARY
        or not isinstance(recommendation, Mapping)
        or not isinstance(terminal, Mapping)
        or not isinstance(sources, Mapping)
        or not _valid_identity(sources.get("quality_bridge_result"))
        or not isinstance(policy, Mapping)
        or policy.get("experiment_selection_only") is not True
        or policy.get("milestone_trend_is_formal_quality_evidence") is not False
        or policy.get("terminal_result_is_promotion_gate") is not False
        or policy.get("cross_tier_numeric_ranking_allowed") is not False
    ):
        return "malformed"

    milestones = sources.get("milestones")
    if (
        not isinstance(milestones, Mapping)
        or set(milestones) != {"50000", "100000"}
        or any(not _valid_identity(value) for value in milestones.values())
    ):
        return "malformed"

    rows = terminal.get("checks")
    failed_checks = terminal.get("failed_checks")
    if not isinstance(rows, list) or not isinstance(failed_checks, list):
        return "malformed"
    indexed: set[str] = set()
    observed_failed: list[str] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("name"), str)
            or row.get("name") in indexed
            or type(row.get("passed")) is not bool
        ):
            return "malformed"
        indexed.add(str(row["name"]))
        if row["passed"] is False:
            observed_failed.append(str(row["name"]))
    expected_status = "pass" if not observed_failed else "hold"
    if (
        indexed != EXPECTED_FOLLOWUP_CHECKS
        or observed_failed != failed_checks
        or terminal.get("status") != expected_status
    ):
        return "malformed"

    route_id = recommendation.get("id")
    category = recommendation.get("category")
    if (
        not isinstance(route_id, str)
        or KNOWN_FOLLOWUP_ROUTES.get(route_id) != category
        or recommendation.get("execution_ready") is not False
        or recommendation.get("gpu_execution_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
        or recommendation.get("release_authorization_allowed") is not False
        or not isinstance(recommendation.get("objective"), str)
        or not recommendation.get("objective")
        or not isinstance(recommendation.get("trigger"), Mapping)
        or not isinstance(recommendation.get("required_next_evidence"), str)
        or not recommendation.get("required_next_evidence")
    ):
        return "malformed"
    if route_id != FOLLOWUP_DECISION_ID:
        return "not_selected"
    try:
        validate_class_conditioning_followup_decision(report)
    except ValueError:
        return "malformed"
    return "selected"


def posttraining_route(report: Mapping[str, Any]) -> str:
    decision = report.get("decision")
    methods = report.get("methods")
    if not isinstance(decision, Mapping) or not isinstance(methods, Mapping):
        return "malformed"
    method_passes = decision.get("method_passes")
    if (
        not isinstance(method_passes, Mapping)
        or set(method_passes) != {"cofitok", "dense_identity"}
        or any(type(value) is not bool for value in method_passes.values())
        or set(methods) != {"cofitok", "dense_identity"}
        or any(
            not isinstance(methods[method], Mapping)
            or methods[method].get("pass") is not method_passes[method]
            for method in method_passes
        )
        or decision.get(
            "shared_posttraining_generated_class_alignment_recovery_confirmed"
        )
        is not all(method_passes.values())
        or decision.get("cofitok_specific_advantage_claim_allowed") is not False
    ):
        return "malformed"

    if all(method_passes.values()):
        if dict(decision) != POSTTRAINING_DECISION:
            return "malformed"
        try:
            validate_posttraining_sampling_confirmation(report)
        except ValueError:
            return "malformed"
        return "selected"

    expected_action = (
        "reject_shared_repair_due_posttraining_method_asymmetry"
        if any(method_passes.values())
        else "revise_training_time_semantic_alignment_objective"
    )
    if decision.get("recommended_next_action") != expected_action:
        return "malformed"
    return "not_selected"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_control_paths(
    *,
    project: Path,
    future_output_root: Path,
    preparation_output: Path,
    status_output: Path,
    pid_file: Path,
) -> None:
    paths = (preparation_output, status_output, pid_file)
    if len(set(paths)) != len(paths):
        raise ValueError("full-data ranking supervisor control paths must be distinct")
    for path in paths:
        if _is_within(path, project):
            raise ValueError(
                "full-data ranking supervisor artifacts must be outside Git"
            )
        if _is_within(path, future_output_root):
            raise ValueError(
                "full-data ranking supervisor artifacts must be outside the future "
                "ranked output root"
            )


def _status(
    *,
    status: str,
    detail: str,
    project: Path,
    output_root: str,
    preparation_output: Path,
    sources: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "updated_at_unix": time.time(),
        "pid": os.getpid(),
        "project": project.as_posix(),
        "output_root": output_root,
        "preparation_output": preparation_output.as_posix(),
        "sources": copy.deepcopy(dict(sources or {})),
        "supervisor_boundary": copy.deepcopy(SUPERVISOR_BOUNDARY),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _write_status(path: Path, **kwargs: Any) -> None:
    write_json_report(path, _status(**kwargs))


def _build_preparation_report(
    *,
    project: Path,
    expected_git: Mapping[str, Any],
    output_root: str,
    posttraining: Mapping[str, Any],
    posttraining_identity: Mapping[str, Any],
    followup: Mapping[str, Any],
    followup_identity: Mapping[str, Any],
    standing: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {
        "base_cofitok": reject_symlink_chain(
            project / BASE_CONFIG_RELATIVE_PATHS["cofitok"],
            name="base CoFiTok config",
        ).resolve(),
        "base_dense_identity": reject_symlink_chain(
            project / BASE_CONFIG_RELATIVE_PATHS["dense_identity"],
            name="base dense config",
        ).resolve(),
        "ranked_cofitok": reject_symlink_chain(
            project / RANKED_CONFIG_RELATIVE_PATHS["cofitok"],
            name="ranked CoFiTok config",
        ).resolve(),
        "ranked_dense_identity": reject_symlink_chain(
            project / RANKED_CONFIG_RELATIVE_PATHS["dense_identity"],
            name="ranked dense config",
        ).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("ranked full-data bridge config is missing")
    configs = {name: config_to_dict(load_config(path)) for name, path in paths.items()}
    parameter_counts = {name: _parameter_count(path) for name, path in paths.items()}
    _assert_exact_clean_git(project, expected_git=expected_git)
    return build_full_data_ranked_bridge_preparation(
        posttraining_confirmation=posttraining,
        posttraining_confirmation_identity=posttraining_identity,
        quality_bridge_followup=followup,
        quality_bridge_followup_identity=followup_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        base_cofitok=configs["base_cofitok"],
        base_dense_identity=configs["base_dense_identity"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense_identity=configs["ranked_dense_identity"],
        config_identities={name: file_identity(path) for name, path in paths.items()},
        parameter_counts=parameter_counts,
        builder_git=git_provenance(project),
        expected_revision=str(expected_git["revision"]),
        expected_branch=str(expected_git["branch"]),
        expected_output_root=output_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for exact source gates and prepare, without GPU work, the fresh "
            "full-data matched 100K conditioning-ranking bridge."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--quality-bridge-followup", type=Path, required=True)
    parser.add_argument("--posttraining-confirmation", type=Path, required=True)
    parser.add_argument("--output-root", default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--preparation-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("full-data ranking supervisor timing values must be positive")
    if args.output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError("full-data ranking supervisor output root differs")

    project = reject_symlink_chain(
        args.project,
        name="full-data ranking supervisor project",
    ).resolve()
    if not project.is_dir():
        raise FileNotFoundError("full-data ranking supervisor project is missing")
    future_output_root = reject_symlink_chain(
        args.output_root,
        name="future ranked output root",
    ).resolve()
    preparation_output = reject_symlink_chain(
        args.preparation_output,
        name="full-data ranking preparation",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="full-data ranking supervisor status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="full-data ranking supervisor PID",
    ).resolve()
    _validate_control_paths(
        project=project,
        future_output_root=future_output_root,
        preparation_output=preparation_output,
        status_output=status_output,
        pid_file=pid_file,
    )
    expected_git = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    _assert_exact_clean_git(project, expected_git=expected_git)

    standing_path = reject_symlink_chain(
        args.standing_authorization,
        name="standing experiment authorization",
    ).resolve()
    followup_path = reject_symlink_chain(
        args.quality_bridge_followup,
        name="quality-bridge follow-up decision",
    ).resolve()
    posttraining_path = reject_symlink_chain(
        args.posttraining_confirmation,
        name="posttraining sampling confirmation",
    ).resolve()
    standing, standing_identity = _bound_json(
        standing_path,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    validate_standing_experiment_authorization(standing)
    sources: dict[str, Any] = {"standing_authorization": standing_identity}
    write_json_report(
        pid_file,
        {
            "schema_version": 1,
            "role": f"{ROLE}_pid",
            "pid": os.getpid(),
            "project": project.as_posix(),
            "expected_revision": args.expected_revision,
            "expected_branch": args.expected_branch,
            "output_root": args.output_root,
            "preparation_output": preparation_output.as_posix(),
        },
    )

    started = time.monotonic()
    while True:
        _assert_exact_clean_git(project, expected_git=expected_git)
        if time.monotonic() - started > args.timeout_seconds:
            _write_status(
                status_output,
                status="failed",
                detail="timeout_before_full_data_ranking_preparation",
                project=project,
                output_root=args.output_root,
                preparation_output=preparation_output,
                sources=sources,
            )
            return 4

        if not followup_path.is_file():
            _write_status(
                status_output,
                status="waiting",
                detail="waiting_for_quality_bridge_followup_decision",
                project=project,
                output_root=args.output_root,
                preparation_output=preparation_output,
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue

        followup, followup_identity = _bound_json(
            followup_path,
            expected_sha256=None,
            label="quality-bridge follow-up decision",
        )
        sources["quality_bridge_followup_decision"] = followup_identity
        route = followup_route(followup)
        if route == "malformed":
            raise ValueError("quality-bridge follow-up decision is malformed")
        if route == "not_selected":
            if preparation_output.exists():
                raise FileExistsError(
                    "ranking preparation exists although its route was not selected"
                )
            _write_status(
                status_output,
                status="completed",
                detail="full_data_ranking_bridge_not_selected_by_quality_bridge",
                project=project,
                output_root=args.output_root,
                preparation_output=preparation_output,
                sources=sources,
            )
            return 0

        if not posttraining_path.is_file():
            _write_status(
                status_output,
                status="waiting",
                detail="waiting_for_exact_posttraining_5k_sampling_confirmation",
                project=project,
                output_root=args.output_root,
                preparation_output=preparation_output,
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue

        posttraining, posttraining_identity = (
            replay_posttraining_sampling_confirmation(posttraining_path)
        )
        sources["posttraining_sampling_confirmation"] = posttraining_identity
        posttraining_selection = posttraining_route(posttraining)
        if posttraining_selection == "malformed":
            raise ValueError("posttraining sampling confirmation is malformed")
        if posttraining_selection == "not_selected":
            if preparation_output.exists():
                raise FileExistsError(
                    "ranking preparation exists although posttraining did not pass"
                )
            _write_status(
                status_output,
                status="completed",
                detail=(
                    "full_data_ranking_bridge_not_selected_by_posttraining_"
                    "sampling_confirmation"
                ),
                project=project,
                output_root=args.output_root,
                preparation_output=preparation_output,
                sources=sources,
            )
            return 0

        if future_output_root.exists() or Path(f"{future_output_root}.lock").exists():
            raise FileExistsError("future ranked output root or lock already exists")
        if file_identity(standing_path) != standing_identity:
            raise ValueError("standing experiment authorization changed")
        if file_identity(followup_path) != followup_identity:
            raise ValueError("quality-bridge follow-up decision changed")
        if file_identity(posttraining_path) != posttraining_identity:
            raise ValueError("posttraining sampling confirmation changed")

        report = _build_preparation_report(
            project=project,
            expected_git=expected_git,
            output_root=args.output_root,
            posttraining=posttraining,
            posttraining_identity=posttraining_identity,
            followup=followup,
            followup_identity=followup_identity,
            standing=standing,
            standing_identity=standing_identity,
        )
        validate_full_data_ranked_bridge_preparation(
            report,
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
            expected_output_root=args.output_root,
        )
        _assert_exact_clean_git(project, expected_git=expected_git)
        preparation_identity = prepare_manifest(
            preparation_output,
            report,
            resume=preparation_output.is_file(),
            overwrite=False,
        )
        replayed = read_json_object(
            preparation_output,
            name="full-data ranking preparation",
        )
        validate_full_data_ranked_bridge_preparation(
            replayed,
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
            expected_output_root=args.output_root,
        )
        if replayed != report:
            raise ValueError("full-data ranking preparation replay differs")
        sources["full_data_ranking_preparation"] = preparation_identity
        _write_status(
            status_output,
            status="completed",
            detail="full_data_ranking_bridge_preparation_completed",
            project=project,
            output_root=args.output_root,
            preparation_output=preparation_output,
            sources=sources,
        )
        return 0


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except Exception as error:
        try:
            project = Path(args.project).resolve()
            future_output_root = Path(args.output_root).resolve()
            status_output = reject_symlink_chain(
                args.status_output,
                name="full-data ranking supervisor status",
            ).resolve()
            if _is_within(status_output, project) or _is_within(
                status_output,
                future_output_root,
            ):
                raise ValueError("unsafe failure-status path")
            _write_status(
                status_output,
                status="failed",
                detail="full_data_ranking_bridge_preparation_supervisor_failed",
                project=project,
                output_root=str(args.output_root),
                preparation_output=Path(args.preparation_output).resolve(),
                sources={},
                error=f"{type(error).__name__}: {error}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
