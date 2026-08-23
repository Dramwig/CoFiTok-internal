from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)


SCHEMA_VERSION = 1
ROLE = "generation_terminal_runtime_strict_conjunct"
TERMINAL_AUDIT_ROLE = "generation_quality_bridge_terminal_completion_audit"
TERMINAL_WAITER_ROLE = "generation_quality_bridge_terminal_completion_audit_waiter"
RUNTIME_COMPARISON_ROLE = "generation_runtime_claim_guard_strict_comparison"
RUNTIME_WAITER_ROLE = "generation_runtime_claim_guard_strict_comparison_waiter"
EXPECTED_METHODS = {"cofitok", "dense_identity"}
TERMINAL_DECISIONS = {
    "pass": "matched_quality_advantage_qualified_with_terminal_system_evidence",
    "hold": "terminal_system_evidence_complete_without_qualified_matched_advantage",
}
TERMINAL_BOUNDARY_REQUIRED = {
    "diagnostic_non_authorizing": True,
    "cpu_only_evidence_replay_allowed": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "inference_export_authorization_allowed": False,
    "process_signals_allowed": False,
    "upstream_decisions_modified": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "sota_claim_allowed": False,
}
AUTHORIZATION_BOUNDARY = {
    **TERMINAL_BOUNDARY_REQUIRED,
    "runtime_strict_comparator_required": True,
    "runtime_direct_ranking_allowed": False,
}
DIRECT_RUNTIME_FIELDS = (
    "training_wall_clock_direct_comparison_allowed",
    "training_throughput_direct_comparison_allowed",
    "cost_efficiency_ranking_allowed",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} expected SHA256 is invalid")
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return identity, read_json_object(source, name=label)


def _terminal_completion(
    report: Mapping[str, Any],
    *,
    expected_training_revision: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    scope = report.get("scope")
    claim = report.get("claim_policy")
    boundary = report.get("authorization_boundary")
    terminal_status = report.get("terminal_status")
    terminal_decision = report.get("terminal_decision")
    advantage = report.get("generation_advantage_proven")
    training_git = scope.get("training_git") if isinstance(scope, Mapping) else None
    if (
        report.get("schema_version") != 1
        or report.get("role") != TERMINAL_AUDIT_ROLE
        or report.get("status") != "pass"
        or report.get("detail")
        != "terminal_quality_bridge_evidence_physically_replayed"
        or terminal_status not in {"pass", "hold"}
        or terminal_decision != TERMINAL_DECISIONS.get(terminal_status)
        or not isinstance(advantage, bool)
        or not isinstance(training_git, Mapping)
        or training_git.get("revision") != expected_training_revision
        or training_git.get("branch") != expected_training_branch
        or training_git.get("tracked_dirty") is not False
        or not isinstance(claim, Mapping)
        or claim.get("terminal_system_evidence_complete") is not True
        or claim.get("matched_distribution_quality_claim_allowed") is not advantage
        or claim.get("absolute_usability_claim_allowed") is not False
        or claim.get("broad_generation_superiority_claim_allowed") is not False
        or claim.get("sota_claim_allowed") is not False
        or claim.get("larger_training_launch_allowed") is not False
        or claim.get("full_300k_launch_allowed") is not False
        or claim.get("promotion_or_release_allowed") is not False
        or not isinstance(boundary, Mapping)
        or any(
            boundary.get(key) is not value
            for key, value in TERMINAL_BOUNDARY_REQUIRED.items()
        )
    ):
        raise ValueError("terminal completion audit contract differs")
    if terminal_status == "hold" and advantage:
        raise ValueError("terminal hold cannot prove a generation advantage")
    if terminal_status == "pass" and not advantage:
        raise ValueError("terminal pass must prove the scoped generation advantage")
    return {
        "terminal_status": terminal_status,
        "terminal_decision": terminal_decision,
        "generation_advantage_proven": advantage,
        "training_git": copy.deepcopy(dict(training_git)),
    }


def _terminal_waiter(
    report: Mapping[str, Any],
    *,
    audit_identity: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    audit = report.get("audit")
    scope = report.get("scope")
    if (
        report.get("schema_version") != 1
        or report.get("role") != TERMINAL_WAITER_ROLE
        or report.get("status") != "pass"
        or report.get("detail") != "terminal_completion_audit_revalidated"
        or report.get("terminal_status") != terminal["terminal_status"]
        or report.get("generation_advantage_proven")
        is not terminal["generation_advantage_proven"]
        or not isinstance(audit, Mapping)
        or audit.get("identity") != dict(audit_identity)
        or audit.get("terminal_status") != terminal["terminal_status"]
        or audit.get("terminal_decision") != terminal["terminal_decision"]
        or audit.get("generation_advantage_proven")
        is not terminal["generation_advantage_proven"]
        or not isinstance(scope, Mapping)
        or scope.get("cpu_only_evidence_replay") is not True
        or scope.get("diagnostic_non_authorizing") is not True
        or scope.get("gpu_required") is not False
        or scope.get("training_launch_allowed") is not False
        or scope.get("sampling_launch_allowed") is not False
        or scope.get("full_training_launch_allowed") is not False
        or scope.get("full_300k_launch_allowed") is not False
        or scope.get("promotion_authorization_allowed") is not False
        or scope.get("release_authorization_allowed") is not False
        or scope.get("inference_export_authorization_allowed") is not False
        or scope.get("process_signals_allowed") is not False
        or scope.get("upstream_decisions_modified") is not False
    ):
        raise ValueError("terminal completion waiter contract differs")
    return {
        "pid": report.get("pid"),
        "detail": report.get("detail"),
        "scope": copy.deepcopy(dict(scope)),
    }


def _runtime_comparison(
    report: Mapping[str, Any],
    *,
    expected_training_revision: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    training = report.get("training_identity")
    strict = report.get("strict_policy")
    policy = report.get("claim_policy")
    boundary = report.get("claim_boundary")
    if (
        report.get("schema_version") != 1
        or report.get("role") != RUNTIME_COMPARISON_ROLE
        or report.get("status") != "pass"
        or report.get("decision")
        != "canonical_observational_runtime_claim_semantically_verified"
        or not isinstance(training, Mapping)
        or training.get("revision") != expected_training_revision
        or training.get("branch") != expected_training_branch
        or not isinstance(strict, Mapping)
        or set(strict.get("physical_lower_bound_methods") or []) != EXPECTED_METHODS
        or strict.get("elapsed_metric_role")
        != "observational_physical_lower_bounds_only"
        or strict.get("direct_ranking_allowed") is not False
        or not isinstance(policy, Mapping)
        or policy.get("canonical_runtime_claim_trusted") is not True
        or policy.get("strict_runtime_claim_guard_required") is not True
        or policy.get("physical_lower_bound_label_required") is not True
        or policy.get("observational_only_label_required") is not True
        or policy.get("adjusted_elapsed_point_estimate_reporting_allowed") is not True
        or any(policy.get(field) is not False for field in DIRECT_RUNTIME_FIELDS)
        or policy.get("equal_wall_clock_budget_claim_allowed") is not False
        or policy.get("equal_gpu_hours_budget_claim_allowed") is not False
        or policy.get("equal_training_flops_budget_claim_allowed") is not False
        or policy.get("quality_or_generation_advantage_claim_allowed") is not False
        or not isinstance(boundary, Mapping)
        or boundary.get("diagnostic_non_authorizing") is not True
        or boundary.get("gpu_execution_allowed") is not False
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("sampling_launch_allowed") is not False
        or boundary.get("inference_export_authorization_allowed") is not False
        or boundary.get("release_authorization_allowed") is not False
        or boundary.get("process_signals_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("runtime strict comparison contract differs")
    return {
        "decision": report["decision"],
        "physical_lower_bound_methods": sorted(EXPECTED_METHODS),
        "direct_runtime_ranking_allowed": False,
        "canonical_runtime_claim_trusted": True,
    }


def _runtime_waiter(
    report: Mapping[str, Any],
    *,
    comparison_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = report.get("comparison")
    scope = report.get("scope")
    if (
        report.get("schema_version") != 1
        or report.get("role") != RUNTIME_WAITER_ROLE
        or report.get("status") != "pass"
        or report.get("detail") != runtime["decision"]
        or not isinstance(comparison, Mapping)
        or comparison.get("identity") != dict(comparison_identity)
        or comparison.get("status") != "pass"
        or comparison.get("decision") != runtime["decision"]
        or not isinstance(comparison.get("claim_policy"), Mapping)
        or comparison["claim_policy"].get("canonical_runtime_claim_trusted")
        is not True
        or comparison["claim_policy"].get("cost_efficiency_ranking_allowed")
        is not False
        or not isinstance(scope, Mapping)
        or scope.get("cpu_only") is not True
        or scope.get("non_authorizing") is not True
        or scope.get("gpu_execution_allowed") is not False
        or scope.get("checkpoint_payload_loading_allowed") is not False
        or scope.get("training_process_signals_allowed") is not False
        or scope.get("unrelated_process_signals_allowed") is not False
        or scope.get("source_waiter_signals_allowed") is not False
        or scope.get("promotion_authorization_allowed") is not False
        or scope.get("release_authorization_allowed") is not False
        or scope.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("runtime strict comparison waiter contract differs")
    return {
        "pid": report.get("pid"),
        "detail": report.get("detail"),
        "scope": copy.deepcopy(dict(scope)),
    }


def build_conjunct(
    *,
    terminal_status_path: Path,
    expected_terminal_status_sha256: str,
    terminal_audit_path: Path,
    expected_terminal_audit_sha256: str,
    runtime_status_path: Path,
    expected_runtime_status_sha256: str,
    runtime_comparison_path: Path,
    expected_runtime_comparison_sha256: str,
    expected_training_revision: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    terminal_audit_identity, terminal_audit = _bound_json(
        terminal_audit_path,
        expected_sha256=expected_terminal_audit_sha256,
        label="terminal completion audit",
    )
    terminal_status_identity, terminal_status = _bound_json(
        terminal_status_path,
        expected_sha256=expected_terminal_status_sha256,
        label="terminal completion waiter status",
    )
    runtime_identity, runtime_report = _bound_json(
        runtime_comparison_path,
        expected_sha256=expected_runtime_comparison_sha256,
        label="runtime strict comparison",
    )
    runtime_status_identity, runtime_status = _bound_json(
        runtime_status_path,
        expected_sha256=expected_runtime_status_sha256,
        label="runtime strict comparison waiter status",
    )
    terminal = _terminal_completion(
        terminal_audit,
        expected_training_revision=expected_training_revision,
        expected_training_branch=expected_training_branch,
    )
    terminal_waiter = _terminal_waiter(
        terminal_status,
        audit_identity=terminal_audit_identity,
        terminal=terminal,
    )
    runtime = _runtime_comparison(
        runtime_report,
        expected_training_revision=expected_training_revision,
        expected_training_branch=expected_training_branch,
    )
    runtime_waiter = _runtime_waiter(
        runtime_status,
        comparison_identity=runtime_identity,
        runtime=runtime,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "pass",
        "detail": "terminal_completion_and_runtime_strict_comparator_conjoined",
        "terminal_status": terminal["terminal_status"],
        "terminal_decision": terminal["terminal_decision"],
        "generation_advantage_proven": terminal["generation_advantage_proven"],
        "training_identity": {
            "revision": expected_training_revision,
            "branch": expected_training_branch,
        },
        "sources": {
            "terminal_completion_waiter_status": terminal_status_identity,
            "terminal_completion_audit": terminal_audit_identity,
            "runtime_strict_comparison_waiter_status": runtime_status_identity,
            "runtime_strict_comparison": runtime_identity,
        },
        "replay": {
            "terminal_completion": terminal,
            "terminal_waiter": terminal_waiter,
            "runtime_strict_comparison": runtime,
            "runtime_waiter": runtime_waiter,
        },
        "claim_policy": {
            "terminal_completion_audit_required": True,
            "runtime_strict_comparator_required": True,
            "runtime_strict_comparator_passed": True,
            "canonical_runtime_claim_trusted_only_as_observational": True,
            "physical_lower_bound_label_required": True,
            "training_wall_clock_direct_comparison_allowed": False,
            "training_throughput_direct_comparison_allowed": False,
            "cost_efficiency_ranking_allowed": False,
            "generation_advantage_proven": terminal["generation_advantage_proven"],
            "absolute_usability_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_or_release_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
        "limitations": [
            (
                "This conjunct closes only the terminal runtime-claim evidence gap; "
                "it does not create a quality, usability, training, promotion, or "
                "release authorization."
            ),
            (
                "Recovery-adjusted elapsed remains a physical lower bound, so direct "
                "wall-clock, throughput, and cost-efficiency ranking stays disabled."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conjoin the terminal completion audit with the independently replayed "
            "strict runtime-claim comparator without authorizing any GPU stage."
        )
    )
    for name in (
        "terminal-status",
        "terminal-audit",
        "runtime-status",
        "runtime-comparison",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="terminal runtime strict conjunct output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="terminal runtime strict conjunct output",
    ).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("terminal runtime conjunct output is outside output root") from error
    report = build_conjunct(
        terminal_status_path=args.terminal_status,
        expected_terminal_status_sha256=args.expected_terminal_status_sha256,
        terminal_audit_path=args.terminal_audit,
        expected_terminal_audit_sha256=args.expected_terminal_audit_sha256,
        runtime_status_path=args.runtime_status,
        expected_runtime_status_sha256=args.expected_runtime_status_sha256,
        runtime_comparison_path=args.runtime_comparison,
        expected_runtime_comparison_sha256=args.expected_runtime_comparison_sha256,
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
    )
    identity = prepare_manifest(output, report, resume=args.resume, overwrite=False)
    print(json.dumps({"status": report["status"], "conjunct": identity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
