from __future__ import annotations

import argparse
import copy
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)

SCHEMA_VERSION = 1
ROLE = "generation_terminal_runtime_strict_route_selection"

QUALITY_RESULT_ROLE = "stability_full_data_quality_bridge_result"
FOLLOWUP_ROLE = "stability_quality_bridge_followup_experiment_decision"
EXPOSURE_WAITER_ROLE = "generation_quality_bridge_exposure_aware_followup_waiter"
TERMINAL_GUARD_ROLE = "generation_terminal_system_claim_guard"
TERMINAL_GUARD_WAITER_ROLE = "generation_terminal_system_claim_guard_waiter"
COMPARISON_ROLE = "stability_full_data_quality_bridge_comparison"
COMPARISON_WAITER_ROLE = "generation_quality_bridge_comparison_waiter"
COMPLETION_ROLE = "generation_quality_bridge_terminal_completion_audit"
COMPLETION_WAITER_ROLE = "generation_quality_bridge_terminal_completion_audit_waiter"
CONJUNCT_ROLE = "generation_terminal_runtime_strict_conjunct"
CONJUNCT_WAITER_ROLE = "generation_terminal_runtime_strict_conjunct_waiter"
INTERLOCK_ROLE = "generation_terminal_route_supersession_interlock"

TRAINING_GIT = {
    "revision": "cf0e5faa94bf4ab38d947b921935b3b765b5537a",
    "tree": "6cef27723196fd363379bca2e7b85b1678ebd777",
    "branch": "scale/generation-stability-quality-bridge-100k",
    "tracked_dirty": False,
}
EXPOSURE_GIT = {
    "revision": "cd78a348769f0efad0d42de063e5b0943444a29b",
    "tree": "845c539e4e38d6d1b29ad65536432a19c371aadc",
    "branch": "analysis/generation-quality-bridge-mixed-route-v3-20260822",
    "tracked_dirty": False,
}
TERMINAL_GUARD_GIT = {
    "revision": "4087f4293e3fd197c81cbfa9029f3d6083654415",
    "tree": "7d5acbe2542f5c0584a7fd847fdb806e6757ba86",
    "branch": "analysis/generation-terminal-classifier-integrity-v1-20260822",
    "tracked_dirty": False,
}
COMPARISON_GIT = {
    "revision": "ac0388869ee5dd0f3a36bbf9f79550eed99a6f9c",
    "tree": "c001efd3c44c1c0c6b8f660a0b9c8f613fc217a4",
    "branch": "analysis/generation-quality-bridge-comparison-runtime-strict-v2-20260823",
    "tracked_dirty": False,
}
COMPLETION_GIT = {
    "revision": "107f739035a71297fa9d970281097e04fa1a0b45",
    "tree": "e57229608c3ce72ab94cfe3a31251f87c3fba89e",
    "branch": "analysis/generation-terminal-completion-runtime-strict-v2-20260823",
    "tracked_dirty": False,
}
CONJUNCT_GIT = {
    "revision": "b9c03a70204fe7e89af39c041894539f31097086",
    "tree": "c3d0435f15d5b47c6f2df8093c25fcee0475cbc7",
    "branch": "analysis/generation-terminal-runtime-strict-conjunct-completion-rebind-v3-20260823",
    "tracked_dirty": False,
}

EXPECTED_CHECKS = {
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
MATCHED_QUALITY_CHECKS = {
    "matched_fid_tolerance",
    "matched_precision_tolerance",
    "matched_recall_tolerance",
}
TERMINAL_DECISIONS = {
    "pass": "matched_quality_advantage_qualified_with_terminal_system_evidence",
    "hold": "terminal_system_evidence_complete_without_qualified_matched_advantage",
}

QUALITY_RESULT_BOUNDARY = {
    "quality_bridge_evidence_complete": True,
    "quality_bridge_execution_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
    "new_gate_required": True,
}
FOLLOWUP_BOUNDARY = {
    "decision_evidence_complete": True,
    "recommended_stage_execution_allowed": False,
    "quality_bridge_execution_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
    "new_source_compatible_gate_required": True,
}
AUTHORIZATION_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "cpu_only_route_selection_allowed": True,
    "route_execution_authorized": False,
    "corrected_executor_launch_allowed": False,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "inference_export_authorization_allowed": False,
    "process_signals_allowed": False,
    "legacy_interlocks_modified": False,
    "upstream_decisions_modified": False,
}
SCOPE = {
    "cpu_only": True,
    "permanently_non_authorizing": True,
    "route_selection_only": True,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "inference_export_authorization_allowed": False,
    "process_signals_allowed": False,
    "legacy_interlocks_modified": False,
    "upstream_decisions_modified": False,
}


@dataclass(frozen=True)
class RouteSourcePaths:
    quality_root: Path
    standing_authorization: Path
    factorization_marker: Path
    conditioning_marker: Path
    random_token_marker: Path

    @property
    def reports(self) -> Path:
        return self.quality_root / "reports"

    def json_paths(self) -> dict[str, Path]:
        reports = self.reports
        return {
            "pair_monitor": self.quality_root / "pair_monitor.json",
            "quality_bridge_result": reports / "quality_bridge_result.json",
            "exposure_waiter_status": reports
            / "exposure_aware_followup_waiter_status.json",
            "exposure_deployment_receipt": reports
            / "exposure_aware_followup_waiter_deployment_receipt.2026-08-22_mixed_route_v3.json",
            "followup_decision": reports
            / "followup_experiment_decision_exposure_aware_v2.json",
            "runtime_guard": reports
            / "runtime_compute_claim_guard_strict_replay_v3"
            / "runtime_compute_claim_guard.json",
            "terminal_guard_status": reports
            / "terminal_system_claim_guard_v2_runtime_strict"
            / "waiter_status.json",
            "terminal_guard": reports
            / "terminal_system_claim_guard_v2_runtime_strict"
            / "terminal_system_claim_guard.json",
            "comparison_deployment_receipt": reports
            / "quality_bridge_comparison_v3_runtime_strict"
            / "deployment_receipt.json",
            "comparison_status": reports
            / "quality_bridge_comparison_v3_runtime_strict"
            / "waiter_status.json",
            "comparison": reports
            / "quality_bridge_comparison_v3_runtime_strict"
            / "quality_bridge_comparison.json",
            "completion_deployment_receipt": reports
            / "terminal_completion_audit_v2_runtime_strict"
            / "deployment_receipt.json",
            "completion_status": reports
            / "terminal_completion_audit_v2_runtime_strict"
            / "waiter_status.json",
            "completion": reports
            / "terminal_completion_audit_v2_runtime_strict"
            / "terminal_completion_audit.json",
            "conjunct_deployment_receipt": reports
            / "terminal_runtime_strict_conjunct_v4_runtime_strict"
            / "deployment_receipt.json",
            "conjunct_status": reports
            / "terminal_runtime_strict_conjunct_v4_runtime_strict"
            / "waiter_status.json",
            "conjunct": reports
            / "terminal_runtime_strict_conjunct_v4_runtime_strict"
            / "terminal_runtime_strict_conjunct.json",
            "interlock_receipt": reports
            / "terminal_route_supersession_interlock_v1"
            / "supersession_receipt.json",
            "standing_authorization": self.standing_authorization,
            "factorization_marker": self.factorization_marker,
            "conditioning_marker": self.conditioning_marker,
            "random_token_marker": self.random_token_marker,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label)
    identity = file_identity(source)
    if expected_sha256 is not None:
        _require(
            identity["sha256"] == expected_sha256,
            f"{label} SHA256 differs",
        )
    payload = read_json_object(source, name=label)
    _require(file_identity(source) == identity, f"{label} changed during replay")
    return identity, payload


def _identity(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} identity is missing")
    normalized = {
        "path": value.get("path"),
        "bytes": value.get("bytes"),
        "sha256": value.get("sha256"),
    }
    _require(
        isinstance(normalized["path"], str)
        and isinstance(normalized["bytes"], int)
        and normalized["bytes"] >= 0
        and isinstance(normalized["sha256"], str)
        and len(normalized["sha256"]) == 64,
        f"{label} identity is malformed",
    )
    return normalized


def _git(value: Any, *, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} Git identity is missing")
    observed = {
        "revision": value.get("revision"),
        "tree": value.get("tree", expected.get("tree")),
        "branch": value.get("branch"),
        "tracked_dirty": value.get("tracked_dirty", value.get("dirty")),
    }
    _require(observed == dict(expected), f"{label} Git identity differs")
    return observed


def _require_false_fields(
    value: Any,
    *,
    fields: set[str],
    label: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    normalized = dict(value)
    for field in fields:
        _require(normalized.get(field) is False, f"{label}.{field} must be false")
    return normalized


def _validate_pair_monitor(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    _require(report.get("status") == "pass", "pair monitor is not pass")
    _require(report.get("stage") == "complete", "pair monitor is not complete")
    _require(report.get("issues") == [], "pair monitor has issues")
    return {"identity": dict(identity), "status": "pass", "stage": "complete"}


def _validate_runtime_guard(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    pair_monitor_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == "generation_runtime_compute_claim_guard"
        and report.get("status") == "pass"
        and report.get("decision") == "runtime_cost_claims_observational_only",
        "runtime-strict compute claim guard contract differs",
    )
    sources = report.get("sources")
    _require(isinstance(sources, Mapping), "runtime-strict guard sources missing")
    _require(
        _identity(
            sources.get("terminal_pair_monitor"),
            label="runtime-strict terminal pair monitor",
        )
        == dict(pair_monitor_identity),
        "runtime-strict guard binds another pair monitor",
    )
    policy = report.get("claim_policy")
    _require(
        isinstance(policy, Mapping)
        and policy.get("training_wall_clock_direct_comparison_allowed") is False
        and policy.get("training_throughput_direct_comparison_allowed") is False
        and policy.get("cost_efficiency_ranking_allowed") is False
        and policy.get("quality_or_generation_advantage_claim_allowed") is False,
        "runtime-strict guard claim policy differs",
    )
    _require_false_fields(
        report.get("claim_boundary"),
        fields={
            "gpu_execution_allowed",
            "training_launch_allowed",
            "sampling_launch_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
            "quality_claim_allowed",
            "broad_generation_superiority_claim_allowed",
        },
        label="runtime-strict guard claim boundary",
    )
    return {"identity": dict(identity), "decision": report["decision"]}


def _validate_standing_authorization(report: Mapping[str, Any]) -> None:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == "cofitok_standing_experiment_authorization_record"
        and report.get("status") == "active",
        "standing authorization contract differs",
    )
    boundaries = report.get("preserved_safety_boundaries")
    required = {
        "exact_revision_stage_and_output_binding_required",
        "formal_remote_checkout_must_not_be_modified",
        "independent_clean_checkout_required",
        "locked_evidence_must_not_be_overwritten",
        "stage_must_remain_non_authorizing_when_protocol_declares_non_authorizing",
        "unrelated_project_processes_must_not_be_modified",
    }
    _require(
        isinstance(boundaries, Mapping), "standing authorization boundaries missing"
    )
    _require(
        all(boundaries.get(key) is True for key in required),
        "standing authorization safety boundary differs",
    )


def _validate_quality_result(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == QUALITY_RESULT_ROLE
        and report.get("status") == "completed"
        and report.get("stage") == "stability_quality_bridge",
        "quality bridge result contract differs",
    )
    _git(report.get("git"), expected=TRAINING_GIT, label="quality bridge result")
    _require(
        report.get("authorization_boundary") == QUALITY_RESULT_BOUNDARY,
        "quality bridge result authorization boundary differs",
    )
    screen = report.get("quality_screen")
    _require(isinstance(screen, Mapping), "quality bridge quality screen missing")
    rows = screen.get("checks")
    _require(isinstance(rows, list), "quality bridge checks missing")
    indexed: dict[str, bool] = {}
    failed: list[str] = []
    for row in rows:
        _require(
            isinstance(row, Mapping)
            and isinstance(row.get("name"), str)
            and row.get("passed") in {True, False},
            "quality bridge check is malformed",
        )
        name = str(row["name"])
        _require(name not in indexed, "quality bridge check is duplicated")
        indexed[name] = bool(row["passed"])
        if row["passed"] is False:
            failed.append(name)
    _require(set(indexed) == EXPECTED_CHECKS, "quality bridge check set differs")
    _require(
        screen.get("failed_checks") == failed, "quality bridge failed checks differ"
    )
    expected_status = "pass" if not failed else "hold"
    _require(
        screen.get("status") == expected_status, "quality bridge screen status differs"
    )
    _require(
        screen.get("non_authorizing") is True, "quality bridge screen is authorizing"
    )
    return {
        "identity": dict(identity),
        "status": expected_status,
        "failed_checks": failed,
        "checks": indexed,
    }


def _validate_followup(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        report.get("schema_version") == 2
        and report.get("role") == FOLLOWUP_ROLE
        and report.get("status") == "completed",
        "exposure-aware follow-up decision contract differs",
    )
    _git(
        report.get("decision_builder_git"),
        expected=EXPOSURE_GIT,
        label="follow-up decision builder",
    )
    quality_git = dict(TRAINING_GIT)
    quality_git.pop("tree")
    observed_quality_git = report.get("quality_bridge_execution_git")
    _require(
        isinstance(observed_quality_git, Mapping)
        and {
            "revision": observed_quality_git.get("revision"),
            "branch": observed_quality_git.get("branch"),
            "tracked_dirty": observed_quality_git.get(
                "tracked_dirty", observed_quality_git.get("dirty")
            ),
        }
        == quality_git,
        "follow-up quality execution Git differs",
    )
    sources = report.get("source_reports")
    _require(isinstance(sources, Mapping), "follow-up source reports missing")
    _require(
        _identity(
            sources.get("quality_bridge_result"), label="follow-up quality result"
        )
        == quality["identity"],
        "follow-up decision binds another quality result",
    )
    terminal = report.get("terminal_quality")
    _require(isinstance(terminal, Mapping), "follow-up terminal quality missing")
    _require(
        terminal.get("status") == quality["status"]
        and terminal.get("failed_checks") == quality["failed_checks"],
        "follow-up terminal quality differs",
    )
    replay = report.get("source_replay")
    _require(
        isinstance(replay, Mapping)
        and replay.get("quality_bridge_result_rebuilt_byte_equivalent") is True
        and replay.get("physical_checkpoint_sample_and_real_set_reverified") is True
        and replay.get("milestone_source_reports_reverified") is True,
        "follow-up source replay differs",
    )
    exposure = report.get("training_exposure")
    _require(
        isinstance(exposure, Mapping)
        and exposure.get("terminal_result_content_bound") is True
        and exposure.get("terminal_checkpoint_binding_verified") is True,
        "follow-up training exposure binding differs",
    )
    _require(
        report.get("authorization_boundary") == FOLLOWUP_BOUNDARY,
        "follow-up authorization boundary differs",
    )
    recommendation = report.get("recommended_next_stage")
    _require(isinstance(recommendation, Mapping), "follow-up recommendation missing")
    for key in (
        "execution_ready",
        "gpu_execution_allowed",
        "full_300k_launch_allowed",
        "release_authorization_allowed",
    ):
        _require(recommendation.get(key) is False, f"follow-up {key} must be false")
    return {
        "identity": dict(identity),
        "recommendation": copy.deepcopy(dict(recommendation)),
        "failed_checks": list(quality["failed_checks"]),
    }


def _validate_exposure_status(
    report: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
) -> None:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == EXPOSURE_WAITER_ROLE
        and report.get("status") == "completed"
        and report.get("detail") == "exposure_aware_followup_decision_verified",
        "exposure-aware waiter status differs",
    )
    _git(report.get("git"), expected=EXPOSURE_GIT, label="exposure waiter")
    _require(
        report.get("decision_sha256") == decision["identity"]["sha256"],
        "exposure waiter binds another decision",
    )
    recommended = report.get("recommended_next_stage")
    expected = decision["recommendation"]
    _require(
        isinstance(recommended, Mapping)
        and recommended.get("id") == expected.get("id")
        and recommended.get("category") == expected.get("category")
        and recommended.get("execution_ready") is False,
        "exposure waiter recommendation differs",
    )
    scope = report.get("scope")
    _require_false_fields(
        scope,
        fields={
            "gpu_use_allowed",
            "training_launch_allowed",
            "sampling_launch_allowed",
            "evaluation_launch_allowed",
            "promotion_allowed",
            "release_allowed",
            "full_300k_launch_allowed",
            "process_signals_allowed",
        },
        label="exposure waiter scope",
    )


def _validate_terminal_guard(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    quality: Mapping[str, Any],
    runtime_guard_identity: Mapping[str, Any],
) -> dict[str, Any]:
    status = report.get("status")
    decision = report.get("decision")
    _require(
        report.get("schema_version") == 1
        and report.get("role") == TERMINAL_GUARD_ROLE
        and status in TERMINAL_DECISIONS
        and decision == TERMINAL_DECISIONS[status],
        "terminal system guard contract differs",
    )
    scope = report.get("scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("training_git")
        == {
            "revision": TRAINING_GIT["revision"],
            "branch": TRAINING_GIT["branch"],
            "tracked_dirty": False,
        },
        "terminal system guard training identity differs",
    )
    sources = report.get("sources")
    _require(isinstance(sources, Mapping), "terminal system guard sources missing")
    _require(
        _identity(sources.get("quality_bridge_result"), label="terminal guard quality")
        == quality["identity"],
        "terminal system guard binds another quality result",
    )
    _require(
        _identity(
            sources.get("runtime_compute_claim_guard"), label="terminal runtime guard"
        )
        == dict(runtime_guard_identity),
        "terminal system guard binds another runtime guard",
    )
    screen = report.get("evidence", {}).get("quality_screen")
    _require(
        isinstance(screen, Mapping)
        and screen.get("status") == quality["status"]
        and screen.get("failed_checks") == quality["failed_checks"],
        "terminal system guard quality screen differs",
    )
    policy = report.get("claim_policy")
    _require(
        isinstance(policy, Mapping)
        and policy.get("larger_training_launch_allowed") is False
        and policy.get("inference_export_authorization_allowed") is False
        and policy.get("release_authorization_allowed") is False
        and policy.get("cross_tier_numeric_ranking_allowed") is False
        and policy.get("broad_generation_superiority_claim_allowed") is False
        and policy.get("sota_claim_allowed") is False,
        "terminal system guard claim boundary differs",
    )
    _require_false_fields(
        report.get("claim_boundary"),
        fields={
            "training_launch_allowed",
            "gpu_execution_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
        },
        label="terminal system guard claim boundary",
    )
    return {
        "identity": dict(identity),
        "status": str(status),
        "decision": str(decision),
    }


def _validate_terminal_guard_status(
    report: Mapping[str, Any],
    *,
    guard: Mapping[str, Any],
) -> None:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == TERMINAL_GUARD_WAITER_ROLE
        and report.get("status") == "completed"
        and report.get("detail") == "terminal_system_claim_guard_source_revalidated",
        "terminal system guard waiter status differs",
    )
    _git(report.get("git"), expected=TERMINAL_GUARD_GIT, label="terminal guard waiter")
    _require(
        _identity(report.get("guard"), label="terminal guard waiter output")
        == guard["identity"]
        and report.get("guard_status") == guard["status"]
        and report.get("guard_decision") == guard["decision"],
        "terminal guard waiter binding differs",
    )
    _require_false_fields(
        report.get("authorization_boundary"),
        fields={
            "full_300k_launch_allowed",
            "gpu_execution_allowed",
            "inference_export_authorization_allowed",
            "process_signals_allowed",
            "promotion_or_release_allowed",
            "sampling_launch_allowed",
            "training_launch_allowed",
            "upstream_decisions_modified",
        },
        label="terminal guard waiter authorization boundary",
    )


def _validate_comparison(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    quality: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == COMPARISON_ROLE
        and report.get("status") == guard["status"]
        and report.get("decision") == guard["decision"],
        "quality bridge comparison contract differs",
    )
    sources = report.get("source_reports")
    _require(isinstance(sources, Mapping), "comparison source reports missing")
    _require(
        _identity(sources.get("terminal_system_claim_guard"), label="comparison guard")
        == guard["identity"]
        and _identity(sources.get("quality_bridge_result"), label="comparison quality")
        == quality["identity"],
        "comparison source binding differs",
    )
    policy = report.get("comparison_policy")
    _require(
        isinstance(policy, Mapping)
        and policy.get("primary_direct_tier") == "matched_training_direct"
        and policy.get("external_context_tier") == "official_pretrained_contextual"
        and policy.get("cross_tier_numeric_ranking_allowed") is False
        and policy.get("compute_matched_claim_allowed") is False
        and policy.get("broad_generation_superiority_claim_allowed") is False
        and policy.get("sota_claim_allowed") is False,
        "comparison tier boundary differs",
    )
    _require_false_fields(
        report.get("authorization_boundary"),
        fields={
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "release_authorization_allowed",
            "inference_export_authorization_allowed",
        },
        label="comparison authorization boundary",
    )
    direct = report.get("matched_training_rows")
    contextual = report.get("official_context_rows")
    _require(
        isinstance(direct, list)
        and len(direct) == 2
        and {row.get("comparison_tier") for row in direct if isinstance(row, Mapping)}
        == {"matched_training_direct"}
        and isinstance(contextual, list)
        and len(contextual) == 3
        and {
            row.get("comparison_tier") for row in contextual if isinstance(row, Mapping)
        }
        == {"official_pretrained_contextual"},
        "comparison row tiers differ",
    )
    return {"identity": dict(identity)}


def _validate_comparison_status(
    report: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any],
    guard: Mapping[str, Any],
    deployment_identity: Mapping[str, Any],
) -> None:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == COMPARISON_WAITER_ROLE
        and report.get("status") == "pass"
        and report.get("terminal_status") == guard["status"],
        "comparison waiter status differs",
    )
    expected = report.get("expected")
    _require(
        isinstance(expected, Mapping), "comparison waiter expected context missing"
    )
    _git(
        expected.get("project_git"), expected=COMPARISON_GIT, label="comparison waiter"
    )
    _require(
        _identity(report.get("deployment_receipt"), label="comparison deployment")
        == dict(deployment_identity),
        "comparison deployment binding differs",
    )
    outputs = report.get("comparison")
    _require(isinstance(outputs, Mapping), "comparison waiter outputs missing")
    _require(
        _identity(outputs.get("json"), label="comparison waiter JSON")
        == comparison["identity"],
        "comparison waiter binds another JSON report",
    )
    scope = report.get("scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("cpu_only") is True
        and scope.get("diagnostic_non_authorizing") is True,
        "comparison waiter scope differs",
    )
    _require_false_fields(
        scope,
        fields={
            "gpu_execution_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "inference_export_authorization_allowed",
            "process_signals_allowed",
            "promotion_authorization_allowed",
            "release_authorization_allowed",
            "sampling_launch_allowed",
            "training_launch_allowed",
        },
        label="comparison waiter scope",
    )


def _validate_completion(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    quality: Mapping[str, Any],
    guard: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    advantage = report.get("generation_advantage_proven")
    _require(
        report.get("schema_version") == 1
        and report.get("role") == COMPLETION_ROLE
        and report.get("status") == "pass"
        and report.get("terminal_status") == guard["status"]
        and report.get("terminal_decision") == guard["decision"]
        and advantage is (guard["status"] == "pass"),
        "terminal completion audit contract differs",
    )
    scope = report.get("scope")
    _require(
        isinstance(scope, Mapping) and scope.get("training_git") == TRAINING_GIT,
        "terminal completion training identity differs",
    )
    sources = report.get("sources")
    _require(isinstance(sources, Mapping), "terminal completion sources missing")
    _require(
        _identity(sources.get("terminal_system_claim_guard"), label="completion guard")
        == guard["identity"]
        and _identity(
            sources.get("quality_bridge_comparison"), label="completion comparison"
        )
        == comparison["identity"]
        and _identity(sources.get("quality_bridge_result"), label="completion quality")
        == quality["identity"],
        "terminal completion source binding differs",
    )
    _require(
        report.get("authorization_boundary")
        and report.get("claim_policy", {}).get("full_300k_launch_allowed") is False
        and report.get("claim_policy", {}).get("promotion_or_release_allowed") is False,
        "terminal completion claim policy differs",
    )
    _require_false_fields(
        report.get("authorization_boundary"),
        fields={
            "gpu_execution_allowed",
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "release_authorization_allowed",
            "inference_export_authorization_allowed",
            "process_signals_allowed",
        },
        label="terminal completion authorization boundary",
    )
    return {
        "identity": dict(identity),
        "terminal_status": guard["status"],
        "terminal_decision": guard["decision"],
        "generation_advantage_proven": bool(advantage),
    }


def _validate_completion_status(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    completion: Mapping[str, Any],
    deployment_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == COMPLETION_WAITER_ROLE
        and report.get("status") == "pass"
        and report.get("detail") == "terminal_completion_audit_revalidated"
        and report.get("terminal_status") == completion["terminal_status"]
        and report.get("generation_advantage_proven")
        is completion["generation_advantage_proven"],
        "terminal completion waiter status differs",
    )
    _git(report.get("git"), expected=COMPLETION_GIT, label="completion waiter")
    _require(
        _identity(report.get("deployment_receipt"), label="completion deployment")
        == dict(deployment_identity),
        "completion deployment binding differs",
    )
    audit = report.get("audit")
    _require(isinstance(audit, Mapping), "completion waiter audit binding missing")
    _require(
        _identity(audit.get("identity"), label="completion waiter audit")
        == completion["identity"],
        "completion waiter binds another audit",
    )
    scope = report.get("scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("cpu_only_evidence_replay") is True
        and scope.get("diagnostic_non_authorizing") is True,
        "completion waiter scope differs",
    )
    _require_false_fields(
        scope,
        fields={
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "release_authorization_allowed",
            "inference_export_authorization_allowed",
            "process_signals_allowed",
            "upstream_decisions_modified",
        },
        label="completion waiter scope",
    )
    return {"identity": dict(identity)}


def _validate_conjunct(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    completion: Mapping[str, Any],
    completion_status: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        report.get("schema_version") == 1
        and report.get("role") == CONJUNCT_ROLE
        and report.get("status") == "pass"
        and report.get("terminal_status") == completion["terminal_status"]
        and report.get("terminal_decision") == completion["terminal_decision"]
        and report.get("generation_advantage_proven")
        is completion["generation_advantage_proven"]
        and report.get("training_identity")
        == {
            "revision": TRAINING_GIT["revision"],
            "branch": TRAINING_GIT["branch"],
        },
        "runtime-strict conjunct contract differs",
    )
    sources = report.get("sources")
    _require(isinstance(sources, Mapping), "runtime-strict conjunct sources missing")
    _require(
        _identity(
            sources.get("terminal_completion_audit"),
            label="conjunct completion audit",
        )
        == completion["identity"]
        and _identity(
            sources.get("terminal_completion_waiter_status"),
            label="conjunct completion status",
        )
        == completion_status["identity"],
        "runtime-strict conjunct terminal binding differs",
    )
    policy = report.get("claim_policy")
    _require(
        isinstance(policy, Mapping)
        and policy.get("runtime_strict_comparator_passed") is True
        and policy.get("strict_recovery_binding_verified") is True
        and policy.get("training_wall_clock_direct_comparison_allowed") is False
        and policy.get("training_throughput_direct_comparison_allowed") is False
        and policy.get("cost_efficiency_ranking_allowed") is False
        and policy.get("full_training_launch_allowed") is False
        and policy.get("full_300k_launch_allowed") is False
        and policy.get("promotion_or_release_allowed") is False,
        "runtime-strict conjunct claim policy differs",
    )
    _require_false_fields(
        report.get("authorization_boundary"),
        fields={
            "gpu_execution_allowed",
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "release_authorization_allowed",
            "inference_export_authorization_allowed",
            "process_signals_allowed",
            "upstream_decisions_modified",
            "runtime_direct_ranking_allowed",
        },
        label="runtime-strict conjunct authorization boundary",
    )
    return {
        "identity": dict(identity),
        "terminal_status": completion["terminal_status"],
        "terminal_decision": completion["terminal_decision"],
        "generation_advantage_proven": completion["generation_advantage_proven"],
    }


def _validate_conjunct_status(
    report: Mapping[str, Any],
    *,
    conjunct: Mapping[str, Any],
    deployment_identity: Mapping[str, Any],
) -> None:
    output = report.get("conjunct")
    _require(
        report.get("schema_version") == 1
        and report.get("role") == CONJUNCT_WAITER_ROLE
        and report.get("status") == "pass"
        and isinstance(output, Mapping)
        and output.get("terminal_status") == conjunct["terminal_status"]
        and output.get("generation_advantage_proven")
        is conjunct["generation_advantage_proven"],
        "runtime-strict conjunct waiter status differs",
    )
    _git(report.get("git"), expected=CONJUNCT_GIT, label="conjunct waiter")
    _require(
        _identity(report.get("deployment_receipt"), label="conjunct deployment")
        == dict(deployment_identity),
        "conjunct deployment binding differs",
    )
    _require(
        _identity(output.get("identity"), label="conjunct waiter output")
        == conjunct["identity"],
        "conjunct waiter binds another output",
    )
    scope = report.get("scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("cpu_only") is True
        and scope.get("diagnostic_non_authorizing") is True,
        "conjunct waiter scope differs",
    )
    _require_false_fields(
        scope,
        fields={
            "gpu_execution_allowed",
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "release_authorization_allowed",
            "inference_export_authorization_allowed",
            "training_process_signals_allowed",
            "unrelated_process_signals_allowed",
            "source_waiter_signals_allowed",
            "upstream_decisions_modified",
        },
        label="conjunct waiter scope",
    )


def _validate_interlock(
    receipt_identity: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    paths: RouteSourcePaths,
    marker_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("role") == INTERLOCK_ROLE
        and receipt.get("status") == "pass"
        and receipt.get("decision")
        == "legacy_v1_gpu_route_consumers_superseded_fail_closed"
        and receipt.get("generation_advantage_proven") is False,
        "terminal route supersession interlock contract differs",
    )
    scope = receipt.get("scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("cpu_only") is True
        and scope.get("diagnostic_non_authorizing") is True
        and scope.get("static_filesystem_interlock_only") is True,
        "terminal route interlock scope differs",
    )
    _require_false_fields(
        scope,
        fields={
            "gpu_execution_allowed",
            "new_gpu_supervisor_launch_allowed",
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "release_authorization_allowed",
            "inference_export_authorization_allowed",
            "process_signals_allowed",
            "upstream_evidence_modified",
        },
        label="terminal route interlock scope",
    )
    marker_paths = {
        "factorization": paths.factorization_marker,
        "conditioning": paths.conditioning_marker,
        "random_token": paths.random_token_marker,
    }
    if os.name == "posix":
        _require(
            stat.S_IMODE(paths.factorization_marker.parent.stat().st_mode) == 0o555,
            "factorization interlock directory mode differs",
        )
        _require(
            stat.S_IMODE(paths.conditioning_marker.parent.stat().st_mode) == 0o555,
            "conditioning interlock directory mode differs",
        )
        for name, marker in marker_paths.items():
            _require(
                stat.S_IMODE(marker.stat().st_mode) == 0o444,
                f"{name} marker mode differs",
            )
        _require(
            stat.S_IMODE(Path(receipt_identity["path"]).stat().st_mode) == 0o444,
            "interlock receipt mode differs",
        )
    factorization_output = paths.factorization_marker.parent.with_suffix("")
    conditioning_output = paths.conditioning_marker.parent.with_suffix("")
    _require(
        not factorization_output.exists(),
        "legacy factorization output root exists despite supersession",
    )
    _require(
        not conditioning_output.exists(),
        "legacy conditioning output root exists despite supersession",
    )
    _require(
        paths.random_token_marker.is_file() and not paths.random_token_marker.is_dir(),
        "legacy random-token marker no longer blocks the output root",
    )
    return {
        "receipt": dict(receipt_identity),
        "markers": {
            name: dict(identity) for name, identity in marker_identities.items()
        },
        "legacy_routes_remain_fail_closed": True,
    }


def select_route(
    *,
    failed_checks: list[str],
    recommendation: Mapping[str, Any],
    terminal_status: str,
    generation_advantage_proven: bool,
) -> dict[str, Any]:
    failed = set(failed_checks)
    common_requirements = {
        "future_executor_must_be_built_after_this_selection": True,
        "new_versioned_control_root_required": True,
        "new_versioned_output_root_required": True,
        "shared_gpu_exclusivity_required": True,
        "terminal_guard_v2_runtime_strict_required": True,
        "legacy_interlocks_must_remain_immutable": True,
        "legacy_supervisors_must_not_be_restarted": True,
        "full_300k_launch_allowed": False,
        "route_execution_authorized": False,
    }
    if failed and failed <= MATCHED_QUALITY_CHECKS:
        _require(
            recommendation.get("id")
            == "run_matched_factorization_quality_regression_probe"
            and recommendation.get("category") == "matched_quality_regression"
            and recommendation.get("trigger", {}).get("failed_checks")
            == sorted(failed),
            "matched-quality recommendation differs",
        )
        _require(
            terminal_status == "hold" and generation_advantage_proven is False,
            "matched-quality route requires a terminal hold",
        )
        return {
            "selected": True,
            "id": "factorization_quality_regression_diagnostic_runtime_strict",
            "category": "matched_quality_regression",
            "reason": "exact_matched_quality_only_regression",
            "failed_checks": sorted(failed),
            "upstream_recommendation": copy.deepcopy(dict(recommendation)),
            "future_executor_requirements": {
                **common_requirements,
                "diagnostic_family": "factorization_only",
            },
        }
    if failed == {"class_fidelity"}:
        _require(
            recommendation.get("id") == "run_class_conditioning_fidelity_diagnostic"
            and recommendation.get("category") == "class_conditioning_recovery"
            and recommendation.get("trigger", {}).get("failed_checks")
            == ["class_fidelity"],
            "class-only recommendation differs",
        )
        _require(
            terminal_status == "hold" and generation_advantage_proven is False,
            "class-only route requires a terminal hold",
        )
        return {
            "selected": True,
            "id": "conditioning_fidelity_diagnostic_runtime_strict",
            "category": "class_conditioning_recovery",
            "reason": "exact_class_fidelity_only_failure",
            "failed_checks": ["class_fidelity"],
            "upstream_recommendation": copy.deepcopy(dict(recommendation)),
            "future_executor_requirements": {
                **common_requirements,
                "diagnostic_family": "conditioning_only",
            },
        }
    return {
        "selected": False,
        "id": "no_gpu_route",
        "category": "fail_closed_no_gpu_route",
        "reason": "terminal_outcome_is_not_an_exact_supported_diagnostic_route",
        "failed_checks": list(failed_checks),
        "upstream_recommendation": copy.deepcopy(dict(recommendation)),
        "future_executor_requirements": common_requirements,
    }


def build_route_selection(
    *,
    paths: RouteSourcePaths,
    expected_hashes: Mapping[str, str],
    selector_git: Mapping[str, Any],
    selector_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    json_paths = paths.json_paths()
    identities: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for name, path in json_paths.items():
        expected = expected_hashes.get(name)
        identities[name], reports[name] = _bound_json(
            path,
            label=name.replace("_", " "),
            expected_sha256=expected,
        )

    _validate_standing_authorization(reports["standing_authorization"])
    pair = _validate_pair_monitor(identities["pair_monitor"], reports["pair_monitor"])
    runtime_guard = _validate_runtime_guard(
        identities["runtime_guard"],
        reports["runtime_guard"],
        pair_monitor_identity=identities["pair_monitor"],
    )
    quality = _validate_quality_result(
        identities["quality_bridge_result"], reports["quality_bridge_result"]
    )
    decision = _validate_followup(
        identities["followup_decision"],
        reports["followup_decision"],
        quality=quality,
    )
    _validate_exposure_status(reports["exposure_waiter_status"], decision=decision)
    _require(
        reports["exposure_deployment_receipt"].get("status") == "pass"
        and reports["exposure_deployment_receipt"].get("role")
        == "generation_quality_bridge_exposure_aware_followup_waiter_deployment",
        "exposure deployment receipt differs",
    )
    guard = _validate_terminal_guard(
        identities["terminal_guard"],
        reports["terminal_guard"],
        quality=quality,
        runtime_guard_identity=runtime_guard["identity"],
    )
    _validate_terminal_guard_status(reports["terminal_guard_status"], guard=guard)
    comparison = _validate_comparison(
        identities["comparison"],
        reports["comparison"],
        quality=quality,
        guard=guard,
    )
    _validate_comparison_status(
        reports["comparison_status"],
        comparison=comparison,
        guard=guard,
        deployment_identity=identities["comparison_deployment_receipt"],
    )
    completion = _validate_completion(
        identities["completion"],
        reports["completion"],
        quality=quality,
        guard=guard,
        comparison=comparison,
    )
    completion_status = _validate_completion_status(
        identities["completion_status"],
        reports["completion_status"],
        completion=completion,
        deployment_identity=identities["completion_deployment_receipt"],
    )
    conjunct = _validate_conjunct(
        identities["conjunct"],
        reports["conjunct"],
        completion=completion,
        completion_status=completion_status,
    )
    _validate_conjunct_status(
        reports["conjunct_status"],
        conjunct=conjunct,
        deployment_identity=identities["conjunct_deployment_receipt"],
    )
    interlock = _validate_interlock(
        identities["interlock_receipt"],
        reports["interlock_receipt"],
        paths=paths,
        marker_identities={
            "factorization": identities["factorization_marker"],
            "conditioning": identities["conditioning_marker"],
            "random_token": identities["random_token_marker"],
        },
    )

    route = select_route(
        failed_checks=quality["failed_checks"],
        recommendation=decision["recommendation"],
        terminal_status=conjunct["terminal_status"],
        generation_advantage_proven=conjunct["generation_advantage_proven"],
    )
    source_identities = {
        name: identity
        for name, identity in identities.items()
        if name
        not in {
            "factorization_marker",
            "conditioning_marker",
            "random_token_marker",
        }
    }
    for name, path in json_paths.items():
        _require(
            file_identity(path) == identities[name], f"{name} changed after replay"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "pass",
        "terminal_status": conjunct["terminal_status"],
        "terminal_decision": conjunct["terminal_decision"],
        "generation_advantage_proven": conjunct["generation_advantage_proven"],
        "selection": route,
        "selector": {
            "git": copy.deepcopy(dict(selector_git)),
            "sources": {
                name: copy.deepcopy(dict(identity))
                for name, identity in selector_sources.items()
            },
        },
        "source_evidence": source_identities,
        "pair_monitor": pair,
        "interlock": interlock,
        "scope": copy.deepcopy(SCOPE),
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
        "limitations": [
            (
                "This receipt selects at most one logical diagnostic family. It does not "
                "contain or launch a GPU executor."
            ),
            (
                "A selected route still requires a new source-bound implementation, new "
                "versioned roots, shared GPU exclusivity, and a separate execution receipt."
            ),
            (
                "The legacy factorization, conditioning, and random-token consumers remain "
                "permanently superseded by immutable filesystem interlocks."
            ),
            (
                "No outcome from this receipt authorizes 300K training, promotion, export, "
                "release, process signaling, broad superiority, or SOTA claims."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic CPU-only, permanently non-authorizing runtime-strict "
            "terminal route-selection receipt."
        )
    )
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--factorization-marker", type=Path, required=True)
    parser.add_argument("--conditioning-marker", type=Path, required=True)
    parser.add_argument("--random-token-marker", type=Path, required=True)
    parser.add_argument("--expected-hashes-json", type=Path, required=True)
    parser.add_argument("--selector-git-json", type=Path, required=True)
    parser.add_argument("--selector-sources-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = RouteSourcePaths(
        quality_root=args.quality_output_root.resolve(),
        standing_authorization=args.standing_authorization.resolve(),
        factorization_marker=args.factorization_marker.resolve(),
        conditioning_marker=args.conditioning_marker.resolve(),
        random_token_marker=args.random_token_marker.resolve(),
    )
    expected_hashes = read_json_object(
        args.expected_hashes_json, name="route selector expected hashes"
    )
    selector_git = read_json_object(args.selector_git_json, name="route selector Git")
    selector_sources = read_json_object(
        args.selector_sources_json, name="route selector sources"
    )
    report = build_route_selection(
        paths=paths,
        expected_hashes=expected_hashes,
        selector_git=selector_git,
        selector_sources=selector_sources,
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(
        {
            "status": report["status"],
            "route": report["selection"]["id"],
            "receipt": identity,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
