from __future__ import annotations

import copy
from typing import Any, Mapping

from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    EXPECTED_CHECKS as FOLLOWUP_EXPECTED_CHECKS,
    FOLLOWUP_DECISION_ROLE,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
)
from cofitok.generation_pair import generation_pair_contract


PROBE_SCHEMA_VERSION = 1
PROBE_ROLE = "generation_conditioning_ranking_four_arm_probe_preparation"
PROBE_SCOPE = "imagenet256_10pct_four_arm_class_ranking_probe1k_only"
PROBE_STAGE = "conditioning_ranking_four_arm_probe1k_v1"
PROBE_AUTHORIZATION_TEXT = (
    "Approve the non-authorizing four-arm 1000-step "
    "class-conditioning-ranking probe only."
)
EXECUTION_AUTHORIZATION_ROLE = (
    "generation_conditioning_ranking_probe_execution_authorization"
)
EXECUTION_AUTHORIZATION_MODE = "active_standing_experiment_authorization"
STANDING_AUTHORIZATION_ROLE = "cofitok_standing_experiment_authorization_record"
STANDING_AUTHORIZATION_TEXT = "之后不要我授权你直接运行需要的实验"
STANDING_AUTHORIZATION_INTERPRETATION = (
    "Authorize Codex to directly run future experiments that are necessary for "
    "the active CoFiTok long-term objective without requesting per-stage approval."
)
STANDING_AUTHORIZATION_SAFETY_BOUNDARIES = {
    "unrelated_project_processes_must_not_be_modified": True,
    "formal_remote_checkout_must_not_be_modified": True,
    "locked_evidence_must_not_be_overwritten": True,
    "independent_clean_checkout_required": True,
    "exact_revision_stage_and_output_binding_required": True,
    "stage_must_remain_non_authorizing_when_protocol_declares_non_authorizing": True,
}
FOLLOWUP_DECISION_ID = "run_class_conditioning_fidelity_diagnostic"
FOLLOWUP_DECISION_CATEGORY = "class_conditioning_recovery"
FOLLOWUP_DECISION_BUILDER_GIT = {
    "revision": "cd78a348769f0efad0d42de063e5b0943444a29b",
    "branch": "analysis/generation-quality-bridge-mixed-route-v3-20260822",
    "tracked_dirty": False,
}
QUALITY_BRIDGE_EXECUTION_GIT = {
    "revision": "cf0e5faa94bf4ab38d947b921935b3b765b5537a",
    "branch": "scale/generation-stability-quality-bridge-100k",
    "tracked_dirty": False,
}
QUALITY_BRIDGE_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_base128_quality_bridge_v1"
)
QUALITY_BRIDGE_RESULT_ROLE = "stability_full_data_quality_bridge_result"
QUALITY_BRIDGE_RESULT_STAGE = "stability_quality_bridge"
TERMINAL_SYSTEM_GUARD_ROLE = "generation_terminal_system_claim_guard"
TERMINAL_SYSTEM_GUARD_WAITER_ROLE = (
    "generation_terminal_system_claim_guard_waiter"
)
TERMINAL_SYSTEM_GUARD_WAITER_DETAIL = (
    "terminal_system_claim_guard_source_revalidated"
)
TERMINAL_SYSTEM_GUARD_WAITER_GIT = {
    "revision": "4087f4293e3fd197c81cbfa9029f3d6083654415",
    "tree": "7d5acbe2542f5c0584a7fd847fdb806e6757ba86",
    "branch": "analysis/generation-terminal-classifier-integrity-v1-20260822",
    "tracked_dirty": False,
}
TERMINAL_SYSTEM_GUARD_WAITER_AUTHORIZATION_BOUNDARY = {
    "cpu_only_evidence_binding_allowed": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "promotion_or_release_allowed": False,
    "process_signals_allowed": False,
    "upstream_decisions_modified": False,
}
TERMINAL_SYSTEM_GUARD_CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "replaces_bound_source_reports": False,
    "visual_audit_is_quantitative_quality_evidence": False,
    "absolute_usability_claim_allowed": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "sota_claim_allowed": False,
}
REQUESTED_CLASS_VISUAL_AUDIT_WAITER_ROLE = (
    "quality_bridge_terminal_requested_class_visual_audit_waiter"
)
REQUESTED_CLASS_VISUAL_AUDIT_WAITER_DETAIL = (
    "terminal_visual_audit_source_revalidated"
)
REQUESTED_CLASS_VISUAL_AUDIT_WAITER_GIT = {
    "revision": "c1abf65fdafb8e198a8f1ac59c83b3038a9702b5",
    "tree": "3362a6dc939ae5d907103211db41eaa88851f1d2",
    "branch": "scale/generation-terminal-visual-audit-waiter-v1",
    "tracked_dirty": False,
}
REQUESTED_CLASS_VISUAL_AUDIT_WAITER_AUTHORIZATION_BOUNDARY = {
    "cpu_only_visual_diagnostic_allowed": True,
    "gpu_use_allowed": False,
    "training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "quality_bridge_or_followup_decision_modified": False,
}
REQUESTED_CLASS_VISUAL_AUDIT_ROLE = "generation_requested_class_visual_audit"
REQUESTED_CLASS_VISUAL_AUDIT_CLAIM_BOUNDARY = {
    "visual_diagnostic_only": True,
    "quantitative_generation_metric": False,
    "replaces_class_fidelity_evaluation": False,
    "replaces_fid_or_distribution_metrics": False,
    "replaces_frozen_promotion_gate": False,
    "training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
}
RANKING_FIELDS = (
    "class_conditioning_ranking_weight",
    "class_conditioning_ranking_start_step",
    "class_conditioning_ranking_warmup_steps",
    "class_conditioning_ranking_batch_fraction",
    "class_conditioning_ranking_margin",
    "class_conditioning_ranking_wrong_label_offset",
    "class_conditioning_ranking_min_timestep",
)
CONTROL_RANKING_CONFIG = {
    "class_conditioning_ranking_weight": 0.0,
    "class_conditioning_ranking_start_step": 0,
    "class_conditioning_ranking_warmup_steps": 0,
    "class_conditioning_ranking_batch_fraction": 0.0625,
    "class_conditioning_ranking_margin": 0.0,
    "class_conditioning_ranking_wrong_label_offset": 1,
    "class_conditioning_ranking_min_timestep": 0,
}
RANKED_RANKING_CONFIG = {
    "class_conditioning_ranking_weight": 0.05,
    "class_conditioning_ranking_start_step": 100,
    "class_conditioning_ranking_warmup_steps": 200,
    "class_conditioning_ranking_batch_fraction": 0.0625,
    "class_conditioning_ranking_margin": 0.01,
    "class_conditioning_ranking_wrong_label_offset": 500,
    "class_conditioning_ranking_min_timestep": 500,
}
EXECUTION_BOUNDARY = {
    "training_allowed": True,
    "training_runs": [
        "control_cofitok",
        "control_dense_identity",
        "ranked_cofitok",
        "ranked_dense_identity",
    ],
    "steps_per_run": 1_000,
    "dataset": "imagenet_256_10pct",
    "sampling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "release_authorization_allowed": False,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _is_sha256(digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _clean_git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or not all(character in "0123456789abcdef" for character in revision)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def validate_standing_experiment_authorization(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    instruction = authorization.get("instruction")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("role") != STANDING_AUTHORIZATION_ROLE
        or authorization.get("status") != "active"
        or not isinstance(instruction, Mapping)
        or instruction.get("language") != "zh-CN"
        or instruction.get("exact_text") != STANDING_AUTHORIZATION_TEXT
        or instruction.get("interpretation")
        != STANDING_AUTHORIZATION_INTERPRETATION
        or authorization.get("preserved_safety_boundaries")
        != STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing experiment authorization contract differs")
    received_at = instruction.get("received_at")
    if not isinstance(received_at, str) or not received_at:
        raise ValueError("standing experiment authorization timestamp is missing")
    return copy.deepcopy(dict(authorization))


def validate_conditioning_ranking_probe_preparation(
    preparation: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    git = _clean_git(preparation.get("git", {}), label="ranking preparation")
    if (
        preparation.get("schema_version") != PROBE_SCHEMA_VERSION
        or preparation.get("status") != "pass"
        or preparation.get("role") != PROBE_ROLE
        or preparation.get("scope") != PROBE_SCOPE
        or preparation.get("valid") is not True
        or preparation.get("output_root") != expected_output_root
        or preparation.get("execution_boundary") != EXECUTION_BOUNDARY
        or preparation.get("gpu_execution_authorized") is not False
        or git
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
    ):
        raise ValueError("conditioning-ranking preparation contract differs")
    return copy.deepcopy(dict(preparation))


def validate_quality_bridge_result_for_ranking_probe(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    screen = result.get("quality_screen")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "completed"
        or result.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or result.get("stage") != QUALITY_BRIDGE_RESULT_STAGE
        or result.get("git") != QUALITY_BRIDGE_EXECUTION_GIT
        or not isinstance(screen, Mapping)
        or screen.get("status") != "hold"
        or screen.get("failed_checks") != ["class_fidelity"]
    ):
        raise ValueError("quality-bridge class-only terminal result differs")
    checks = screen.get("checks")
    if not isinstance(checks, list):
        raise ValueError("quality-bridge class-only terminal checks are missing")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in checks:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise ValueError("quality-bridge class-only terminal check is malformed")
        name = str(row["name"])
        if name in rows or type(row.get("passed")) is not bool:
            raise ValueError("quality-bridge class-only terminal checks differ")
        rows[name] = row
    if set(rows) != set(FOLLOWUP_EXPECTED_CHECKS) or any(
        row["passed"] is (name == "class_fidelity")
        for name, row in rows.items()
    ):
        raise ValueError("quality-bridge class-only terminal failure is not exact")
    return copy.deepcopy(dict(result))


def validate_class_conditioning_followup_decision(
    decision: Mapping[str, Any],
    *,
    expected_quality_bridge_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recommendation = decision.get("recommended_next_stage")
    terminal = decision.get("terminal_quality")
    sources = decision.get("source_reports")
    claim_policy = decision.get("claim_policy")
    if (
        decision.get("schema_version") != FOLLOWUP_DECISION_SCHEMA_VERSION
        or decision.get("status") != "completed"
        or decision.get("role") != FOLLOWUP_DECISION_ROLE
        or decision.get("decision_builder_git") != FOLLOWUP_DECISION_BUILDER_GIT
        or decision.get("quality_bridge_execution_git")
        != QUALITY_BRIDGE_EXECUTION_GIT
        or decision.get("authorization_boundary") != FOLLOWUP_AUTHORIZATION_BOUNDARY
        or not isinstance(claim_policy, Mapping)
        or claim_policy.get("experiment_selection_only") is not True
        or claim_policy.get("terminal_result_is_promotion_gate") is not False
        or not isinstance(recommendation, Mapping)
        or recommendation.get("id") != FOLLOWUP_DECISION_ID
        or recommendation.get("category") != FOLLOWUP_DECISION_CATEGORY
        or recommendation.get("execution_ready") is not False
        or recommendation.get("gpu_execution_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
        or recommendation.get("release_authorization_allowed") is not False
        or recommendation.get("trigger")
        != {"failed_checks": ["class_fidelity"]}
        or not isinstance(terminal, Mapping)
        or terminal.get("failed_checks") != ["class_fidelity"]
        or not isinstance(sources, Mapping)
        or set(sources)
        != {"quality_bridge_result", "milestones", "terminal_training_exposure"}
    ):
        raise ValueError("quality-bridge class-conditioning follow-up differs")
    quality_identity = _identity(
        sources.get("quality_bridge_result", {}),
        label="quality-bridge result",
    )
    if expected_quality_bridge_result is not None and quality_identity != _identity(
        expected_quality_bridge_result,
        label="expected quality-bridge result",
    ):
        raise ValueError("quality-bridge class-conditioning result binding differs")
    milestones = sources.get("milestones")
    if not isinstance(milestones, Mapping) or set(milestones) != {"50000", "100000"}:
        raise ValueError("quality-bridge class-conditioning milestones differ")
    for step in ("50000", "100000"):
        _identity(milestones[step], label=f"quality-bridge milestone {step}")
    _identity(
        sources.get("terminal_training_exposure", {}),
        label="terminal training exposure",
    )
    checks = terminal.get("checks")
    if not isinstance(checks, list):
        raise ValueError("quality-bridge terminal checks are missing")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in checks:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise ValueError("quality-bridge terminal check is malformed")
        name = str(row["name"])
        if name in rows or type(row.get("passed")) is not bool:
            raise ValueError("quality-bridge terminal checks are not unique and boolean")
        rows[name] = row
    if set(rows) != set(FOLLOWUP_EXPECTED_CHECKS) or any(
        row["passed"] is (name == "class_fidelity")
        for name, row in rows.items()
    ):
        raise ValueError("quality-bridge class-fidelity failure is not exact")
    return {
        "report": copy.deepcopy(dict(decision)),
        "quality_bridge_result": quality_identity,
        "failed_checks": copy.deepcopy(list(terminal["failed_checks"])),
        "decision_builder_git": copy.deepcopy(FOLLOWUP_DECISION_BUILDER_GIT),
    }


def validate_requested_class_visual_audit_evidence(
    status: Mapping[str, Any],
    *,
    status_identity: Mapping[str, Any],
    report: Mapping[str, Any],
    report_identity: Mapping[str, Any],
    expected_quality_bridge_result: Mapping[str, Any],
) -> dict[str, Any]:
    expected_quality = _identity(
        expected_quality_bridge_result,
        label="expected visual-audit quality result",
    )
    expected_status = _identity(
        status_identity,
        label="requested-class visual-audit waiter status",
    )
    expected_report = _identity(
        report_identity,
        label="requested-class visual-audit report",
    )
    expected = status.get("expected")
    quality = status.get("quality_result")
    if (
        status.get("schema_version") != 1
        or status.get("role") != REQUESTED_CLASS_VISUAL_AUDIT_WAITER_ROLE
        or status.get("status") != "completed"
        or status.get("detail") != REQUESTED_CLASS_VISUAL_AUDIT_WAITER_DETAIL
        or status.get("authorization_boundary")
        != REQUESTED_CLASS_VISUAL_AUDIT_WAITER_AUTHORIZATION_BOUNDARY
        or status.get("git") != REQUESTED_CLASS_VISUAL_AUDIT_WAITER_GIT
        or not isinstance(expected, Mapping)
        or expected.get("git") != REQUESTED_CLASS_VISUAL_AUDIT_WAITER_GIT
        or expected.get("quality_result") != expected_quality["path"]
        or not isinstance(quality, Mapping)
        or quality.get("identity") != expected_quality
        or status.get("visual_audit") != expected_report
    ):
        raise ValueError("requested-class visual-audit waiter binding differs")
    panels = report.get("panels")
    sources = report.get("sources")
    if (
        report.get("schema_version") != 1
        or report.get("role") != REQUESTED_CLASS_VISUAL_AUDIT_ROLE
        or report.get("status") != "completed"
        or report.get("claim_boundary") != REQUESTED_CLASS_VISUAL_AUDIT_CLAIM_BOUNDARY
        or report.get("indices") != list(range(16))
        or not isinstance(panels, list)
        or len(panels) != 2
        or not isinstance(sources, Mapping)
        or set(sources) != {"cofitok", "dense_identity"}
    ):
        raise ValueError("requested-class visual-audit report differs")
    for panel_index, panel in enumerate(panels):
        if (
            not isinstance(panel, Mapping)
            or _identity(panel, label="requested-class visual panel")
            != {key: panel[key] for key in ("path", "bytes", "sha256")}
            or panel.get("indices")
            != list(range(panel_index * 8, (panel_index + 1) * 8))
            or panel.get("row_order")
            != ["real_validation", "cofitok", "dense_identity"]
            or panel.get("columns") != 8
        ):
            raise ValueError("requested-class visual-audit panel differs")
    return {
        "status": "completed",
        "quantitative_quality_evidence": False,
        "fixed_indices": list(range(16)),
        "panel_count": 2,
        "report": expected_report,
        "waiter_status": expected_status,
    }


def validate_terminal_system_evidence_for_ranking_probe(
    guard: Mapping[str, Any],
    *,
    guard_identity: Mapping[str, Any],
    guard_status: Mapping[str, Any],
    guard_status_identity: Mapping[str, Any],
    visual_audit_status: Mapping[str, Any],
    visual_audit_status_identity: Mapping[str, Any],
    visual_audit_report: Mapping[str, Any],
    visual_audit_report_identity: Mapping[str, Any],
    expected_quality_bridge_result: Mapping[str, Any],
    expected_failed_checks: list[Any],
) -> dict[str, Any]:
    guard_id = _identity(guard_identity, label="terminal-system claim guard")
    guard_status_id = _identity(
        guard_status_identity,
        label="terminal-system claim guard waiter status",
    )
    quality_id = _identity(
        expected_quality_bridge_result,
        label="expected quality result",
    )
    visual_status_id = _identity(
        visual_audit_status_identity,
        label="requested-class visual-audit waiter status",
    )
    visual = validate_requested_class_visual_audit_evidence(
        visual_audit_status,
        status_identity=visual_status_id,
        report=visual_audit_report,
        report_identity=visual_audit_report_identity,
        expected_quality_bridge_result=quality_id,
    )
    sources = guard.get("sources")
    evidence = guard.get("evidence")
    policy = guard.get("claim_policy")
    boundary = guard.get("claim_boundary")
    scope = guard.get("scope")
    if (
        guard.get("schema_version") != 1
        or guard.get("role") != TERMINAL_SYSTEM_GUARD_ROLE
        or guard.get("status") not in {"pass", "hold"}
        or not isinstance(sources, Mapping)
        or set(sources)
        != {
            "quality_bridge_result",
            "statistical_claim_language_guard",
            "requested_class_visual_audit_waiter_status",
            "runtime_compute_claim_guard",
        }
        or _identity(
            sources.get("quality_bridge_result", {}),
            label="terminal-system quality result",
        )
        != quality_id
        or _identity(
            sources.get("requested_class_visual_audit_waiter_status", {}),
            label="terminal-system visual-audit waiter status",
        )
        != visual_status_id
        or not isinstance(scope, Mapping)
        or scope.get("dataset") != "imagenet_256"
        or scope.get("training_steps_per_method") != 100_000
        or scope.get("quality_output_root") != QUALITY_BRIDGE_OUTPUT_ROOT
        or scope.get("training_git") != QUALITY_BRIDGE_EXECUTION_GIT
        or not isinstance(evidence, Mapping)
        or not isinstance(policy, Mapping)
        or boundary != TERMINAL_SYSTEM_GUARD_CLAIM_BOUNDARY
        or policy.get("terminal_system_evidence_complete") is not True
        or policy.get("requested_class_visual_evidence_available") is not True
        or policy.get("requested_class_visual_evidence_is_quantitative") is not False
        or policy.get("larger_training_launch_allowed") is not False
        or policy.get("inference_export_authorization_allowed") is not False
        or policy.get("release_authorization_allowed") is not False
        or policy.get("broad_generation_superiority_claim_allowed") is not False
    ):
        raise ValueError("terminal-system evidence contract differs")
    _identity(
        sources.get("statistical_claim_language_guard", {}),
        label="terminal statistical claim-language guard",
    )
    _identity(
        sources.get("runtime_compute_claim_guard", {}),
        label="terminal runtime-compute claim guard",
    )
    quality_screen = evidence.get("quality_screen")
    guard_visual = evidence.get("requested_class_visual_audit")
    runtime = evidence.get("runtime_compute")
    classifier_integrity = evidence.get("class_fidelity_classifier_integrity")
    if (
        not isinstance(quality_screen, Mapping)
        or quality_screen.get("failed_checks") != expected_failed_checks
        or quality_screen.get("absolute_quality_passed") is not True
        or int(quality_screen.get("check_count", -1))
        != len(FOLLOWUP_EXPECTED_CHECKS)
        or guard_visual != visual
        or not isinstance(runtime, Mapping)
        or runtime.get("status") != "verified"
        or policy.get("class_fidelity_classifier_physical_integrity_verified")
        is not True
        or not isinstance(classifier_integrity, Mapping)
        or classifier_integrity.get("status") != "verified"
    ):
        raise ValueError("terminal requested-class evidence differs")
    status_expected = guard_status.get("expected")
    status_sources = guard_status.get("sources")
    if (
        guard_status.get("schema_version") != 1
        or guard_status.get("role") != TERMINAL_SYSTEM_GUARD_WAITER_ROLE
        or guard_status.get("status") != "completed"
        or guard_status.get("detail") != TERMINAL_SYSTEM_GUARD_WAITER_DETAIL
        or guard_status.get("authorization_boundary")
        != TERMINAL_SYSTEM_GUARD_WAITER_AUTHORIZATION_BOUNDARY
        or guard_status.get("git") != TERMINAL_SYSTEM_GUARD_WAITER_GIT
        or not isinstance(status_expected, Mapping)
        or status_expected.get("git") != TERMINAL_SYSTEM_GUARD_WAITER_GIT
        or status_expected.get("quality_output_root") != QUALITY_BRIDGE_OUTPUT_ROOT
        or status_expected.get("output") != guard_id["path"]
        or not isinstance(status_sources, Mapping)
        or status_sources.get("quality_result") != quality_id
        or status_sources.get("visual_audit_waiter_status") != visual_status_id
        or guard_status.get("guard") != guard_id
        or guard_status.get("guard_status") != guard.get("status")
        or guard_status.get("guard_decision") != guard.get("decision")
    ):
        raise ValueError("terminal-system guard waiter binding differs")
    if guard_status_id["path"] == guard_id["path"]:
        raise ValueError("terminal-system guard and waiter status paths must differ")
    return copy.deepcopy(dict(guard))


def _ranking_config(config: Mapping[str, Any]) -> dict[str, Any]:
    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        return {}
    return {field: loss.get(field) for field in RANKING_FIELDS}


def _without_probe_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized.pop("name", None)
    loss = normalized.get("loss")
    if isinstance(loss, dict):
        for field in RANKING_FIELDS:
            loss.pop(field, None)
    return normalized


def conditioning_ranking_probe_contract(
    *,
    control_cofitok: dict[str, Any],
    control_dense: dict[str, Any],
    ranked_cofitok: dict[str, Any],
    ranked_dense: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    pairs = {
        "control": generation_pair_contract(control_cofitok, control_dense),
        "ranked": generation_pair_contract(ranked_cofitok, ranked_dense),
    }
    for name, contract in pairs.items():
        issues.extend(f"{name}_pair: {issue}" for issue in contract["issues"])

    for label, config, expected in (
        ("control_cofitok", control_cofitok, CONTROL_RANKING_CONFIG),
        ("control_dense", control_dense, CONTROL_RANKING_CONFIG),
        ("ranked_cofitok", ranked_cofitok, RANKED_RANKING_CONFIG),
        ("ranked_dense", ranked_dense, RANKED_RANKING_CONFIG),
    ):
        actual = _ranking_config(config)
        if actual != expected:
            issues.append(f"{label} ranking config differs from the probe contract")
        runtime = config.get("runtime", {})
        data = config.get("data", {})
        optimization = config.get("optimization", {})
        if runtime.get("steps") != 1_000:
            issues.append(f"{label} runtime.steps is not 1000")
        if data.get("dataset") != "imagenet_256_10pct":
            issues.append(f"{label} dataset is not imagenet_256_10pct")
        if int(data.get("batch_size", 0)) * int(
            optimization.get("gradient_accumulation_steps", 0)
        ) != 64:
            issues.append(f"{label} effective batch size is not 64")

    for method, control, ranked in (
        ("cofitok", control_cofitok, ranked_cofitok),
        ("dense_identity", control_dense, ranked_dense),
    ):
        if _without_probe_fields(control) != _without_probe_fields(ranked):
            issues.append(
                f"{method} control/ranked configs differ outside the ranking fields"
            )

    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "pass" if not issues else "fail",
        "role": PROBE_ROLE,
        "scope": PROBE_SCOPE,
        "valid": not issues,
        "issues": issues,
        "ranking_fields": list(RANKING_FIELDS),
        "control_ranking_config": copy.deepcopy(CONTROL_RANKING_CONFIG),
        "ranked_ranking_config": copy.deepcopy(RANKED_RANKING_CONFIG),
        "pair_contracts": pairs,
        "execution_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
        "authorization_required": True,
        "authorization_source": (
            "active_standing_experiment_authorization_plus_source_bound_"
            "terminal_execution_receipt"
        ),
        "gpu_execution_authorized": False,
    }


def validate_conditioning_ranking_probe_approval(
    approval: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_preparation_sha256: str,
    expected_output_root: str,
) -> dict[str, Any]:
    expected = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "role": "generation_conditioning_ranking_probe_execution_approval",
        "status": "approved",
        "scope": PROBE_SCOPE,
        "user_authorization_text": PROBE_AUTHORIZATION_TEXT,
        "authorized_revision": expected_revision,
        "preparation_report_sha256": expected_preparation_sha256,
        "output_root": expected_output_root,
        "authorization_boundary": EXECUTION_BOUNDARY,
    }
    if dict(approval) != expected:
        raise ValueError("conditioning-ranking probe approval does not match exact scope")
    return copy.deepcopy(expected)


def build_conditioning_ranking_probe_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    followup_decision: Mapping[str, Any],
    followup_decision_identity: Mapping[str, Any],
    quality_bridge_result: Mapping[str, Any],
    quality_bridge_result_identity: Mapping[str, Any],
    terminal_system_guard: Mapping[str, Any],
    terminal_system_guard_identity: Mapping[str, Any],
    terminal_system_guard_status: Mapping[str, Any],
    terminal_system_guard_status_identity: Mapping[str, Any],
    requested_class_visual_audit_status: Mapping[str, Any],
    requested_class_visual_audit_status_identity: Mapping[str, Any],
    requested_class_visual_audit_report: Mapping[str, Any],
    requested_class_visual_audit_report_identity: Mapping[str, Any],
    authorization_git: Mapping[str, Any],
    authorization_tree: str,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    preparation_id = _identity(
        preparation_identity,
        label="conditioning-ranking preparation",
    )
    standing_id = _identity(
        standing_authorization_identity,
        label="standing authorization",
    )
    followup_id = _identity(
        followup_decision_identity,
        label="class-conditioning follow-up decision",
    )
    quality_id = _identity(
        quality_bridge_result_identity,
        label="quality-bridge result",
    )
    terminal_id = _identity(
        terminal_system_guard_identity,
        label="terminal-system claim guard",
    )
    terminal_status_id = _identity(
        terminal_system_guard_status_identity,
        label="terminal-system claim guard waiter status",
    )
    visual_status_id = _identity(
        requested_class_visual_audit_status_identity,
        label="requested-class visual-audit waiter status",
    )
    visual_report_id = _identity(
        requested_class_visual_audit_report_identity,
        label="requested-class visual-audit report",
    )
    builder_git = _clean_git(
        authorization_git,
        label="conditioning-ranking authorization builder",
    )
    if builder_git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("conditioning-ranking authorization builder Git differs")
    if (
        not isinstance(authorization_tree, str)
        or len(authorization_tree) != 40
        or not all(character in "0123456789abcdef" for character in authorization_tree)
        or authorization_tree != expected_tree
    ):
        raise ValueError("conditioning-ranking authorization builder tree differs")
    validated_preparation = validate_conditioning_ranking_probe_preparation(
        preparation,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )
    validated_standing = validate_standing_experiment_authorization(
        standing_authorization
    )
    validate_quality_bridge_result_for_ranking_probe(quality_bridge_result)
    followup = validate_class_conditioning_followup_decision(
        followup_decision,
        expected_quality_bridge_result=quality_id,
    )
    validated_terminal = validate_terminal_system_evidence_for_ranking_probe(
        terminal_system_guard,
        guard_identity=terminal_id,
        guard_status=terminal_system_guard_status,
        guard_status_identity=terminal_status_id,
        visual_audit_status=requested_class_visual_audit_status,
        visual_audit_status_identity=visual_status_id,
        visual_audit_report=requested_class_visual_audit_report,
        visual_audit_report_identity=visual_report_id,
        expected_quality_bridge_result=followup["quality_bridge_result"],
        expected_failed_checks=followup["failed_checks"],
    )
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "role": EXECUTION_AUTHORIZATION_ROLE,
        "status": "authorized",
        "authorization_mode": EXECUTION_AUTHORIZATION_MODE,
        "scope": PROBE_SCOPE,
        "stage": PROBE_STAGE,
        "authorized_revision": expected_revision,
        "authorized_tree": expected_tree,
        "authorized_branch": expected_branch,
        "output_root": expected_output_root,
        "authorization_builder_git": builder_git,
        "authorization_builder_tree": authorization_tree,
        "source_reports": {
            "preparation": preparation_id,
            "standing_authorization": standing_id,
            "quality_bridge_followup_decision": followup_id,
            "terminal_system_claim_guard": terminal_id,
            "terminal_system_claim_guard_waiter_status": terminal_status_id,
            "requested_class_visual_audit_waiter_status": visual_status_id,
            "requested_class_visual_audit_report": visual_report_id,
            "quality_bridge_result": quality_id,
        },
        "standing_authorization": {
            "source": standing_id,
            "validated_record": validated_standing,
        },
        "source_decision": {
            "id": FOLLOWUP_DECISION_ID,
            "category": FOLLOWUP_DECISION_CATEGORY,
            "failed_checks": followup["failed_checks"],
            "decision_builder_git": followup["decision_builder_git"],
            "terminal_system_status": validated_terminal["status"],
            "requested_class_visual_evidence_complete": True,
            "requested_class_visual_audit": {
                "waiter_status": visual_status_id,
                "report": visual_report_id,
            },
        },
        "preparation_contract": {
            "status": validated_preparation["status"],
            "scope": validated_preparation["scope"],
            "valid": validated_preparation["valid"],
        },
        "authorization_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
        "claim_boundary": {
            "diagnostic_non_authorizing": True,
            "training_quality_claim_allowed": False,
            "sample_quality_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
            "followup_training_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
        },
    }


def validate_conditioning_ranking_probe_execution_authorization(
    authorization: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    followup_decision: Mapping[str, Any],
    followup_decision_identity: Mapping[str, Any],
    quality_bridge_result: Mapping[str, Any],
    quality_bridge_result_identity: Mapping[str, Any],
    terminal_system_guard: Mapping[str, Any],
    terminal_system_guard_identity: Mapping[str, Any],
    terminal_system_guard_status: Mapping[str, Any],
    terminal_system_guard_status_identity: Mapping[str, Any],
    requested_class_visual_audit_status: Mapping[str, Any],
    requested_class_visual_audit_status_identity: Mapping[str, Any],
    requested_class_visual_audit_report: Mapping[str, Any],
    requested_class_visual_audit_report_identity: Mapping[str, Any],
    authorization_git: Mapping[str, Any],
    authorization_tree: str,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    expected = build_conditioning_ranking_probe_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_identity,
        standing_authorization=standing_authorization,
        standing_authorization_identity=standing_authorization_identity,
        followup_decision=followup_decision,
        followup_decision_identity=followup_decision_identity,
        quality_bridge_result=quality_bridge_result,
        quality_bridge_result_identity=quality_bridge_result_identity,
        terminal_system_guard=terminal_system_guard,
        terminal_system_guard_identity=terminal_system_guard_identity,
        terminal_system_guard_status=terminal_system_guard_status,
        terminal_system_guard_status_identity=terminal_system_guard_status_identity,
        requested_class_visual_audit_status=requested_class_visual_audit_status,
        requested_class_visual_audit_status_identity=(
            requested_class_visual_audit_status_identity
        ),
        requested_class_visual_audit_report=requested_class_visual_audit_report,
        requested_class_visual_audit_report_identity=(
            requested_class_visual_audit_report_identity
        ),
        authorization_git=authorization_git,
        authorization_tree=authorization_tree,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )
    if dict(authorization) != expected:
        raise ValueError(
            "conditioning-ranking standing execution authorization differs"
        )
    return expected
