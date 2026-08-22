from __future__ import annotations

import copy
from typing import Any, Mapping

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
FOLLOWUP_DECISION_ROLE = "stability_quality_bridge_followup_experiment_decision"
FOLLOWUP_DECISION_ID = "run_class_conditioning_fidelity_diagnostic"
FOLLOWUP_DECISION_CATEGORY = "class_conditioning_recovery"
QUALITY_BRIDGE_EXECUTION_GIT = {
    "revision": "cf0e5faa94bf4ab38d947b921935b3b765b5537a",
    "branch": "scale/generation-stability-quality-bridge-100k",
    "tracked_dirty": False,
}
TERMINAL_SYSTEM_GUARD_ROLE = "generation_terminal_system_claim_guard"
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


def validate_class_conditioning_followup_decision(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    recommendation = decision.get("recommended_next_stage")
    terminal = decision.get("terminal_quality")
    sources = decision.get("source_reports")
    if (
        decision.get("schema_version") != 1
        or decision.get("status") != "completed"
        or decision.get("role") != FOLLOWUP_DECISION_ROLE
        or decision.get("quality_bridge_execution_git")
        != QUALITY_BRIDGE_EXECUTION_GIT
        or not isinstance(recommendation, Mapping)
        or recommendation.get("id") != FOLLOWUP_DECISION_ID
        or recommendation.get("category") != FOLLOWUP_DECISION_CATEGORY
        or recommendation.get("execution_ready") is not False
        or recommendation.get("gpu_execution_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
        or not isinstance(terminal, Mapping)
        or terminal.get("failed_checks") != ["class_fidelity"]
        or not isinstance(sources, Mapping)
    ):
        raise ValueError("quality-bridge class-conditioning follow-up differs")
    quality_identity = _identity(
        sources.get("quality_bridge_result", {}),
        label="quality-bridge result",
    )
    checks = terminal.get("checks")
    if not isinstance(checks, list):
        raise ValueError("quality-bridge terminal checks are missing")
    class_rows = [
        row
        for row in checks
        if isinstance(row, Mapping) and row.get("name") == "class_fidelity"
    ]
    if len(class_rows) != 1 or class_rows[0].get("passed") is not False:
        raise ValueError("quality-bridge class-fidelity failure is not exact")
    return {
        "report": copy.deepcopy(dict(decision)),
        "quality_bridge_result": quality_identity,
        "failed_checks": copy.deepcopy(list(terminal["failed_checks"])),
    }


def validate_terminal_system_evidence_for_ranking_probe(
    guard: Mapping[str, Any],
    *,
    expected_quality_bridge_result: Mapping[str, Any],
    expected_failed_checks: list[Any],
) -> dict[str, Any]:
    sources = guard.get("sources")
    evidence = guard.get("evidence")
    policy = guard.get("claim_policy")
    boundary = guard.get("claim_boundary")
    if (
        guard.get("schema_version") != 1
        or guard.get("role") != TERMINAL_SYSTEM_GUARD_ROLE
        or guard.get("status") not in {"pass", "hold"}
        or not isinstance(sources, Mapping)
        or _identity(
            sources.get("quality_bridge_result", {}),
            label="terminal-system quality result",
        )
        != _identity(
            expected_quality_bridge_result,
            label="expected quality result",
        )
        or not isinstance(evidence, Mapping)
        or not isinstance(policy, Mapping)
        or not isinstance(boundary, Mapping)
        or policy.get("terminal_system_evidence_complete") is not True
        or policy.get("requested_class_visual_evidence_available") is not True
        or policy.get("requested_class_visual_evidence_is_quantitative") is not False
        or policy.get("larger_training_launch_allowed") is not False
        or policy.get("inference_export_authorization_allowed") is not False
        or policy.get("release_authorization_allowed") is not False
        or policy.get("broad_generation_superiority_claim_allowed") is not False
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("gpu_execution_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("terminal-system evidence contract differs")
    quality_screen = evidence.get("quality_screen")
    visual = evidence.get("requested_class_visual_audit")
    if (
        not isinstance(quality_screen, Mapping)
        or quality_screen.get("failed_checks") != expected_failed_checks
        or not isinstance(visual, Mapping)
        or visual.get("status") != "completed"
        or visual.get("quantitative_quality_evidence") is not False
        or int(visual.get("panel_count", -1)) != 2
    ):
        raise ValueError("terminal requested-class evidence differs")
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
    terminal_system_guard: Mapping[str, Any],
    terminal_system_guard_identity: Mapping[str, Any],
    authorization_git: Mapping[str, Any],
    expected_revision: str,
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
    terminal_id = _identity(
        terminal_system_guard_identity,
        label="terminal-system claim guard",
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
    validated_preparation = validate_conditioning_ranking_probe_preparation(
        preparation,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )
    validated_standing = validate_standing_experiment_authorization(
        standing_authorization
    )
    followup = validate_class_conditioning_followup_decision(followup_decision)
    validated_terminal = validate_terminal_system_evidence_for_ranking_probe(
        terminal_system_guard,
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
        "authorized_branch": expected_branch,
        "output_root": expected_output_root,
        "authorization_builder_git": builder_git,
        "source_reports": {
            "preparation": preparation_id,
            "standing_authorization": standing_id,
            "quality_bridge_followup_decision": followup_id,
            "terminal_system_claim_guard": terminal_id,
            "quality_bridge_result": followup["quality_bridge_result"],
        },
        "standing_authorization": {
            "source": standing_id,
            "validated_record": validated_standing,
        },
        "source_decision": {
            "id": FOLLOWUP_DECISION_ID,
            "category": FOLLOWUP_DECISION_CATEGORY,
            "failed_checks": followup["failed_checks"],
            "terminal_system_status": validated_terminal["status"],
            "requested_class_visual_evidence_complete": True,
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
    terminal_system_guard: Mapping[str, Any],
    terminal_system_guard_identity: Mapping[str, Any],
    authorization_git: Mapping[str, Any],
    expected_revision: str,
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
        terminal_system_guard=terminal_system_guard,
        terminal_system_guard_identity=terminal_system_guard_identity,
        authorization_git=authorization_git,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )
    if dict(authorization) != expected:
        raise ValueError(
            "conditioning-ranking standing execution authorization differs"
        )
    return expected
