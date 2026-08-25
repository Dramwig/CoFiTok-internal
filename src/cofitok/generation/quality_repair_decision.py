from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cofitok.generation.quality_repair_artifact import (
    normalize_epsilon_stability_identity,
)
from cofitok.generation.quality_repair_result import (
    EPSILON_STABILITY_RESULT_SCHEMA,
)


POST_DIAGNOSTIC_DECISION_SCHEMA = (
    "cofitok_epsilon_stability_post_diagnostic_decision_v1"
)
POST_DIAGNOSTIC_DECISION_ROLE = (
    "generation_epsilon_stability_post_diagnostic_decision"
)
POST_DIAGNOSTIC_VERIFICATION_SCHEMA = (
    "cofitok_epsilon_stability_post_diagnostic_decision_verification_v1"
)
QUALITY_BRIDGE_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
QUALITY_BRIDGE_BRANCH = "scale/generation-stability-quality-bridge-100k"
METHODS = ("cofitok", "dense_identity")
SOURCE_NAMES = (
    "epsilon_stability_sampling_result",
    "post_reconciliation_decision",
    "cross_protocol_reconciliation",
    "pair_monitor",
    "training_exposure_report",
    "runtime_compute_fairness",
    "terminal_system_claim_guard",
    "terminal_completion_audit",
    "terminal_route_supersession_receipt",
)
POST_DIAGNOSTIC_AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "gpu_execution_allowed": False,
    "separate_source_bound_execution_gate_required": True,
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    return value


def _identity(value: Any, *, label: str) -> dict[str, Any]:
    return normalize_epsilon_stability_identity(value, label=label)


def _git_identity(value: Any, *, label: str) -> dict[str, Any]:
    candidate = _mapping(value, label=label)
    if set(candidate) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{label} Git identity is malformed")
    revision = str(candidate["revision"])
    tree = str(candidate["tree"])
    branch = str(candidate["branch"])
    if (
        len(revision) != 40
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or any(character not in "0123456789abcdef" for character in tree)
        or not branch
        or candidate["tracked_dirty"] is not False
    ):
        raise ValueError(f"{label} Git identity is malformed")
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def _require_false_fields(
    value: Any,
    *,
    fields: tuple[str, ...],
    label: str,
) -> None:
    boundary = _mapping(value, label=label)
    for field in fields:
        if boundary.get(field) is not False:
            raise ValueError(f"{label} permits {field}")


def _validate_training_git(value: Any, *, label: str) -> None:
    git = _mapping(value, label=label)
    if (
        git.get("revision") != QUALITY_BRIDGE_REVISION
        or git.get("branch") != QUALITY_BRIDGE_BRANCH
        or git.get("tracked_dirty", git.get("dirty")) is not False
    ):
        raise ValueError(f"{label} does not bind the matched 100K training")


def _validate_diagnostic(result: Mapping[str, Any]) -> dict[str, Any]:
    claim = _mapping(result.get("claim_boundary"), label="diagnostic claim boundary")
    authorization = _mapping(
        result.get("authorization_boundary"),
        label="diagnostic authorization boundary",
    )
    observations = result.get("observations")
    if (
        result.get("schema") != EPSILON_STABILITY_RESULT_SCHEMA
        or result.get("status") != "pass"
        or result.get("scientific_status") != "screening_only"
        or result.get("selection_status")
        != "no_shared_sampling_recovery_candidate"
        or result.get("selected_case_id") is not None
        or result.get("generation_advantage_proven") is not False
        or claim.get("one_thousand_sample_screening_only") is not True
        or claim.get("min_snr_training_tested") is not False
        or claim.get("cofitok_generation_advantage_claim_allowed") is not False
        or not isinstance(observations, list)
        or len(observations) != 16
    ):
        raise ValueError("epsilon-stability diagnostic outcome differs")
    keys = {
        (str(row.get("case_id")), str(row.get("method")))
        for row in observations
        if isinstance(row, Mapping)
    }
    if (
        len(keys) != 16
        or {method for _, method in keys} != set(METHODS)
        or any(row.get("status") != "pass" for row in observations)
    ):
        raise ValueError("epsilon-stability diagnostic observation matrix differs")
    _require_false_fields(
        authorization,
        fields=(
            "matched_1000_sample_sampling_launch_allowed",
            "independent_matched_10000_confirmation_launch_allowed",
            "training_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_allowed",
            "inference_export_allowed",
            "release_allowed",
            "process_signals_allowed",
        ),
        label="diagnostic authorization boundary",
    )
    return {
        "selection_status": "no_shared_sampling_recovery_candidate",
        "matched_case_count": len(keys) // len(METHODS),
        "observation_count": len(observations),
        "min_snr_training_tested": False,
    }


def _validate_post_reconciliation(decision: Mapping[str, Any]) -> None:
    next_stage = _mapping(
        decision.get("recommended_next_stage"), label="post-reconciliation stage"
    )
    boundary = _mapping(
        decision.get("authorization_boundary"),
        label="post-reconciliation authorization boundary",
    )
    if (
        decision.get("schema_version") != 1
        or decision.get("role")
        != "generation_100k_post_reconciliation_experiment_decision"
        or decision.get("status") != "completed"
        or decision.get("operational_status") != "pass"
        or decision.get("terminal_status") != "hold"
        or decision.get("generation_advantage_proven") is not False
        or next_stage.get("id")
        != "prepare_matched_100k_epsilon_stability_sampling_diagnostic"
        or next_stage.get("execution_ready") is not False
        or next_stage.get("gpu_execution_allowed") is not False
        or boundary.get("new_source_bound_execution_gate_required") is not True
    ):
        raise ValueError("post-reconciliation decision differs")
    _validate_training_git(decision.get("training_git"), label="decision training Git")
    _require_false_fields(
        boundary,
        fields=(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_allowed",
            "export_allowed",
            "release_allowed",
            "process_signals_allowed",
        ),
        label="post-reconciliation authorization boundary",
    )


def _validate_reconciliation(report: Mapping[str, Any]) -> None:
    boundary = _mapping(
        report.get("claim_boundary"), label="reconciliation claim boundary"
    )
    comparison = _mapping(
        report.get("comparison"), label="reconciliation comparison"
    )
    if (
        report.get("schema_version") != 1
        or report.get("role") != "generation_100k_cross_protocol_reconciliation"
        or report.get("status") != "completed"
        or report.get("operational_status") != "pass"
        or report.get("terminal_status") != "hold"
        or comparison.get("sampler_step_changes_matched_ranking") is not True
        or comparison.get("sample_count_changes_matched_ranking") is not False
        or comparison.get("generation_advantage_proven") is not False
        or boundary.get("generation_advantage_proven") is not False
    ):
        raise ValueError("cross-protocol reconciliation differs")
    _require_false_fields(
        boundary,
        fields=(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
        ),
        label="reconciliation claim boundary",
    )


def _validate_pair_monitor(report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != 2
        or report.get("status") != "pass"
        or report.get("stage") != "complete"
        or report.get("issues") != []
        or set(_mapping(report.get("runs"), label="pair-monitor runs"))
        != set(METHODS)
    ):
        raise ValueError("pair monitor is not a clean completed pair")
    _validate_training_git(report.get("git"), label="pair-monitor Git")


def _validate_training_exposure(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = _mapping(report.get("rows"), label="training exposure rows")
    comparison = _mapping(
        report.get("comparison"), label="training exposure comparison"
    )
    milestone = _mapping(
        report.get("milestone_binding"), label="training exposure milestone"
    )
    if (
        report.get("schema_version") != 1
        or report.get("role") != "generation_training_exposure_audit"
        or report.get("status") != "pass"
        or set(rows) != set(METHODS)
        or comparison.get("same_completed_steps") is not True
        or comparison.get("same_images_seen") is not True
        or comparison.get("same_dataset_normalized_exposure") is not True
        or milestone.get("expected_step") != 100_000
        or milestone.get("matched_training_exposure_verified") is not True
    ):
        raise ValueError("terminal training exposure differs")
    _validate_training_git(milestone.get("training_git"), label="exposure training Git")
    for method in METHODS:
        row = _mapping(rows[method], label=f"{method} exposure row")
        if (
            row.get("status") != "complete"
            or row.get("training_complete") is not True
            or row.get("completed_steps") != 100_000
            or row.get("samples_seen") != 6_400_000
            or row.get("effective_batch_size") != 64
        ):
            raise ValueError(f"{method} training exposure differs")
    plan = _mapping(
        _mapping(report.get("quality_bridge_plan"), label="quality bridge plan").get(
            "training_exposure"
        ),
        label="quality bridge exposure plan",
    )
    bridge = _mapping(plan.get("bridge"), label="bridge exposure")
    source = _mapping(plan.get("source"), label="historical exposure")
    return {
        "steps_per_method": 100_000,
        "images_seen_per_method": 6_400_000,
        "full_data_equivalent_epochs": float(bridge["equivalent_epochs"]),
        "historical_10pct_reference_equivalent_epochs": float(
            source["equivalent_epochs"]
        ),
        "insufficient_exposure_remains_live": True,
    }


def _validate_runtime_fairness(report: Mapping[str, Any]) -> dict[str, Any]:
    methods = _mapping(report.get("methods"), label="runtime fairness methods")
    cofitok = _mapping(methods.get("cofitok"), label="CoFiTok runtime fairness")
    verification = _mapping(
        cofitok.get("resume_compute_adjustment_verification"),
        label="CoFiTok recovery compute verification",
    )
    summary = _mapping(verification.get("summary"), label="CoFiTok recovery summary")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "pass"
        or _mapping(report.get("observed_runtime_parity"), label="runtime parity").get(
            "status"
        )
        != "proven"
        or verification.get("status") != "verified"
        or summary.get("discovered_orphan_archive_count") != 2
        or summary.get("covered_orphan_archive_count") != 2
        or summary.get("event_count") != 2
        or summary.get("issues") != []
    ):
        raise ValueError("runtime/compute fairness evidence differs")
    _require_false_fields(
        report.get("claim_boundary"),
        fields=("full_300k_launch_allowed",),
        label="runtime fairness claim boundary",
    )
    return {
        "matched_runtime_parity": True,
        "cofitok_orphan_archives_discovered": 2,
        "cofitok_orphan_archives_covered": 2,
    }


def _validate_terminal_guard(report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("role") != "generation_terminal_system_claim_guard"
        or report.get("status") != "hold"
        or report.get("decision")
        != "terminal_system_evidence_complete_without_qualified_matched_advantage"
    ):
        raise ValueError("terminal system guard differs")
    _require_false_fields(
        report.get("claim_boundary"),
        fields=(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
        ),
        label="terminal guard claim boundary",
    )


def _validate_terminal_completion(report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("role")
        != "generation_quality_bridge_terminal_completion_audit"
        or report.get("status") != "pass"
        or report.get("terminal_status") != "hold"
        or report.get("generation_advantage_proven") is not False
        or report.get("detail")
        != "terminal_quality_bridge_evidence_physically_replayed"
    ):
        raise ValueError("terminal completion audit differs")
    _require_false_fields(
        report.get("authorization_boundary"),
        fields=(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
        ),
        label="terminal completion authorization boundary",
    )


def _validate_supersession(report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("role") != "generation_terminal_route_supersession_interlock"
        or report.get("status") != "pass"
        or report.get("decision")
        != "legacy_v1_gpu_route_consumers_superseded_fail_closed"
        or report.get("generation_advantage_proven") is not False
    ):
        raise ValueError("terminal route supersession receipt differs")
    _require_false_fields(
        report.get("scope"),
        fields=(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
            "gpu_execution_allowed",
        ),
        label="terminal route supersession scope",
    )


def build_epsilon_stability_post_diagnostic_decision(
    *,
    epsilon_stability_sampling_result: Mapping[str, Any],
    post_reconciliation_decision: Mapping[str, Any],
    cross_protocol_reconciliation: Mapping[str, Any],
    pair_monitor: Mapping[str, Any],
    training_exposure_report: Mapping[str, Any],
    runtime_compute_fairness: Mapping[str, Any],
    terminal_system_claim_guard: Mapping[str, Any],
    terminal_completion_audit: Mapping[str, Any],
    terminal_route_supersession_receipt: Mapping[str, Any],
    source_identities: Mapping[str, Any],
    builder_git: Mapping[str, Any],
) -> dict[str, Any]:
    if set(source_identities) != set(SOURCE_NAMES):
        raise ValueError("post-diagnostic decision source set is incomplete")
    normalized_sources = {
        name: _identity(source_identities[name], label=name)
        for name in SOURCE_NAMES
    }
    normalized_builder_git = _git_identity(builder_git, label="decision builder")
    diagnostic = _validate_diagnostic(epsilon_stability_sampling_result)
    _validate_post_reconciliation(post_reconciliation_decision)
    _validate_reconciliation(cross_protocol_reconciliation)
    _validate_pair_monitor(pair_monitor)
    exposure = _validate_training_exposure(training_exposure_report)
    runtime = _validate_runtime_fairness(runtime_compute_fairness)
    _validate_terminal_guard(terminal_system_claim_guard)
    _validate_terminal_completion(terminal_completion_audit)
    _validate_supersession(terminal_route_supersession_receipt)

    return {
        "schema": POST_DIAGNOSTIC_DECISION_SCHEMA,
        "role": POST_DIAGNOSTIC_DECISION_ROLE,
        "status": "completed",
        "terminal_status": "hold",
        "decision": "prepare_fresh_matched_min_snr_training_pilot",
        "generation_advantage_proven": False,
        "builder_git": normalized_builder_git,
        "training_git": {
            "revision": QUALITY_BRIDGE_REVISION,
            "branch": QUALITY_BRIDGE_BRANCH,
            "tracked_dirty": False,
        },
        "source_identities": normalized_sources,
        "evidence_resolution": {
            "terminal_system_evidence": "operational_pass_scientific_hold",
            "matched_sampling_recovery": diagnostic,
            "sampler_only_repair_conclusion": (
                "no_shared_candidate_under_the_completed_matched_1k_screen"
            ),
            "min_snr_training_hypothesis": "untested_and_scientifically_live",
            "training_exposure": exposure,
            "runtime_compute_fairness": runtime,
            "factorization_specific_failure_supported": False,
            "class_only_failure_supported": False,
        },
        "recommended_next_stage": {
            "id": "prepare_fresh_matched_min_snr_training_pilot",
            "category": "matched_primary_epsilon_loss_intervention",
            "objective": (
                "Test whether matched standard Min-SNR epsilon-loss weighting "
                "improves the shared high-noise generation failure while "
                "preserving CoFiTok's ordered restricted epsilon-token semantics."
            ),
            "execution_ready": False,
            "matched_methods": list(METHODS),
            "controlled_change": {
                "field": "loss.min_snr_gamma",
                "legacy_value": 0.0,
                "pilot_value_constraint": "finite_positive_and_identical_for_both_methods",
                "weighting": "min(SNR, gamma) / SNR",
                "prediction_target": "epsilon",
                "cofitok_representation_changed": False,
            },
            "attribution_policy": {
                "may_be_described_as_exposure_only": False,
                "primary_question": "matched_min_snr_training_recipe_effect",
                "exposure_remains_a_live_alternative": True,
                "legacy_control_milestone_binding_required": True,
                "fresh_gamma_zero_control_required_if_exact_control_equivalence_fails": True,
            },
            "preparation_requirements": {
                "fresh_initialization_required": True,
                "changed_config_resume_allowed": False,
                "isolated_output_roots_required": True,
                "matched_pair_contract_required": True,
                "identical_positive_min_snr_gamma_required": True,
                "exact_data_initialization_and_random_stream_binding_required": True,
                "physical_checkpoint_integrity_required": True,
                "matched_ddim100_quality_and_class_fidelity_evaluation_required": True,
                "new_versioned_execution_gate_required": True,
                "terminal_hold_replacement_allowed": False,
            },
            "gate_must_fix": [
                "positive min_snr_gamma",
                "pilot step and image budget",
                "matched legacy control policy",
                "checkpoint milestones",
                "sample count and fresh random stream",
                "quality, support, class-fidelity, and mechanism thresholds",
                "exact Git revision/tree/branch, configs, runtime, data, and output roots",
            ],
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "claim_policy": {
            "recommendation_is_not_execution_authorization": True,
            "one_thousand_sample_diagnostic_is_screening_only": True,
            "terminal_result_remains_authoritative": True,
            "matched_fid_point_estimate_proves_advantage": False,
            "absolute_usability_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "cross_tier_numeric_ranking_allowed": False,
        },
        "authorization_boundary": dict(POST_DIAGNOSTIC_AUTHORIZATION_BOUNDARY),
    }


def validate_epsilon_stability_post_diagnostic_decision(
    decision: Mapping[str, Any],
    **sources: Any,
) -> dict[str, Any]:
    expected = build_epsilon_stability_post_diagnostic_decision(**sources)
    if dict(decision) != expected:
        raise ValueError("post-diagnostic decision does not replay exactly")
    return expected
