"""Fail-closed synthesis of terminal evidence after a capacity-screen hold."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_screen import ARM_NAMES
from cofitok.generation.capacity_screen_arm import (
    validate_capacity_screen_arm_validation,
)
from cofitok.generation.capacity_screen_result import (
    validate_capacity_screen_result_contract,
    validate_capacity_screen_validation_receipt,
)
from cofitok.generation.exposure_capacity_decision import (
    validate_decision_contract as validate_exposure_decision_contract,
)
from cofitok.generation.exposure_capacity_result import (
    validate_validation_receipt as validate_exposure_result_receipt,
)
from cofitok.generation.sampling_recovery_audit import (
    validate_result as validate_sampling_recovery_result,
)


DECISION_SCHEMA = "cofitok_generation_capacity_screen_hold_recovery_v1"
DECISION_ROLE = "source_bound_capacity_screen_hold_recovery_decision"
VALIDATION_SCHEMA = "cofitok_generation_capacity_screen_hold_recovery_validation_v1"
VALIDATION_ROLE = "content_addressed_capacity_screen_hold_recovery_validation"

EXPOSURE_EXECUTION_GIT = {
    "revision": "5c23141a24a3385101ed6081c1ed656aad955a99",
    "tree": "ffe66d2d2fc666884a21856bbc84a23fe667f88b",
    "branch": "analysis/generation-exposure-capacity-source-compatible-20260901",
    "tracked_dirty": False,
}
CAPACITY_EXECUTION_GIT = {
    "revision": "3370e68727259fb9144216d5c50ec9262645dc86",
    "tree": "5dc3e72beb969e574aec60e652f761602c2ae31b",
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}

CAPACITY_FAILED_CHECKS = [
    "cofitok_recall_improves_with_capacity",
    "cofitok_precision_non_regression",
    "dense_identity_fid_improves_with_capacity",
    "dense_identity_recall_improves_with_capacity",
]
ROLLOUT_NAMES = (
    "exposure_cofitok",
    "exposure_dense_identity",
    "capacity_base128_cofitok",
    "capacity_base128_dense_identity",
    "capacity_base256_cofitok",
    "capacity_base256_dense_identity",
)

AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "capacity_confirmation_preparation_allowed": False,
    "capacity_confirmation_launch_allowed": False,
    "full_training_preparation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
CORE_DISABLED_PERMISSIONS = {
    "gpu_execution_allowed",
    "training_launch_allowed",
    "sampling_launch_allowed",
    "evaluation_launch_allowed",
    "full_training_launch_allowed",
    "full_300k_launch_allowed",
    "promotion_allowed",
    "release_allowed",
    "process_signals_allowed",
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    path = row.get("path")
    byte_count = row.get("bytes")
    digest = row.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 1
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} identity is malformed")
    return {"path": path, "bytes": byte_count, "sha256": digest}


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    if (
        not isinstance(row.get("revision"), str)
        or len(row["revision"]) != 40
        or not isinstance(row.get("tree"), str)
        or len(row["tree"]) != 40
        or not isinstance(row.get("branch"), str)
        or not row["branch"]
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} must identify one exact clean checkout")
    return copy.deepcopy(row)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite domain")
    return result


def _disabled_boundary(value: Any, name: str) -> dict[str, Any]:
    boundary = _object(value, name)
    missing = sorted(CORE_DISABLED_PERMISSIONS - set(boundary))
    if missing:
        raise ValueError(f"{name} omits disabled permissions: {missing}")
    for key, enabled in boundary.items():
        if (
            key == "decision_is_execution_authorization"
            or key.endswith("_allowed")
            or key.endswith("_authorized")
        ) and enabled is not False:
            raise ValueError(f"{name} does not explicitly disable {key}")
    return copy.deepcopy(boundary)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_by_name(result: Mapping[str, Any], name: str) -> dict[str, Any]:
    matches = [
        _object(row, f"capacity check {name}")
        for row in result.get("checks", [])
        if isinstance(row, Mapping) and row.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"capacity check is missing or duplicated: {name}")
    return matches[0]


def _validate_exposure_chain(
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    result_receipt: Mapping[str, Any],
    result_receipt_identity: Mapping[str, Any],
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    decision_validation: Mapping[str, Any],
    decision_validation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    result_id = _identity(result_identity, "exposure result")
    result_receipt_id = _identity(
        result_receipt_identity, "exposure result validation"
    )
    decision_id = _identity(decision_identity, "exposure decision")
    decision_validation_id = _identity(
        decision_validation_identity, "exposure decision validation"
    )
    if result.get("execution_checkout") != EXPOSURE_EXECUTION_GIT:
        raise ValueError("exposure result execution Git differs")
    validate_exposure_result_receipt(
        result_receipt,
        result=result,
        result_identity=result_id,
    )
    validated_decision = validate_exposure_decision_contract(decision)
    if validated_decision.get("decision") != "capacity_screen":
        raise ValueError("exposure decision does not select the capacity screen")
    decision_sources = _object(
        validated_decision.get("source_evidence"), "exposure decision sources"
    )
    if (
        decision_sources.get("exposure_result") != result_id
        or decision_sources.get("exposure_result_validation_receipt")
        != result_receipt_id
        or decision_sources.get("execution_git") != EXPOSURE_EXECUTION_GIT
    ):
        raise ValueError("exposure decision source chain differs")
    validation = _object(decision_validation, "exposure decision validation")
    verified = _object(
        validation.get("verified_sources"), "exposure validation sources"
    )
    if (
        validation.get("schema_version")
        != "cofitok_generation_exposure_capacity_decision_validation_v1"
        or validation.get("role")
        != "content_addressed_exposure_capacity_scientific_decision_validation"
        or validation.get("status") != "pass"
        or validation.get("scientific_route") != "capacity_screen"
        or validation.get("generation_advantage_proven") is not False
        or validation.get("decision") != decision_id
        or verified.get("exposure_result") != result_id
        or verified.get("exposure_result_validation_receipt")
        != result_receipt_id
    ):
        raise ValueError("exposure decision validation chain differs")
    _disabled_boundary(
        validation.get("authorization_boundary"),
        "exposure decision validation boundary",
    )
    return {
        "result": result_id,
        "result_validation": result_receipt_id,
        "decision": decision_id,
        "decision_validation": decision_validation_id,
        "execution_git": copy.deepcopy(EXPOSURE_EXECUTION_GIT),
        "route": "capacity_screen",
    }


def _validate_capacity_chain(
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    result_receipt: Mapping[str, Any],
    result_receipt_identity: Mapping[str, Any],
    arm_validations: Mapping[str, Mapping[str, Any]],
    arm_validation_identities: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    validated = validate_capacity_screen_result_contract(result)
    result_id = _identity(result_identity, "capacity screen result")
    receipt_id = _identity(
        result_receipt_identity, "capacity screen result validation"
    )
    if validated.get("result_git") != CAPACITY_EXECUTION_GIT:
        raise ValueError("capacity screen result Git differs")
    if (
        validated.get("scientific_status") != "hold"
        or validated.get("failed_checks") != CAPACITY_FAILED_CHECKS
        or _object(validated.get("next_stage"), "capacity next stage").get(
            "capacity_confirmation_preparation_allowed"
        )
        is not False
    ):
        raise ValueError("capacity screen is not the canonical scientific hold")
    validate_capacity_screen_validation_receipt(
        result_receipt,
        result=validated,
        result_identity=result_id,
    )
    if set(arm_validations) != set(ARM_NAMES) or set(
        arm_validation_identities
    ) != set(ARM_NAMES):
        raise ValueError("capacity hold arm validation set differs")
    source_arms = _object(
        _object(validated.get("source_evidence"), "capacity result sources").get(
            "arm_validations"
        ),
        "capacity result arm sources",
    )
    normalized_arms: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        arm_id = _identity(
            arm_validation_identities[arm], f"{arm} validation"
        )
        if source_arms.get(arm) != arm_id:
            raise ValueError(f"capacity result binds another {arm} validation")
        normalized_arms[arm] = validate_capacity_screen_arm_validation(
            arm_validations[arm]
        )
    return (
        {
            "result": result_id,
            "result_validation": receipt_id,
            "execution_git": copy.deepcopy(CAPACITY_EXECUTION_GIT),
            "scientific_status": "hold",
            "failed_checks": list(CAPACITY_FAILED_CHECKS),
        },
        normalized_arms,
    )


def _validate_sampling_recovery(
    report: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    allowed_output_prefix: str,
) -> dict[str, Any]:
    source_id = _identity(identity, "sampling recovery result")
    summary = validate_sampling_recovery_result(
        report,
        result_path=Path(source_id["path"]),
        expected_result_sha256=source_id["sha256"],
        allowed_output_prefix=allowed_output_prefix,
    )
    if (
        summary.get("status") != "pass"
        or summary.get("selection_status")
        != "no_shared_sampling_recovery_candidate"
        or summary.get("generation_advantage_proven") is not False
    ):
        raise ValueError("sampling recovery result does not preserve no-candidate hold")
    return {
        "result": source_id,
        "selection_status": "no_shared_sampling_recovery_candidate",
        "case_count": summary["case_count"],
        "candidate_count": summary["candidate_count"],
        "physical_identity_count": summary["physical_identity_count"],
    }


def _validate_min_snr(
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    guard: Mapping[str, Any],
    guard_identity: Mapping[str, Any],
) -> dict[str, Any]:
    result_id = _identity(result_identity, "Min-SNR result")
    guard_id = _identity(guard_identity, "Min-SNR physical guard")
    if (
        result.get("schema") != "cofitok_matched_min_snr_pilot_result_v1"
        or result.get("role") != "generation_matched_min_snr_training_pilot_result"
        or result.get("status") != "completed"
        or result.get("scientific_status") != "screening_only"
        or result.get("selection_status")
        != "no_shared_min_snr_candidate_at_50k"
        or result.get("terminal_status") != "hold"
        or result.get("generation_advantage_proven") is not False
    ):
        raise ValueError("Min-SNR result does not preserve the terminal hold")
    _disabled_boundary(result.get("authorization_boundary"), "Min-SNR boundary")
    evidence = _object(guard.get("evidence"), "Min-SNR guard evidence")
    double_replay = _object(
        guard.get("double_replay"), "Min-SNR guard double replay"
    )
    source_replay = _object(
        guard.get("source_code_replay"), "Min-SNR source replay"
    )
    required_physical_flags = {
        "all_checkpoint_payloads_physically_hashed",
        "all_checkpoint_payloads_rehashed_after_replay",
        "all_classifier_weights_physically_hashed",
        "all_sample_sets_physically_hashed",
        "all_source_reports_physically_hashed",
        "all_terminal_sources_rehashed_after_replay",
        "all_training_control_checkpoints_physically_bound",
        "result_logical_replay_exact",
    }
    if (
        guard.get("schema")
        != "cofitok_matched_min_snr_terminal_physical_guard_v1"
        or guard.get("role")
        != "generation_matched_min_snr_terminal_physical_guard"
        or guard.get("status") != "pass"
        or guard.get("scientific_status")
        != "physical_evidence_replayed_non_authorizing"
        or guard.get("selection_status")
        != "no_shared_min_snr_candidate_at_50k"
        or guard.get("terminal_status") != "hold"
        or guard.get("generation_advantage_proven") is not False
        or evidence.get("result") != result_id
        or evidence.get("selection_status")
        != "no_shared_min_snr_candidate_at_50k"
        or any(evidence.get(name) is not True for name in required_physical_flags)
        or double_replay.get("identical") is not True
        or int(double_replay.get("pass_count", 0)) < 2
        or source_replay.get("identical") is not True
        or source_replay.get("physical_identities_verified") is not True
        or int(source_replay.get("pass_count", 0)) < 3
    ):
        raise ValueError("Min-SNR physical replay guard differs")
    _disabled_boundary(
        guard.get("authorization_boundary"), "Min-SNR guard boundary"
    )
    return {
        "result": result_id,
        "physical_guard": guard_id,
        "selection_status": "no_shared_min_snr_candidate_at_50k",
        "logical_replay_exact": True,
        "physical_sources_replayed": True,
    }


def _rollout_summary(
    report: Mapping[str, Any],
    *,
    name: str,
    expected_identity: Mapping[str, Any],
    actual_identity: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    report_id = _identity(actual_identity, f"{name} rollout")
    if report_id != expected_identity:
        raise ValueError(f"{name} rollout identity differs from its parent evidence")
    git = _object(report.get("git"), f"{name} rollout Git")
    if (
        git.get("revision") != expected_git["revision"]
        or git.get("branch") != expected_git["branch"]
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} rollout Git differs")
    free = _object(
        report.get("free_sampling_rollout"), f"{name} free sampling rollout"
    )
    timesteps = free.get("timesteps")
    steps = free.get("steps")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "completed"
        or report.get("weights") != "ema"
        or not isinstance(timesteps, list)
        or not timesteps
        or timesteps[0] != 999
        or not isinstance(steps, list)
        or len(steps) != len(timesteps)
    ):
        raise ValueError(f"{name} rollout protocol differs")
    first = _object(steps[0], f"{name} first free rollout step")
    if first.get("timestep") != 999:
        raise ValueError(f"{name} first free rollout step is not t=999")
    clip_fraction = _finite(
        first.get("raw_x0_clip_fraction"),
        f"{name} t=999 raw-x0 clip fraction",
        minimum=0.0,
    )
    predicted_x0_rms = _finite(
        first.get("predicted_x0_rms"),
        f"{name} t=999 predicted-x0 RMS",
        minimum=0.0,
    )
    if clip_fraction > 1.0:
        raise ValueError(f"{name} t=999 clip fraction exceeds one")
    return {
        "report": report_id,
        "timestep": 999,
        "raw_x0_clip_fraction": clip_fraction,
        "predicted_x0_rms": predicted_x0_rms,
    }


def build_hold_recovery_decision(
    *,
    exposure_result: Mapping[str, Any],
    exposure_result_identity: Mapping[str, Any],
    exposure_result_validation: Mapping[str, Any],
    exposure_result_validation_identity: Mapping[str, Any],
    exposure_decision: Mapping[str, Any],
    exposure_decision_identity: Mapping[str, Any],
    exposure_decision_validation: Mapping[str, Any],
    exposure_decision_validation_identity: Mapping[str, Any],
    capacity_result: Mapping[str, Any],
    capacity_result_identity: Mapping[str, Any],
    capacity_result_validation: Mapping[str, Any],
    capacity_result_validation_identity: Mapping[str, Any],
    capacity_arm_validations: Mapping[str, Mapping[str, Any]],
    capacity_arm_validation_identities: Mapping[str, Mapping[str, Any]],
    sampling_recovery_result: Mapping[str, Any],
    sampling_recovery_identity: Mapping[str, Any],
    min_snr_result: Mapping[str, Any],
    min_snr_result_identity: Mapping[str, Any],
    min_snr_guard: Mapping[str, Any],
    min_snr_guard_identity: Mapping[str, Any],
    rollout_reports: Mapping[str, Mapping[str, Any]],
    rollout_identities: Mapping[str, Mapping[str, Any]],
    decision_git: Mapping[str, Any],
    allowed_sampling_output_prefix: str,
) -> dict[str, Any]:
    exposure = _validate_exposure_chain(
        result=exposure_result,
        result_identity=exposure_result_identity,
        result_receipt=exposure_result_validation,
        result_receipt_identity=exposure_result_validation_identity,
        decision=exposure_decision,
        decision_identity=exposure_decision_identity,
        decision_validation=exposure_decision_validation,
        decision_validation_identity=exposure_decision_validation_identity,
    )
    capacity, validated_arms = _validate_capacity_chain(
        result=capacity_result,
        result_identity=capacity_result_identity,
        result_receipt=capacity_result_validation,
        result_receipt_identity=capacity_result_validation_identity,
        arm_validations=capacity_arm_validations,
        arm_validation_identities=capacity_arm_validation_identities,
    )
    sampling = _validate_sampling_recovery(
        sampling_recovery_result,
        sampling_recovery_identity,
        allowed_output_prefix=allowed_sampling_output_prefix,
    )
    min_snr = _validate_min_snr(
        result=min_snr_result,
        result_identity=min_snr_result_identity,
        guard=min_snr_guard,
        guard_identity=min_snr_guard_identity,
    )
    if set(rollout_reports) != set(ROLLOUT_NAMES) or set(
        rollout_identities
    ) != set(ROLLOUT_NAMES):
        raise ValueError("terminal rollout evidence set differs")
    exposure_expected = {
        "exposure_cofitok": _object(
            _object(exposure_result.get("rollout"), "exposure rollouts").get(
                "cofitok"
            ),
            "exposure CoFiTok rollout",
        ).get("report"),
        "exposure_dense_identity": _object(
            _object(exposure_result.get("rollout"), "exposure rollouts").get(
                "dense_identity"
            ),
            "exposure dense rollout",
        ).get("report"),
    }
    capacity_expected = {
        f"capacity_{arm}": _object(
            validated_arms[arm].get("rollout"), f"{arm} rollout evidence"
        ).get("report")
        for arm in ARM_NAMES
    }
    rollout_summaries: dict[str, dict[str, Any]] = {}
    for name in ROLLOUT_NAMES:
        expected_git = (
            EXPOSURE_EXECUTION_GIT
            if name.startswith("exposure_")
            else CAPACITY_EXECUTION_GIT
        )
        expected_identity = (
            exposure_expected[name]
            if name in exposure_expected
            else capacity_expected[name]
        )
        rollout_summaries[name] = _rollout_summary(
            rollout_reports[name],
            name=name,
            expected_identity=_identity(
                expected_identity, f"{name} parent rollout identity"
            ),
            actual_identity=rollout_identities[name],
            expected_git=expected_git,
        )

    clip_values = [
        row["raw_x0_clip_fraction"] for row in rollout_summaries.values()
    ]
    samples_per_arm = _object(
        capacity_result.get("validated_preparation"),
        "capacity validated preparation",
    ).get("evaluation_contract", {}).get("samples_per_arm")
    if samples_per_arm != 1_000:
        raise ValueError("capacity screen is not the bounded 1K-per-arm evaluation")
    recalls = {
        arm: _finite(
            _object(
                _object(summary, f"{arm} summary").get("distribution"),
                f"{arm} distribution",
            ).get("recall"),
            f"{arm} recall",
            minimum=0.0,
        )
        for arm, summary in _object(
            capacity_result.get("arm_summaries"), "capacity arm summaries"
        ).items()
    }
    if set(recalls) != set(ARM_NAMES) or any(value != 0.0 for value in recalls.values()):
        raise ValueError("capacity recall evidence is not the observed all-zero screen")
    precision_check = _check_by_name(
        capacity_result, "cofitok_precision_non_regression"
    )
    dense_fid_check = _check_by_name(
        capacity_result, "dense_identity_fid_improves_with_capacity"
    )
    if precision_check.get("passed") is not False or dense_fid_check.get(
        "passed"
    ) is not False:
        raise ValueError("capacity hold lacks independent precision/FID failures")

    return {
        "schema_version": DECISION_SCHEMA,
        "role": DECISION_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "decision": "hold_for_source_bound_training_objective_reassessment",
        "decision_git": _git(decision_git, "capacity hold recovery decision"),
        "source_evidence": {
            "exposure": exposure,
            "capacity": capacity,
            "sampling_recovery": sampling,
            "min_snr": min_snr,
            "rollouts": rollout_summaries,
        },
        "evidence_interpretation": {
            "exposure": {
                "observation": "additional_matched_exposure_preserved_mechanism_but_did_not_clear_quality_floors",
                "capacity_route_was_eligible_before_screen": True,
                "insufficient_exposure_is_proven_root_cause": False,
            },
            "capacity": {
                "observation": "larger_capacity_failed_the_predeclared_four_arm_screen",
                "failed_checks": list(CAPACITY_FAILED_CHECKS),
                "capacity_is_proven_root_cause": False,
                "original_thresholds_may_be_relaxed": False,
            },
            "terminal_start": {
                "observation": "all_six_free_rollouts_begin_at_t999_with_more_than_99_percent_raw_x0_clipping",
                "minimum_raw_x0_clip_fraction": min(clip_values),
                "maximum_raw_x0_clip_fraction": max(clip_values),
                "all_at_least_0_99": all(value >= 0.99 for value in clip_values),
                "terminal_low_snr_or_distribution_support_issue_is_plausible": True,
                "causal_root_cause_is_proven": False,
            },
            "recall_power": {
                "samples_per_arm": 1_000,
                "observed_recall": recalls,
                "interpretation": "directional_only_and_limited_for_zero_recall_differences",
                "may_override_precision_or_fid_failures": False,
                "may_upgrade_screen_to_pass": False,
            },
            "independent_adverse_evidence": {
                "cofitok_precision_capacity_delta": _finite(
                    precision_check.get("observed"), "CoFiTok precision delta"
                ),
                "cofitok_precision_threshold": _finite(
                    precision_check.get("threshold"), "CoFiTok precision threshold"
                ),
                "dense_relative_fid_capacity_change": _finite(
                    dense_fid_check.get("observed"), "dense relative FID change"
                ),
                "dense_fid_improvement_threshold": _finite(
                    dense_fid_check.get("threshold"), "dense FID threshold"
                ),
                "hold_remains_required_without_recall_checks": True,
            },
            "ruled_out_routes": {
                "sampling_only_recovery": "no_shared_sampling_recovery_candidate",
                "min_snr_gamma5": "no_shared_min_snr_candidate_at_50k",
            },
        },
        "next_stage": {
            "route": "hold",
            "selected_intervention": None,
            "intervention_selected": False,
            "preparation_allowed": False,
            "reason": "current_evidence_does_not_identify_one_clean_matched_intervention",
            "required_next_evidence": "A separate source-bound objective reassessment selecting exactly one bounded matched terminal-low-SNR or distribution-support intervention.",
            "capacity_confirmation_allowed": False,
            "gpu_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "claim_policy": {
            "formal_generation_claim_allowed": False,
            "capacity_screen_pass_claim_allowed": False,
            "terminal_start_observation_is_causal_claim": False,
            "one_thousand_sample_recall_is_confirmatory": False,
            "generation_advantage_proven": False,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_hold_recovery_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(report, "capacity screen hold recovery decision")
    next_stage = _object(row.get("next_stage"), "hold recovery next stage")
    interpretation = _object(
        row.get("evidence_interpretation"), "hold recovery interpretation"
    )
    terminal = _object(
        interpretation.get("terminal_start"), "terminal-start interpretation"
    )
    recall = _object(
        interpretation.get("recall_power"), "recall-power interpretation"
    )
    adverse = _object(
        interpretation.get("independent_adverse_evidence"),
        "independent adverse evidence",
    )
    capacity = _object(
        interpretation.get("capacity"), "capacity interpretation"
    )
    if (
        row.get("schema_version") != DECISION_SCHEMA
        or row.get("role") != DECISION_ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("scientific_status") != "hold"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("decision")
        != "hold_for_source_bound_training_objective_reassessment"
        or row.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or capacity.get("failed_checks") != CAPACITY_FAILED_CHECKS
        or capacity.get("original_thresholds_may_be_relaxed") is not False
        or terminal.get("all_at_least_0_99") is not True
        or terminal.get("causal_root_cause_is_proven") is not False
        or recall.get("samples_per_arm") != 1_000
        or recall.get("may_upgrade_screen_to_pass") is not False
        or recall.get("may_override_precision_or_fid_failures") is not False
        or adverse.get("hold_remains_required_without_recall_checks") is not True
        or next_stage.get("route") != "hold"
        or next_stage.get("selected_intervention") is not None
        or next_stage.get("intervention_selected") is not False
        or next_stage.get("preparation_allowed") is not False
        or next_stage.get("capacity_confirmation_allowed") is not False
        or next_stage.get("gpu_execution_allowed") is not False
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("capacity screen hold recovery contract differs")
    _git(row.get("decision_git"), "capacity hold recovery decision")
    sources = _object(row.get("source_evidence"), "hold recovery sources")
    if set(sources) != {
        "exposure",
        "capacity",
        "sampling_recovery",
        "min_snr",
        "rollouts",
    }:
        raise ValueError("capacity hold recovery source set differs")
    rollouts = _object(sources.get("rollouts"), "hold recovery rollouts")
    if set(rollouts) != set(ROLLOUT_NAMES):
        raise ValueError("capacity hold recovery rollout set differs")
    for name, summary in rollouts.items():
        rollout = _object(summary, f"{name} rollout summary")
        _identity(rollout.get("report"), f"{name} rollout report")
        if rollout.get("timestep") != 999:
            raise ValueError(f"{name} terminal timestep differs")
        value = _finite(
            rollout.get("raw_x0_clip_fraction"), f"{name} clip fraction"
        )
        if value < 0.99 or value > 1.0:
            raise ValueError(f"{name} clip-fraction observation differs")
    return copy.deepcopy(row)


def build_hold_recovery_validation(
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_hold_recovery_decision(decision)
    decision_id = _identity(decision_identity, "capacity hold recovery decision")
    validator = _git(validator_git, "capacity hold recovery validator")
    sources = copy.deepcopy(
        _object(validated.get("source_evidence"), "hold recovery sources")
    )
    basis = {
        "decision": decision_id,
        "decision_git": validated["decision_git"],
        "validator_git": validator,
        "source_evidence": sources,
    }
    return {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "decision": decision_id,
        "decision_git": copy.deepcopy(validated["decision_git"]),
        "validator_git": validator,
        "source_evidence": sources,
        "validation_basis_sha256": _canonical_sha256(basis),
        "scientific_status": "hold",
        "generation_advantage_proven": False,
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_hold_recovery_validation(
    receipt: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(receipt, "capacity hold recovery validation")
    validator = _git(row.get("validator_git"), "capacity hold recovery validator")
    expected = build_hold_recovery_validation(
        decision=decision,
        decision_identity=decision_identity,
        validator_git=validator,
    )
    if row != expected:
        raise ValueError("capacity hold recovery validation is not reproducible")
    return expected


__all__ = [
    "AUTHORIZATION_BOUNDARY",
    "CAPACITY_EXECUTION_GIT",
    "CAPACITY_FAILED_CHECKS",
    "CORE_DISABLED_PERMISSIONS",
    "DECISION_ROLE",
    "DECISION_SCHEMA",
    "EXPOSURE_EXECUTION_GIT",
    "ROLLOUT_NAMES",
    "VALIDATION_ROLE",
    "VALIDATION_SCHEMA",
    "build_hold_recovery_decision",
    "build_hold_recovery_validation",
    "validate_hold_recovery_decision",
    "validate_hold_recovery_validation",
]
