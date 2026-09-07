"""Source-bound selection of a bounded terminal-SNR training screen."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from cofitok.configs import config_from_dict, config_to_dict
from cofitok.generation.capacity_screen_hold_recovery import (
    validate_hold_recovery_decision,
    validate_hold_recovery_validation,
)
from cofitok.generation.sampling_recovery_audit import (
    validate_result as validate_sampling_recovery_result,
)
from cofitok.generation_pair import generation_pair_contract


REASSESSMENT_SCHEMA = "cofitok_generation_terminal_snr_reassessment_v1"
REASSESSMENT_ROLE = "source_bound_terminal_snr_objective_reassessment"
VALIDATION_SCHEMA = "cofitok_generation_terminal_snr_reassessment_validation_v1"
VALIDATION_ROLE = "terminal_snr_objective_reassessment_validation"
SELECTED_INTERVENTION = "matched_cosine_endpoint_fraction_0p975"
CONTROL_ENDPOINT_FRACTION = 1.0
SELECTED_ENDPOINT_FRACTION = 0.975
BF16_SPACING_AT_ONE = 2.0**-7

AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "process_signals_allowed": False,
    "capacity_confirmation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
}

SCREEN_THRESHOLDS = {
    "both_methods_min_relative_fid_improvement": 0.05,
    "both_methods_max_precision_regression": 0.05,
    "both_methods_max_recall_regression": 0.01,
    "both_methods_max_class_top1_regression": 0.01,
    "both_methods_max_class_top5_regression": 0.02,
    "both_methods_max_terminal_raw_x0_clip_fraction": 0.95,
    "both_methods_min_terminal_raw_x0_clip_fraction_reduction": 0.04,
    "cofitok_ordered_rank_by_path_auc": 1,
    "cofitok_min_coarse_token_energy_ratio": 0.1,
    "cofitok_max_tail_two_energy_ratio": 0.65,
    "cofitok_zero_token_max_abs": 1e-8,
    "cofitok_min_shuffled_to_ordered_endpoint_ratio": 2.0,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    if not isinstance(row["path"], str) or not row["path"].startswith("/"):
        raise ValueError(f"{name} path must be absolute")
    if not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] < 1:
        raise ValueError(f"{name} byte count is invalid")
    digest = row["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} SHA256 is invalid")
    return row


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"branch", "revision", "tracked_dirty", "tree"}:
        raise ValueError(f"{name} Git fields differ")
    for field in ("revision", "tree"):
        digest = row[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 40
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} {field} is invalid")
    if not isinstance(row["branch"], str) or not row["branch"]:
        raise ValueError(f"{name} branch is invalid")
    if row["tracked_dirty"] is not False:
        raise ValueError(f"{name} must be tracked-clean")
    return row


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _diff_paths(first: Any, second: Any, prefix: str = "") -> list[str]:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        paths: list[str] = []
        for key in sorted(set(first) | set(second)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in first or key not in second:
                paths.append(child)
            else:
                paths.extend(_diff_paths(first[key], second[key], child))
        return paths
    return [] if first == second else [prefix]


def _cosine_summary(num_timesteps: int, endpoint_fraction: float) -> dict[str, Any]:
    if num_timesteps != 1_000:
        raise ValueError("terminal-SNR reassessment requires 1000 diffusion timesteps")
    if not 0.0 < endpoint_fraction <= 1.0:
        raise ValueError("cosine endpoint fraction is invalid")
    initial = math.cos(((0.0 + 0.008) / 1.008) * math.pi * 0.5) ** 2
    alpha_bar = 1.0
    previous = 1.0
    for index in range(1, num_timesteps + 1):
        fraction = endpoint_fraction * index / num_timesteps
        cumulative = (
            math.cos(((fraction + 0.008) / 1.008) * math.pi * 0.5) ** 2
            / initial
        )
        beta = min(max(1.0 - cumulative / previous, 1e-4), 0.999)
        alpha_bar *= 1.0 - beta
        previous = cumulative
    sqrt_alpha_bar = math.sqrt(alpha_bar)
    return {
        "calculation": "float64_reference_matching_cosine_beta_clamps",
        "num_train_timesteps": num_timesteps,
        "endpoint_fraction": endpoint_fraction,
        "terminal_alpha_bar": alpha_bar,
        "terminal_sqrt_alpha_bar": sqrt_alpha_bar,
        "terminal_snr": alpha_bar / (1.0 - alpha_bar),
        "epsilon_to_x0_condition_factor": 1.0 / sqrt_alpha_bar,
        "bf16_spacing_at_one": BF16_SPACING_AT_ONE,
        "bf16_spacing_x0_scale": BF16_SPACING_AT_ONE / sqrt_alpha_bar,
    }


def _resolved_config(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        return config_to_dict(config_from_dict(dict(value)))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid experiment config") from exc


def _config_contract(
    *,
    control_configs: Mapping[str, Mapping[str, Any]],
    control_identities: Mapping[str, Mapping[str, Any]],
    intervention_configs: Mapping[str, Mapping[str, Any]],
    intervention_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    methods = {"cofitok", "dense_identity"}
    if (
        set(control_configs) != methods
        or set(control_identities) != methods
        or set(intervention_configs) != methods
        or set(intervention_identities) != methods
    ):
        raise ValueError("terminal-SNR config pair sets differ")
    controls = {
        method: _resolved_config(control_configs[method], f"{method} control config")
        for method in sorted(methods)
    }
    interventions = {
        method: _resolved_config(
            intervention_configs[method], f"{method} intervention config"
        )
        for method in sorted(methods)
    }
    control_pair = generation_pair_contract(
        controls["cofitok"], controls["dense_identity"]
    )
    intervention_pair = generation_pair_contract(
        interventions["cofitok"], interventions["dense_identity"]
    )
    if not control_pair["valid"] or not intervention_pair["valid"]:
        raise ValueError("terminal-SNR configs do not form valid matched pairs")

    expected_differences = ["diffusion.cosine_endpoint_fraction", "name"]
    differences: dict[str, list[str]] = {}
    for method in sorted(methods):
        control = controls[method]
        intervention = interventions[method]
        differences[method] = _diff_paths(control, intervention)
        if differences[method] != expected_differences:
            raise ValueError(
                f"{method} intervention changes more than the endpoint fraction"
            )
        for label, config, endpoint in (
            ("control", control, CONTROL_ENDPOINT_FRACTION),
            ("intervention", intervention, SELECTED_ENDPOINT_FRACTION),
        ):
            diffusion = _object(config.get("diffusion"), f"{method} {label} diffusion")
            runtime = _object(config.get("runtime"), f"{method} {label} runtime")
            model = _object(config.get("model"), f"{method} {label} model")
            if (
                diffusion.get("schedule_type") != "cosine"
                or diffusion.get("prediction_target") != "epsilon"
                or diffusion.get("num_train_timesteps") != 1_000
                or diffusion.get("cosine_endpoint_fraction") != endpoint
                or runtime.get("precision") != "bf16"
                or runtime.get("seed") != 2027
                or runtime.get("steps") != 100_000
                or 10_000 not in runtime.get("protected_checkpoint_steps", [])
                or model.get("base_channels") != 128
            ):
                raise ValueError(f"{method} {label} config contract differs")
    return {
        "control_configs": {
            method: _identity(control_identities[method], f"{method} control config")
            for method in sorted(methods)
        },
        "intervention_configs": {
            method: _identity(
                intervention_identities[method], f"{method} intervention config"
            )
            for method in sorted(methods)
        },
        "control_pair_contract": control_pair,
        "intervention_pair_contract": intervention_pair,
        "per_method_resolved_difference_paths": differences,
        "scientific_difference_paths": ["diffusion.cosine_endpoint_fraction"],
        "metadata_difference_paths": ["name"],
    }


def _sampling_case_summary(
    result: Mapping[str, Any], case_id: str
) -> dict[str, Any]:
    candidates = result.get("candidates")
    observations = result.get("observations")
    if not isinstance(candidates, list) or not isinstance(observations, list):
        raise ValueError("sampling recovery candidate evidence is invalid")
    matching = [row for row in candidates if row.get("case_id") == case_id]
    if len(matching) != 1:
        raise ValueError(f"sampling recovery case {case_id} is missing")
    candidate = _object(matching[0], f"{case_id} candidate")
    if candidate.get("passes_shared_recovery_screen") is not False:
        raise ValueError("sampling-only recovery must remain rejected")
    methods = _object(candidate.get("methods"), f"{case_id} methods")
    summaries: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        comparison = _object(
            _object(methods.get(method), f"{case_id}/{method}").get("versus_legacy"),
            f"{case_id}/{method} legacy comparison",
        )
        improvement = float(comparison.get("fid_relative_improvement", math.nan))
        checks = _object(comparison.get("checks"), f"{case_id}/{method} checks")
        if not math.isfinite(improvement) or improvement <= 0.0:
            raise ValueError(f"{case_id}/{method} did not improve FID directionally")
        summaries[method] = {
            "fid_relative_improvement": improvement,
            "failed_checks": sorted(name for name, passed in checks.items() if passed is False),
            "passes_sampling_only_screen": comparison.get("passes"),
        }
    protocols = []
    for observation in observations:
        if observation.get("case_id") != case_id:
            continue
        sampling = _object(observation.get("sampling"), f"{case_id} sampling")
        protocols.append(
            {
                "method": observation.get("method"),
                "start_timestep": sampling.get("start_timestep"),
                "initial_noise_scale": sampling.get("initial_noise_scale"),
                "x0_constraint": sampling.get("x0_constraint"),
                "recompute_epsilon_after_x0_constraint": sampling.get(
                    "recompute_epsilon_after_x0_constraint"
                ),
            }
        )
    if len(protocols) != 2 or {row["method"] for row in protocols} != {
        "cofitok",
        "dense_identity",
    }:
        raise ValueError(f"{case_id} matched observation pair is incomplete")
    if any(row["start_timestep"] != 975 for row in protocols):
        raise ValueError(f"{case_id} is not the t=975 diagnostic")
    return {
        "case_id": case_id,
        "passes_shared_recovery_screen": False,
        "methods": summaries,
        "protocols": sorted(protocols, key=lambda row: str(row["method"])),
    }


def build_terminal_snr_reassessment(
    *,
    hold_decision: Mapping[str, Any],
    hold_decision_identity: Mapping[str, Any],
    hold_validation: Mapping[str, Any],
    hold_validation_identity: Mapping[str, Any],
    sampling_recovery_result: Mapping[str, Any],
    sampling_recovery_identity: Mapping[str, Any],
    control_configs: Mapping[str, Mapping[str, Any]],
    control_config_identities: Mapping[str, Mapping[str, Any]],
    intervention_configs: Mapping[str, Mapping[str, Any]],
    intervention_config_identities: Mapping[str, Mapping[str, Any]],
    decision_git: Mapping[str, Any],
    allowed_sampling_output_prefix: str,
) -> dict[str, Any]:
    hold_id = _identity(hold_decision_identity, "capacity hold decision")
    hold_receipt_id = _identity(hold_validation_identity, "capacity hold validation")
    sampling_id = _identity(sampling_recovery_identity, "sampling recovery result")
    validated_hold = validate_hold_recovery_decision(hold_decision)
    validate_hold_recovery_validation(
        hold_validation,
        decision=validated_hold,
        decision_identity=hold_id,
    )
    if _object(hold_validation, "capacity hold validation").get("decision") != hold_id:
        raise ValueError("capacity hold validation decision identity differs")
    hold_sources = _object(validated_hold.get("source_evidence"), "hold sources")
    hold_sampling = _object(hold_sources.get("sampling_recovery"), "hold sampling source")
    if hold_sampling.get("result") != sampling_id:
        raise ValueError("sampling recovery identity differs from the capacity hold")
    sampling_summary = validate_sampling_recovery_result(
        sampling_recovery_result,
        result_path=sampling_id["path"],
        expected_result_sha256=sampling_id["sha256"],
        allowed_output_prefix=allowed_sampling_output_prefix,
    )
    if sampling_summary.get("selection_status") != "no_shared_sampling_recovery_candidate":
        raise ValueError("sampling recovery no-candidate boundary differs")

    config_contract = _config_contract(
        control_configs=control_configs,
        control_identities=control_config_identities,
        intervention_configs=intervention_configs,
        intervention_identities=intervention_config_identities,
    )
    unit_case = _sampling_case_summary(
        sampling_recovery_result, "start975_unit_hard_clip"
    )
    sigma_case = _sampling_case_summary(
        sampling_recovery_result, "start975_sigma_hard_clip"
    )
    control_numerics = _cosine_summary(1_000, CONTROL_ENDPOINT_FRACTION)
    intervention_numerics = _cosine_summary(1_000, SELECTED_ENDPOINT_FRACTION)
    condition_reduction = (
        control_numerics["epsilon_to_x0_condition_factor"]
        / intervention_numerics["epsilon_to_x0_condition_factor"]
    )
    if (
        control_numerics["epsilon_to_x0_condition_factor"] < 10_000.0
        or intervention_numerics["epsilon_to_x0_condition_factor"] > 30.0
        or condition_reduction < 500.0
    ):
        raise ValueError("terminal cosine conditioning evidence differs")

    terminal = _object(
        _object(validated_hold.get("evidence_interpretation"), "hold interpretation").get(
            "terminal_start"
        ),
        "hold terminal-start evidence",
    )
    if (
        terminal.get("all_at_least_0_99") is not True
        or terminal.get("causal_root_cause_is_proven") is not False
    ):
        raise ValueError("terminal clipping evidence differs")

    return {
        "schema_version": REASSESSMENT_SCHEMA,
        "role": REASSESSMENT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "bounded_screen_selected_not_executed",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "decision": "select_one_matched_terminal_snr_intervention_for_preparation",
        "decision_git": _git(decision_git, "terminal-SNR reassessment"),
        "source_evidence": {
            "capacity_hold_decision": hold_id,
            "capacity_hold_validation": hold_receipt_id,
            "sampling_recovery_result": sampling_id,
            "sampling_recovery_replay": sampling_summary,
            "config_contract": config_contract,
        },
        "numerical_discriminator": {
            "scope": "epsilon_to_x0_terminal_conditioning_under_bf16",
            "control": control_numerics,
            "intervention": intervention_numerics,
            "condition_factor_reduction": condition_reduction,
            "interpretation": "The endpoint screen tests a numerically conditioned forward process; this calculation is a selection rationale, not a causal result.",
        },
        "sampling_diagnostic": {
            "selection_status_preserved": "no_shared_sampling_recovery_candidate",
            "start975_unit_hard_clip": unit_case,
            "start975_sigma_hard_clip": sigma_case,
            "interpretation": "Both matched t=975 diagnostics lowered FID directionally but failed their full sampling-only contract; they select an endpoint for fresh matched training and do not authorize reuse as a sampler fix.",
        },
        "candidate_disposition": [
            {
                "id": SELECTED_INTERVENTION,
                "status": "selected_for_bounded_matched_screen",
                "reason": "It preserves direct epsilon-component semantics while testing the t=975 endpoint implicated by the terminal-conditioning and matched sampling evidence.",
            },
            {
                "id": "velocity_prediction",
                "status": "rejected_as_primary_cofitok_intervention",
                "reason": "A velocity-output head would require an input-dependent x_t bypass to recover epsilon, so the restricted token components would no longer sum directly to the dense noise prediction.",
            },
            {
                "id": "x0_prediction",
                "status": "rejected_as_primary_cofitok_intervention",
                "reason": "An x0-output head changes the denoising-token target away from compressed negative-noise components.",
            },
            {
                "id": "another_scalar_timestep_weighting",
                "status": "rejected_without_new_discriminator",
                "reason": "The matched Min-SNR gamma-5 pilot already failed to identify a shared weighting candidate.",
            },
            {
                "id": "sampling_only_nonterminal_start",
                "status": "rejected_by_completed_screen",
                "reason": "The immutable sampling recovery result has no shared candidate and remains non-authorizing.",
            },
            {
                "id": "exposure_or_capacity_only",
                "status": "rejected_by_completed_screens",
                "reason": "The matched exposure and four-arm capacity stages completed without clearing their predeclared scientific gates.",
            },
        ],
        "selected_intervention": {
            "id": SELECTED_INTERVENTION,
            "single_scientific_config_field": "diffusion.cosine_endpoint_fraction",
            "control_value": CONTROL_ENDPOINT_FRACTION,
            "intervention_value": SELECTED_ENDPOINT_FRACTION,
            "prediction_target": "epsilon",
            "token_component_semantics": "dense_epsilon_components",
            "applies_identically_to": ["cofitok", "dense_identity"],
            "fresh_training_required": True,
            "checkpoint_resume_allowed": False,
        },
        "bounded_screen_contract": {
            "design": "fresh_same_revision_four_arm_control_vs_endpoint0975",
            "methods": ["cofitok", "dense_identity"],
            "conditions": ["cosine_endpoint_1p0", "cosine_endpoint_0p975"],
            "training_steps_per_arm": 10_000,
            "samples_seen_per_arm": 640_000,
            "evaluation_samples_per_arm": 1_000,
            "sampler": "ddim",
            "sample_steps": 100,
            "weights": "ema",
            "guidance_scale": 1.5,
            "precision": "bf16",
            "seed": 2027,
            "thresholds": copy.deepcopy(SCREEN_THRESHOLDS),
            "recall_role": "directional_non_regression_only_at_1000_samples",
            "independent_frozen_confirmation_required_after_pass": True,
        },
        "next_stage": {
            "route": "build_separate_source_bound_terminal_snr_screen_preparation",
            "intervention_selected": True,
            "source_bound_preparation_may_be_built": True,
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "claim_policy": {
            "terminal_conditioning_is_proven_root_cause": False,
            "sampling_only_recovery_claim_allowed": False,
            "bounded_screen_pass_claim_allowed": False,
            "formal_generation_claim_allowed": False,
            "generation_advantage_proven": False,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_terminal_snr_reassessment(report: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(report, "terminal-SNR reassessment")
    intervention = _object(row.get("selected_intervention"), "selected intervention")
    screen = _object(row.get("bounded_screen_contract"), "bounded screen")
    next_stage = _object(row.get("next_stage"), "next stage")
    claims = _object(row.get("claim_policy"), "claim policy")
    if (
        row.get("schema_version") != REASSESSMENT_SCHEMA
        or row.get("role") != REASSESSMENT_ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("scientific_status") != "bounded_screen_selected_not_executed"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("decision")
        != "select_one_matched_terminal_snr_intervention_for_preparation"
        or row.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or intervention.get("id") != SELECTED_INTERVENTION
        or intervention.get("single_scientific_config_field")
        != "diffusion.cosine_endpoint_fraction"
        or intervention.get("control_value") != CONTROL_ENDPOINT_FRACTION
        or intervention.get("intervention_value") != SELECTED_ENDPOINT_FRACTION
        or intervention.get("prediction_target") != "epsilon"
        or intervention.get("token_component_semantics")
        != "dense_epsilon_components"
        or intervention.get("fresh_training_required") is not True
        or intervention.get("checkpoint_resume_allowed") is not False
        or screen.get("design")
        != "fresh_same_revision_four_arm_control_vs_endpoint0975"
        or screen.get("training_steps_per_arm") != 10_000
        or screen.get("evaluation_samples_per_arm") != 1_000
        or screen.get("thresholds") != SCREEN_THRESHOLDS
        or screen.get("independent_frozen_confirmation_required_after_pass")
        is not True
        or next_stage.get("intervention_selected") is not True
        or next_stage.get("source_bound_preparation_may_be_built") is not True
        or next_stage.get("execution_ready") is not False
        or next_stage.get("gpu_execution_allowed") is not False
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or claims.get("terminal_conditioning_is_proven_root_cause") is not False
        or claims.get("bounded_screen_pass_claim_allowed") is not False
        or claims.get("formal_generation_claim_allowed") is not False
    ):
        raise ValueError("terminal-SNR reassessment contract differs")
    _git(row.get("decision_git"), "terminal-SNR reassessment")
    sources = _object(row.get("source_evidence"), "terminal-SNR sources")
    if set(sources) != {
        "capacity_hold_decision",
        "capacity_hold_validation",
        "sampling_recovery_result",
        "sampling_recovery_replay",
        "config_contract",
    }:
        raise ValueError("terminal-SNR source set differs")
    for field in (
        "capacity_hold_decision",
        "capacity_hold_validation",
        "sampling_recovery_result",
    ):
        _identity(sources[field], field)
    return copy.deepcopy(row)


def build_terminal_snr_reassessment_validation(
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_terminal_snr_reassessment(decision)
    decision_id = _identity(decision_identity, "terminal-SNR reassessment")
    validator = _git(validator_git, "terminal-SNR validator")
    basis = {
        "decision": decision_id,
        "decision_git": validated["decision_git"],
        "validator_git": validator,
        "source_evidence": validated["source_evidence"],
        "selected_intervention": validated["selected_intervention"],
        "bounded_screen_contract": validated["bounded_screen_contract"],
    }
    return {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "scientific_status": "bounded_screen_selected_not_executed",
        "decision": decision_id,
        "decision_git": copy.deepcopy(validated["decision_git"]),
        "validator_git": validator,
        "selected_intervention": copy.deepcopy(validated["selected_intervention"]),
        "bounded_screen_contract": copy.deepcopy(validated["bounded_screen_contract"]),
        "source_evidence": copy.deepcopy(validated["source_evidence"]),
        "validation_basis_sha256": _canonical_sha256(basis),
        "generation_advantage_proven": False,
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_terminal_snr_reassessment_validation(
    receipt: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(receipt, "terminal-SNR validation")
    validator = _git(row.get("validator_git"), "terminal-SNR validator")
    expected = build_terminal_snr_reassessment_validation(
        decision=decision,
        decision_identity=decision_identity,
        validator_git=validator,
    )
    if row != expected:
        raise ValueError("terminal-SNR validation is not reproducible")
    return expected


__all__ = [
    "AUTHORIZATION_BOUNDARY",
    "BF16_SPACING_AT_ONE",
    "CONTROL_ENDPOINT_FRACTION",
    "REASSESSMENT_ROLE",
    "REASSESSMENT_SCHEMA",
    "SCREEN_THRESHOLDS",
    "SELECTED_ENDPOINT_FRACTION",
    "SELECTED_INTERVENTION",
    "VALIDATION_ROLE",
    "VALIDATION_SCHEMA",
    "build_terminal_snr_reassessment",
    "build_terminal_snr_reassessment_validation",
    "validate_terminal_snr_reassessment",
    "validate_terminal_snr_reassessment_validation",
]
