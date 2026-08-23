from __future__ import annotations

from typing import Any

import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule, select_sampling_timesteps
from cofitok.generation.protocol import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA


EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA = (
    "cofitok_matched_epsilon_stability_sampling_design_v1"
)


def _schedule_point(
    schedule: DiffusionSchedule,
    timestep: int,
) -> dict[str, float | int]:
    timestep_tensor = torch.tensor([timestep])
    return {
        "timestep": timestep,
        "alpha_bar": round(float(schedule.alphas_cumprod[timestep]), 12),
        "sqrt_alpha_bar": round(
            float(schedule.sqrt_alphas_cumprod[timestep]),
            12,
        ),
        "schedule_sigma": round(
            float(schedule.sqrt_one_minus_alphas_cumprod[timestep]),
            12,
        ),
        "snr": round(float(schedule.snr(timestep_tensor)[0]), 12),
    }


def _case(
    *,
    case_id: str,
    parent_case_id: str | None,
    incremental_control: str | None,
    requested_start_timestep: int | None,
    scale_initial_noise_by_sigma: bool,
    x0_constraint: str,
    dynamic_threshold_percentile: float,
    recompute_epsilon_after_x0_constraint: bool,
    attribution_role: str,
) -> dict[str, Any]:
    effective_start_timestep = (
        999 if requested_start_timestep is None else requested_start_timestep
    )
    return {
        "case_id": case_id,
        "parent_case_id": parent_case_id,
        "incremental_control": incremental_control,
        "attribution_role": attribution_role,
        "sampling_controls": {
            "requested_start_timestep": requested_start_timestep,
            "start_timestep": effective_start_timestep,
            "actual_timesteps": select_sampling_timesteps(
                1000,
                100,
                start_timestep=effective_start_timestep,
            ),
            "scale_initial_noise_by_sigma": scale_initial_noise_by_sigma,
            "initial_noise_scale": (
                "schedule_sigma" if scale_initial_noise_by_sigma else "unit"
            ),
            "clip_x0": True,
            "x0_constraint": x0_constraint,
            "dynamic_threshold_percentile": dynamic_threshold_percentile,
            "recompute_epsilon_after_x0_constraint": (
                recompute_epsilon_after_x0_constraint
            ),
        },
    }


def build_epsilon_stability_sampling_design() -> dict[str, Any]:
    """Build a deterministic, permanently non-authorizing diagnostic design."""

    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=1000, schedule_type="cosine"),
        device="cpu",
    )
    cases = [
        _case(
            case_id="legacy_terminal_hard_clip",
            parent_case_id=None,
            incremental_control=None,
            requested_start_timestep=None,
            scale_initial_noise_by_sigma=False,
            x0_constraint="clip",
            dynamic_threshold_percentile=0.0,
            recompute_epsilon_after_x0_constraint=False,
            attribution_role="matched_legacy_reference",
        ),
        _case(
            case_id="start975_unit_hard_clip",
            parent_case_id="legacy_terminal_hard_clip",
            incremental_control="nonterminal_start",
            requested_start_timestep=975,
            scale_initial_noise_by_sigma=False,
            x0_constraint="clip",
            dynamic_threshold_percentile=0.0,
            recompute_epsilon_after_x0_constraint=False,
            attribution_role="single_factor_nonterminal_start",
        ),
        _case(
            case_id="start975_sigma_hard_clip",
            parent_case_id="start975_unit_hard_clip",
            incremental_control="schedule_sigma_initialization",
            requested_start_timestep=975,
            scale_initial_noise_by_sigma=True,
            x0_constraint="clip",
            dynamic_threshold_percentile=0.0,
            recompute_epsilon_after_x0_constraint=False,
            attribution_role="nested_initial_noise_scale",
        ),
        _case(
            case_id="terminal_dynamic_threshold",
            parent_case_id="legacy_terminal_hard_clip",
            incremental_control="dynamic_threshold",
            requested_start_timestep=None,
            scale_initial_noise_by_sigma=False,
            x0_constraint="dynamic_threshold",
            dynamic_threshold_percentile=0.995,
            recompute_epsilon_after_x0_constraint=False,
            attribution_role="single_factor_x0_constraint",
        ),
        _case(
            case_id="terminal_dynamic_threshold_recompute",
            parent_case_id="terminal_dynamic_threshold",
            incremental_control="epsilon_after_x0_constraint",
            requested_start_timestep=None,
            scale_initial_noise_by_sigma=False,
            x0_constraint="dynamic_threshold",
            dynamic_threshold_percentile=0.995,
            recompute_epsilon_after_x0_constraint=True,
            attribution_role="nested_dynamic_threshold_consistency",
        ),
        _case(
            case_id="terminal_hard_clip_recompute",
            parent_case_id="legacy_terminal_hard_clip",
            incremental_control="epsilon_after_x0_constraint",
            requested_start_timestep=None,
            scale_initial_noise_by_sigma=False,
            x0_constraint="clip",
            dynamic_threshold_percentile=0.0,
            recompute_epsilon_after_x0_constraint=True,
            attribution_role="single_factor_hard_clip_consistency",
        ),
        _case(
            case_id="start975_sigma_dynamic_threshold",
            parent_case_id="start975_sigma_hard_clip",
            incremental_control="dynamic_threshold",
            requested_start_timestep=975,
            scale_initial_noise_by_sigma=True,
            x0_constraint="dynamic_threshold",
            dynamic_threshold_percentile=0.995,
            recompute_epsilon_after_x0_constraint=False,
            attribution_role="nested_nonterminal_x0_constraint",
        ),
        _case(
            case_id="start975_sigma_dynamic_threshold_recompute",
            parent_case_id="start975_sigma_dynamic_threshold",
            incremental_control="epsilon_after_x0_constraint",
            requested_start_timestep=975,
            scale_initial_noise_by_sigma=True,
            x0_constraint="dynamic_threshold",
            dynamic_threshold_percentile=0.995,
            recompute_epsilon_after_x0_constraint=True,
            attribution_role="combined_recovery_upper_bound",
        ),
    ]
    return {
        "schema": EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA,
        "status": "design_only",
        "execution_ready": False,
        "scientific_role": "matched_sampling_only_factorization_diagnostic",
        "common_sampling_contract": {
            "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
            "inference_api": INFERENCE_API,
            "sampler": "ddim",
            "num_train_timesteps": 1000,
            "sample_steps": 100,
            "sample_count_per_method_case": 1000,
            "total_case_count": len(cases),
            "total_generated_image_count": len(cases) * 2 * 1000,
            "batch_size": 32,
            "weights": "ema",
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "eta": 0.0,
            "precision": "bf16",
            "class_schedule": "balanced_modulo",
            "class_coverage": {
                "num_classes": 1000,
                "samples_per_class": 1,
            },
            "random_stream": {
                "shared_across_methods": True,
                "shared_across_cases": True,
                "batch_size_invariant": True,
                "resume_index_invariant": True,
                "fresh_namespace_required": True,
                "seed": None,
                "start_index": None,
            },
        },
        "matched_methods": {
            "cofitok": {
                "role": "matched_training_direct",
                "checkpoint_step": 100000,
                "prefix_budget": 8,
            },
            "dense_identity": {
                "role": "matched_training_direct",
                "checkpoint_step": 100000,
                "prefix_budget": 1,
            },
        },
        "schedule_diagnostics": {
            "legacy_terminal": _schedule_point(schedule, 999),
            "conservative_nonterminal": _schedule_point(schedule, 975),
        },
        "cases": cases,
        "evaluation_contract": {
            "same_real_set_required": True,
            "same_evaluator_required": True,
            "same_classifier_required": True,
            "required_metrics": [
                "fid",
                "inception_score",
                "class_top1",
                "class_top5",
                "channel_saturation_fraction",
                "total_variation",
                "median_filter_residual_fraction",
            ],
            "precision_recall_role": "descriptive_only_at_1000_samples",
            "paired_visual_indices_required": True,
        },
        "causal_boundary": {
            "sampling_only": True,
            "checkpoint_weights_immutable": True,
            "min_snr_training_tested": False,
            "fresh_training_allowed": False,
            "single_factor_claims_require_parent_case_comparison": True,
            "combined_case_role": "recovery_feasibility_not_single_factor_attribution",
            "one_thousand_sample_results": "screening_only",
            "independent_matched_10000_confirmation_required": True,
        },
        "required_execution_bindings": [
            "terminal_route_receipt_identity",
            "separate_execution_authorization_identity",
            "repair_code_revision_and_tree",
            "cofitok_checkpoint_payload_and_sidecar_identity",
            "dense_checkpoint_payload_and_sidecar_identity",
            "paired_training_report_identity",
            "dataset_and_real_set_identity",
            "runtime_environment_identity",
            "evaluator_and_classifier_identity",
            "fresh_random_stream_namespace",
            "versioned_non_overlapping_output_root",
        ],
        "blocking_reasons": [
            "terminal_route_receipt_not_bound",
            "execution_authorization_not_bound",
            "source_identities_not_bound",
            "fresh_random_stream_not_bound",
            "output_root_not_bound",
        ],
        "authorization_boundary": {
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
            "evaluation_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "inference_export_allowed": False,
            "release_allowed": False,
            "process_signals_allowed": False,
        },
    }


def materialize_epsilon_stability_case_protocol(
    design: dict[str, Any],
    case_id: str,
    method: str,
    *,
    seed: int,
    start_index: int,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or start_index < 0
    ):
        raise ValueError("start_index must be a non-negative integer")
    if design.get("schema") != EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA:
        raise ValueError("epsilon-stability sampling design schema mismatch")
    cases = {
        str(case["case_id"]): case
        for case in design.get("cases", [])
        if isinstance(case, dict) and "case_id" in case
    }
    if case_id not in cases:
        raise ValueError(f"unknown epsilon-stability case: {case_id}")
    methods = design.get("matched_methods", {})
    if method not in methods:
        raise ValueError(f"unknown epsilon-stability method: {method}")
    common = design["common_sampling_contract"]
    controls = cases[case_id]["sampling_controls"]
    return {
        "protocol_schema": common["protocol_schema"],
        "inference_api": common["inference_api"],
        "sampler": common["sampler"],
        "num_samples": common["sample_count_per_method_case"],
        "num_train_timesteps": common["num_train_timesteps"],
        "sample_steps": common["sample_steps"],
        "actual_timesteps": controls["actual_timesteps"],
        "batch_size": common["batch_size"],
        "weights": common["weights"],
        "prefix_budgets": [methods[method]["prefix_budget"]],
        "guidance_scale": common["guidance_scale"],
        "guidance_rescale": common["guidance_rescale"],
        "cfg_batch_mode": common["cfg_batch_mode"],
        "eta": common["eta"],
        "clip_x0": controls["clip_x0"],
        "x0_constraint": controls["x0_constraint"],
        "dynamic_threshold_percentile": controls[
            "dynamic_threshold_percentile"
        ],
        "requested_start_timestep": controls["requested_start_timestep"],
        "start_timestep": controls["start_timestep"],
        "scale_initial_noise_by_sigma": controls[
            "scale_initial_noise_by_sigma"
        ],
        "recompute_epsilon_after_x0_constraint": controls[
            "recompute_epsilon_after_x0_constraint"
        ],
        "initial_noise_scale": controls["initial_noise_scale"],
        "precision": common["precision"],
        "seed": seed,
        "start_index": start_index,
        "class_schedule": common["class_schedule"],
        "random_stream": {
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": common["random_stream"][
                "batch_size_invariant"
            ],
            "resume_index_invariant": common["random_stream"][
                "resume_index_invariant"
            ],
        },
    }
