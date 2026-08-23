from __future__ import annotations

import math
from typing import Any

from cofitok.diffusion import select_sampling_timesteps

SAMPLING_PROTOCOL_SCHEMA = "cofitok_ddim_sampling_v1"
SAMPLING_MANIFEST_SCHEMA_VERSION = 3
SAMPLING_REPORT_SCHEMA_VERSION = 6
INFERENCE_API = {
    "name": "cofitok.generation.GenerationSession",
    "version": 1,
}


def sampling_protocol_contract(
    sampling: dict[str, Any],
    *,
    stage: str | None = None,
    expected_num_train_timesteps: int | None = None,
) -> dict[str, Any]:
    if stage not in {None, "milestone", "scaling", "full"}:
        raise ValueError("stage must be milestone, scaling, full, or None")
    issues: list[str] = []
    if sampling.get("protocol_schema") != SAMPLING_PROTOCOL_SCHEMA:
        issues.append("protocol_schema")
    if sampling.get("inference_api") != INFERENCE_API:
        issues.append("inference_api")
    if sampling.get("sampler") != "ddim":
        issues.append("sampler")

    try:
        num_train_timesteps = int(sampling.get("num_train_timesteps", -1))
        sample_steps = int(sampling.get("sample_steps", -1))
    except (TypeError, ValueError):
        num_train_timesteps = -1
        sample_steps = -1
    if num_train_timesteps < 1:
        issues.append("num_train_timesteps")
    if sample_steps < 1:
        issues.append("sample_steps")
    start_timestep = sampling.get("start_timestep")
    if start_timestep is not None and (
        isinstance(start_timestep, bool) or not isinstance(start_timestep, int)
    ):
        issues.append("start_timestep")
        start_timestep = None
    try:
        expected_timesteps = (
            select_sampling_timesteps(
                num_train_timesteps,
                sample_steps,
                start_timestep=start_timestep,
            )
            if num_train_timesteps > 0 and sample_steps > 0
            else None
        )
    except ValueError:
        issues.append("start_timestep")
        expected_timesteps = None
    if sampling.get("actual_timesteps") != expected_timesteps:
        issues.append("actual_timesteps")
    if not isinstance(sampling.get("clip_x0"), bool):
        issues.append("clip_x0")
    x0_constraint = sampling.get("x0_constraint")
    if x0_constraint is not None and x0_constraint not in {
        "clip",
        "dynamic_threshold",
        "none",
    }:
        issues.append("x0_constraint")
    try:
        dynamic_threshold_percentile = float(
            sampling.get("dynamic_threshold_percentile", 0.0)
        )
    except (TypeError, ValueError):
        dynamic_threshold_percentile = math.nan
    if not math.isfinite(dynamic_threshold_percentile) or (
        dynamic_threshold_percentile != 0.0
        and not 0.5 <= dynamic_threshold_percentile < 1.0
    ):
        issues.append("dynamic_threshold_percentile")
    if x0_constraint == "dynamic_threshold" and dynamic_threshold_percentile <= 0.0:
        issues.append("x0_constraint.dynamic_threshold")
    if x0_constraint == "dynamic_threshold" and sampling.get("clip_x0") is not True:
        issues.append("x0_constraint.dynamic_threshold")
    if dynamic_threshold_percentile > 0.0 and x0_constraint != "dynamic_threshold":
        issues.append("dynamic_threshold_percentile.binding")
    if x0_constraint == "clip" and sampling.get("clip_x0") is not True:
        issues.append("x0_constraint.clip")
    if x0_constraint == "none" and sampling.get("clip_x0") is not False:
        issues.append("x0_constraint.none")
    initial_noise_scale = sampling.get("initial_noise_scale")
    if initial_noise_scale is not None and initial_noise_scale not in {
        "unit",
        "schedule_sigma",
    }:
        issues.append("initial_noise_scale")
    scale_initial_noise = sampling.get("scale_initial_noise_by_sigma")
    if scale_initial_noise is not None and not isinstance(scale_initial_noise, bool):
        issues.append("scale_initial_noise_by_sigma")
    if scale_initial_noise is True and initial_noise_scale != "schedule_sigma":
        issues.append("initial_noise_scale.binding")
    if scale_initial_noise is False and initial_noise_scale not in {None, "unit"}:
        issues.append("initial_noise_scale.binding")
    if initial_noise_scale == "schedule_sigma" and scale_initial_noise is not True:
        issues.append("initial_noise_scale.binding")
    requested_start_timestep = sampling.get("requested_start_timestep")
    if requested_start_timestep is not None and (
        isinstance(requested_start_timestep, bool)
        or not isinstance(requested_start_timestep, int)
        or requested_start_timestep != start_timestep
    ):
        issues.append("requested_start_timestep")
    if sampling.get("precision") not in {"fp32", "bf16", "fp16"}:
        issues.append("precision")
    for field in ("guidance_scale", "guidance_rescale", "eta"):
        try:
            value = float(sampling.get(field, math.nan))
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            issues.append(field)

    expected: dict[str, Any] = {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
    }
    if expected_num_train_timesteps is not None:
        expected["num_train_timesteps"] = expected_num_train_timesteps
        if num_train_timesteps != expected_num_train_timesteps:
            issues.append("matched_num_train_timesteps")
    if stage is not None:
        stage_samples = {
            "milestone": 2_048,
            "scaling": 10_000,
            "full": 50_000,
        }
        stage_steps = {
            "milestone": 50,
            "scaling": 100,
            "full": 250,
        }
        expected.update(
            {
                "num_samples": stage_samples[stage],
                "sample_steps": stage_steps[stage],
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "clip_x0": True,
                "precision": "bf16",
                "seed": 0,
                "start_index": 0,
                "class_schedule": "balanced_modulo",
            }
        )
        if stage == "milestone":
            expected["batch_size"] = 32
        for field, expected_value in expected.items():
            if field in {
                "protocol_schema",
                "inference_api",
                "sampler",
                "num_train_timesteps",
            }:
                continue
            if sampling.get(field) != expected_value:
                issues.append(f"formal_{field}")
        if sampling.get("start_timestep") not in {
            None,
            num_train_timesteps - 1,
        }:
            issues.append("formal_start_timestep")
        if sampling.get("scale_initial_noise_by_sigma") not in {None, False}:
            issues.append("formal_scale_initial_noise_by_sigma")
        if sampling.get("x0_constraint") not in {None, "clip"}:
            issues.append("formal_x0_constraint")
        if dynamic_threshold_percentile != 0.0:
            issues.append("formal_dynamic_threshold_percentile")
        random_stream = sampling.get("random_stream", {})
        for field in (
            "prefix_budgets_share_stream",
            "batch_size_invariant",
            "resume_index_invariant",
        ):
            if random_stream.get(field) is not True:
                issues.append(f"random_stream.{field}")

    return {
        "valid": not issues,
        "stage": stage,
        "issues": sorted(set(issues)),
        "expected": expected,
        "actual": {
            key: sampling.get(key)
            for key in (
                "protocol_schema",
                "inference_api",
                "sampler",
                "num_samples",
                "num_train_timesteps",
                "sample_steps",
                "actual_timesteps",
                "guidance_scale",
                "guidance_rescale",
                "cfg_batch_mode",
                "eta",
                "clip_x0",
                "x0_constraint",
                "dynamic_threshold_percentile",
                "requested_start_timestep",
                "start_timestep",
                "scale_initial_noise_by_sigma",
                "initial_noise_scale",
                "precision",
                "seed",
                "start_index",
                "class_schedule",
                "random_stream",
            )
        },
    }
