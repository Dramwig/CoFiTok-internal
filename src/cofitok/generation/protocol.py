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
    if stage not in {None, "scaling", "full"}:
        raise ValueError("stage must be scaling, full, or None")
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
    expected_timesteps = (
        select_sampling_timesteps(num_train_timesteps, sample_steps)
        if num_train_timesteps > 0 and sample_steps > 0
        else None
    )
    if sampling.get("actual_timesteps") != expected_timesteps:
        issues.append("actual_timesteps")
    if not isinstance(sampling.get("clip_x0"), bool):
        issues.append("clip_x0")
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
        expected.update(
            {
                "num_samples": 10_000 if stage == "scaling" else 50_000,
                "sample_steps": 100 if stage == "scaling" else 250,
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
                "precision",
                "seed",
                "start_index",
                "class_schedule",
                "random_stream",
            )
        },
    }
