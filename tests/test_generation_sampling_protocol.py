from __future__ import annotations

import copy

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.generation import (
    INFERENCE_API,
    SAMPLING_PROTOCOL_SCHEMA,
    sampling_protocol_contract,
)


def _sampling(stage: str) -> dict:
    sample_steps = {"milestone": 50, "scaling": 100, "full": 250}[stage]
    return {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
        "num_samples": {"milestone": 2_048, "scaling": 10_000, "full": 50_000}[
            stage
        ],
        "start_index": 0,
        "batch_size": 32,
        "sample_steps": sample_steps,
        "num_train_timesteps": 1000,
        "actual_timesteps": select_sampling_timesteps(1000, sample_steps),
        "prefix_budgets": [8],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "class_schedule": "balanced_modulo",
        "random_stream": {
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
    }


@pytest.mark.parametrize("stage", ["milestone", "scaling", "full"])
def test_formal_sampling_protocol_accepts_exact_stage_contract(stage: str) -> None:
    contract = sampling_protocol_contract(
        _sampling(stage),
        stage=stage,
        expected_num_train_timesteps=1000,
    )

    assert contract["valid"] is True
    assert contract["issues"] == []


@pytest.mark.parametrize(
    ("field", "value", "issue"),
    [
        ("sampler", "ddpm", "sampler"),
        ("num_samples", 2_048, "formal_num_samples"),
        ("sample_steps", 20, "actual_timesteps"),
        ("clip_x0", False, "formal_clip_x0"),
        (
            "recompute_epsilon_after_x0_constraint",
            True,
            "formal_recompute_epsilon_after_x0_constraint",
        ),
        ("guidance_scale", 2.0, "formal_guidance_scale"),
        ("precision", "fp32", "formal_precision"),
    ],
)
def test_formal_sampling_protocol_rejects_weakened_or_drifted_fields(
    field: str,
    value: object,
    issue: str,
) -> None:
    sampling = copy.deepcopy(_sampling("full"))
    sampling[field] = value

    contract = sampling_protocol_contract(
        sampling,
        stage="full",
        expected_num_train_timesteps=1000,
    )

    assert contract["valid"] is False
    assert issue in contract["issues"]
