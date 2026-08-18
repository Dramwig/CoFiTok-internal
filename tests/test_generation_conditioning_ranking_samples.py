from __future__ import annotations

import copy

import pytest

from scripts.evaluate_generation_conditioning_ranking_samples import (
    EXPECTED_NUM_SAMPLES,
    EXPECTED_SAMPLE_STEPS,
    EXPECTED_SEED,
    paired_class_fidelity_summary,
    validate_sampling_pair,
)


def _row(index: int, *, delta: float = 0.1) -> dict:
    control_log = -7.0
    ranked_log = control_log + delta
    return {
        "sample_index": index,
        "requested_class": index % 1000,
        "control": {
            "target_log_probability": control_log,
            "target_probability": 0.001,
            "predicted_class": index % 1000,
            "top1_correct": index % 4 == 0,
            "top5_correct": index % 2 == 0,
        },
        "ranked": {
            "target_log_probability": ranked_log,
            "target_probability": 0.0011,
            "predicted_class": index % 1000,
            "top1_correct": index % 4 == 0,
            "top5_correct": index % 2 == 0,
        },
        "ranked_minus_control_target_log_probability": delta,
    }


def test_paired_summary_requires_practical_and_significant_shared_direction() -> None:
    report = paired_class_fidelity_summary([_row(index) for index in range(8)])

    assert report["paired_sign_test"] == {
        "positive": 8,
        "negative": 0,
        "ties": 0,
        "one_sided_pvalue": pytest.approx(1 / 256),
        "alternative": "ranked_target_log_probability_greater_than_control",
    }
    assert report["ranked_minus_control"]["mean_target_log_probability"] == pytest.approx(
        0.1
    )
    assert report["ranked_minus_control"]["target_probability_ratio"] == pytest.approx(
        1.1
    )
    assert report["class_alignment_pass"] is True


def test_paired_summary_rejects_tiny_or_asymmetric_direction() -> None:
    tiny = paired_class_fidelity_summary([_row(index, delta=0.001) for index in range(8)])
    assert tiny["gates"]["mean_target_log_probability_delta"] is False
    assert tiny["class_alignment_pass"] is False

    rows = [_row(index, delta=0.1 if index < 4 else -0.1) for index in range(8)]
    mixed = paired_class_fidelity_summary(rows)
    assert mixed["gates"]["paired_target_log_probability_significant"] is False
    assert mixed["class_alignment_pass"] is False


def test_paired_summary_recomputes_rows_and_rejects_index_or_delta_drift() -> None:
    index_drift = [_row(index) for index in range(8)]
    index_drift[3]["sample_index"] = 30
    with pytest.raises(ValueError, match="indices"):
        paired_class_fidelity_summary(index_drift)

    delta_drift = [_row(index) for index in range(8)]
    delta_drift[0]["ranked_minus_control_target_log_probability"] = 99.0
    with pytest.raises(ValueError, match="delta"):
        paired_class_fidelity_summary(delta_drift)


def _sampling_source(*, budget: int, checkpoint: str) -> dict:
    sampling = {
        "num_samples": EXPECTED_NUM_SAMPLES,
        "start_index": 0,
        "sample_steps": EXPECTED_SAMPLE_STEPS,
        "num_train_timesteps": 1000,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "seed": EXPECTED_SEED,
        "precision": "bf16",
        "class_schedule": "balanced_modulo",
        "sampler": "ddim",
        "clip_x0": True,
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "prefix_budgets": [budget],
        "image_shape": [3, 256, 256],
        "actual_timesteps": list(range(999, -1, -20))[:EXPECTED_SAMPLE_STEPS],
    }
    from cofitok.diffusion import select_sampling_timesteps

    sampling["actual_timesteps"] = select_sampling_timesteps(1000, EXPECTED_SAMPLE_STEPS)
    return {
        "sampling": sampling,
        "weights": "ema",
        "checkpoint_step": 1000,
        "selected_prefix_budget": budget,
        "checkpoint_sha256": checkpoint * 64,
        "git": {
            "revision": "a" * 40,
            "branch": "scale/generation-label-ranking-standing-authorization-v1",
            "tracked_dirty": False,
        },
        "runtime_environment": {"torch": "test"},
        "runtime_environment_sha256": "e" * 64,
    }


def test_sampling_pair_requires_exact_matched_protocol_and_environment() -> None:
    control = _sampling_source(budget=8, checkpoint="1")
    ranked = _sampling_source(budget=8, checkpoint="2")

    contract = validate_sampling_pair(control, ranked, method="cofitok")
    assert contract["prefix_budget"] == 8
    assert contract["sampling"]["num_samples"] == 5000

    protocol_drift = copy.deepcopy(ranked)
    protocol_drift["sampling"]["seed"] += 1
    with pytest.raises(ValueError, match="protocol"):
        validate_sampling_pair(control, protocol_drift, method="cofitok")

    environment_drift = copy.deepcopy(ranked)
    environment_drift["runtime_environment"] = {"torch": "drift"}
    with pytest.raises(ValueError, match="environments"):
        validate_sampling_pair(control, environment_drift, method="cofitok")


def test_sampling_pair_rejects_wrong_budget_or_identical_checkpoints() -> None:
    control = _sampling_source(budget=1, checkpoint="1")
    ranked = _sampling_source(budget=1, checkpoint="2")
    assert validate_sampling_pair(control, ranked, method="dense_identity")[
        "prefix_budget"
    ] == 1

    with pytest.raises(ValueError, match="protocol"):
        validate_sampling_pair(control, ranked, method="cofitok")

    identical = copy.deepcopy(control)
    with pytest.raises(ValueError, match="distinct"):
        validate_sampling_pair(control, identical, method="dense_identity")
