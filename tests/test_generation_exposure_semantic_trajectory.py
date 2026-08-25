from __future__ import annotations

from typing import Any

import pytest

from cofitok.generation.exposure_semantic_trajectory import (
    CLAIM_BOUNDARY,
    ELIGIBLE_TIMESTEPS,
)
from scripts.build_generation_exposure_semantic_trajectory_report import (
    _json_document,
    aggregate_checkpoint,
    classify_decision,
    exposure_trajectory,
    summarize_positive,
)


def test_json_document_matches_written_report_shape() -> None:
    normalized = _json_document(
        {
            "steps": (1_250, 2_500, 5_000),
            "by_step": {1_250: {"eligible_timesteps": (500, 700, 900)}},
        }
    )

    assert normalized == {
        "steps": [1_250, 2_500, 5_000],
        "by_step": {"1250": {"eligible_timesteps": [500, 700, 900]}},
    }


def _sensitivity_report(*, correct_mse: float, advantage: float) -> dict[str, Any]:
    rows = []
    for sample_index in range(16):
        scale = 1.0 + sample_index * 0.001
        correct = correct_mse * scale
        reference = correct / (1.0 - advantage)
        for timestep in (100, 500, 700, 900):
            rows.append(
                {
                    "sample_index": sample_index,
                    "timestep": timestep,
                    "correct_label": 128 + sample_index,
                    "wrong_label": 378 + sample_index,
                    "noise_seed": 314159 + sample_index,
                    "conditions": {
                        "correct": {"epsilon_mse_to_noise": correct},
                        "wrong": {"epsilon_mse_to_noise": reference},
                        "null": {"epsilon_mse_to_noise": reference},
                    },
                }
            )
    return {"sample_rows": rows}


def _method_result(*, semantic: bool, denoising: bool) -> dict[str, Any]:
    if semantic:
        advantages = {1_250: 0.01, 2_500: 0.03, 5_000: 0.08}
    else:
        advantages = {1_250: -0.03, 2_500: -0.03, 5_000: -0.03}
    mse = (
        {1_250: 1.0, 2_500: 0.8, 5_000: 0.6}
        if denoising
        else {1_250: 1.0, 2_500: 1.1, 5_000: 1.2}
    )
    checkpoints = {
        step: aggregate_checkpoint(
            _sensitivity_report(correct_mse=mse[step], advantage=advantages[step])
        )
        for step in (1_250, 2_500, 5_000)
    }
    return {
        "checkpoints": checkpoints,
        "trajectory": exposure_trajectory(checkpoints),
    }


def test_sign_test_uses_images_not_timestep_rows() -> None:
    summary = summarize_positive([1.0] * 12 + [-1.0] * 4)

    assert summary["positive_count"] == 12
    assert summary["one_sided_sign_test_pvalue"] < 0.05
    assert summarize_positive([1.0] * 11 + [-1.0] * 5)[
        "one_sided_sign_test_pvalue"
    ] > 0.05


def test_exposure_trajectory_aggregates_eligible_timesteps_per_image() -> None:
    checkpoints = {
        1_250: aggregate_checkpoint(
            _sensitivity_report(correct_mse=1.0, advantage=0.01)
        ),
        2_500: aggregate_checkpoint(
            _sensitivity_report(correct_mse=0.8, advantage=0.03)
        ),
        5_000: aggregate_checkpoint(
            _sensitivity_report(correct_mse=0.6, advantage=0.08)
        ),
    }

    trajectory = exposure_trajectory(checkpoints)

    assert checkpoints[1_250]["sample_count"] == 16
    assert checkpoints[1_250]["eligible_row_count"] == 48
    assert checkpoints[1_250]["per_sample"][0]["eligible_timesteps"] == (
        ELIGIBLE_TIMESTEPS
    )
    assert trajectory["semantic_recovery"] is True
    assert trajectory["gates"] == {
        "denoising_mse_improves_with_exposure": True,
        "final_absolute_semantic_direction": True,
        "semantic_advantage_improves_with_exposure": True,
    }


@pytest.mark.parametrize(
    ("cofitok", "dense", "category"),
    [
        (
            (True, True),
            (True, True),
            "shared_exposure_semantic_recovery",
        ),
        (
            (False, True),
            (False, True),
            "denoising_improves_but_semantic_alignment_does_not",
        ),
        ((True, True), (False, True), "method_asymmetry"),
        ((False, False), (False, False), "no_exposure_recovery"),
    ],
)
def test_decision_categories_remain_mutually_exclusive(
    cofitok: tuple[bool, bool],
    dense: tuple[bool, bool],
    category: str,
) -> None:
    decision = classify_decision(
        {
            "cofitok": _method_result(
                semantic=cofitok[0],
                denoising=cofitok[1],
            ),
            "dense_identity": _method_result(
                semantic=dense[0],
                denoising=dense[1],
            ),
        }
    )

    assert decision["category"] == category
    flags = (
        decision["shared_exposure_semantic_recovery_supported"],
        decision["denoising_improves_but_semantic_alignment_does_not"],
        decision["method_asymmetry"],
        decision["no_exposure_recovery"],
    )
    assert sum(flags) == 1


def test_claim_boundary_is_permanently_non_authorizing() -> None:
    assert CLAIM_BOUNDARY["diagnostic_only"] is True
    assert CLAIM_BOUNDARY["generation_advantage_proven"] is False
    for field, value in CLAIM_BOUNDARY.items():
        if field.startswith("authorizes_"):
            assert value is False
