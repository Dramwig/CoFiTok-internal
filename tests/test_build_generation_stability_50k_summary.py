from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import build_generation_stability_50k_summary as summary


def _training() -> dict:
    return {
        "final_metrics": {"samples_seen": 3_200_000},
        "config": {
            "data": {"batch_size": 16},
            "optimization": {"gradient_accumulation_steps": 4},
        },
    }


def _decision() -> dict:
    return {
        "status": "pass",
        "decision": "authorize_fresh_matched_50k_preparation",
        "authorized_next_stage": "fresh_matched_50k_preparation",
        "robust_seeds": [2029, 2039],
        "min_robust_images": 64,
    }


def _config_validation() -> dict:
    return {
        "status": "pass",
        "training_recipe": {
            "schema": "cofitok_generation_training_recipe_v4",
            "stage": "stability_scaling",
            "valid": True,
        },
    }


def test_summary_binds_stability_authorization_without_authorizing_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = {
        "status": "pass",
        "relative_parameter_gap": 0.00015,
        "training_recipe": {"stage": "stability_scaling", "valid": True},
    }
    monkeypatch.setattr(summary, "validate_training_pair", lambda *args, **kwargs: pair)

    report = summary.build_summary(
        cofitok_training=_training(),
        dense_training=_training(),
        decision_validation=_decision(),
        config_validation=_config_validation(),
        expected_revision="a" * 40,
        expected_branch="scale/generation-stability",
    )

    assert report["status"] == "completed"
    assert report["stage"] == "stability_matched_50k"
    assert report["images_seen_per_method"] == 3_200_000
    assert report["training_pair"] == pair
    assert report["formal_300k_authorization_allowed"] is False
    assert report["formal_ema_sampling_gate_required"] is True


def test_summary_rejects_an_incomplete_compute_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        summary,
        "validate_training_pair",
        lambda *args, **kwargs: {"status": "pass"},
    )
    cofitok = _training()
    cofitok["final_metrics"]["samples_seen"] -= 64

    with pytest.raises(ValueError, match="exactly 3200000 images"):
        summary.build_summary(
            cofitok_training=cofitok,
            dense_training=_training(),
            decision_validation=_decision(),
            config_validation=_config_validation(),
            expected_revision="a" * 40,
            expected_branch="scale/generation-stability",
        )


def test_summary_rejects_a_weakened_or_wrong_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        summary,
        "validate_training_pair",
        lambda *args, **kwargs: {"status": "pass"},
    )
    config_validation = deepcopy(_config_validation())
    config_validation["training_recipe"]["stage"] = "scaling"

    with pytest.raises(ValueError, match="config validation did not pass"):
        summary.build_summary(
            cofitok_training=_training(),
            dense_training=_training(),
            decision_validation=_decision(),
            config_validation=config_validation,
            expected_revision="a" * 40,
            expected_branch="scale/generation-stability",
        )
