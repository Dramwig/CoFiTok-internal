from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import cofitok.generation.terminal_snr_screen as screen
from cofitok.generation.terminal_snr_reassessment import SCREEN_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    "control_cofitok": ROOT
    / "configs/generation/imagenet256_capacity_reference_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json",
    "control_dense_identity": ROOT
    / "configs/generation/imagenet256_capacity_reference_"
    "rollout_x0_u2_ema_teacher_dense_100k.json",
    "endpoint0975_cofitok": ROOT
    / "configs/generation/imagenet256_terminal_snr_endpoint0975_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json",
    "endpoint0975_dense_identity": ROOT
    / "configs/generation/imagenet256_terminal_snr_endpoint0975_"
    "rollout_x0_u2_ema_teacher_dense_100k.json",
}
GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/generation-terminal-snr-screen",
    "tracked_dirty": False,
}


def _identity(name: str, character: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": 100 + len(name),
        "sha256": character * 64,
    }


def _pair_validation(condition: str) -> dict[str, object]:
    endpoint = screen.CONDITION_ENDPOINTS[condition]
    return {
        "status": "pass",
        "mismatches": [],
        "relative_parameter_gap": 0.00015,
        "cofitok": {
            "parameter_count": screen.ARM_SPECS[f"{condition}_cofitok"][
                "parameter_count"
            ]
        },
        "dense": {
            "parameter_count": screen.ARM_SPECS[f"{condition}_dense_identity"][
                "parameter_count"
            ]
        },
        "matched_diffusion": {
            "num_train_timesteps": 1000,
            "schedule_type": "cosine",
            "prediction_target": "epsilon",
            "cosine_endpoint_fraction": endpoint,
        },
        "training_recipe": {
            "valid": True,
            "stage": "stability_capacity_reference",
            "schema": "cofitok_generation_training_recipe_v4",
            "effective_batches": {
                method: {
                    "micro_batch_size": 16,
                    "gradient_accumulation_steps": 4,
                    "effective_batch_size": 64,
                }
                for method in screen.METHOD_NAMES
            },
        },
    }


def _inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    config_ids = {
        arm: _identity(f"config-{arm}", str(index + 1))
        for index, arm in enumerate(screen.ARM_NAMES)
    }
    source_configs = {
        "control_configs": {
            method: copy.deepcopy(config_ids[f"control_{method}"])
            for method in screen.METHOD_NAMES
        },
        "intervention_configs": {
            method: copy.deepcopy(config_ids[f"endpoint0975_{method}"])
            for method in screen.METHOD_NAMES
        },
    }
    reassessment_id = _identity("reassessment", "8")
    reassessment = {
        "status": "completed",
        "scientific_status": "bounded_screen_selected_not_executed",
        "decision_git": copy.deepcopy(GIT),
        "selected_intervention": {
            "id": "matched_cosine_endpoint_fraction_0p975",
            "single_scientific_config_field": (
                "diffusion.cosine_endpoint_fraction"
            ),
        },
        "bounded_screen_contract": {
            "training_steps_per_arm": 10_000,
            "evaluation_samples_per_arm": 1_000,
            "sample_steps": 100,
            "sampler": "ddim",
            "weights": "ema",
            "guidance_scale": 1.5,
            "precision": "bf16",
            "seed": 2027,
            "thresholds": copy.deepcopy(SCREEN_THRESHOLDS),
        },
        "next_stage": {
            "route": (
                "build_separate_source_bound_terminal_snr_screen_preparation"
            ),
            "source_bound_preparation_may_be_built": True,
            "execution_ready": False,
        },
        "source_evidence": {"config_contract": source_configs},
    }
    validation = {"status": "pass", "decision": reassessment_id}
    monkeypatch.setattr(
        screen,
        "validate_terminal_snr_reassessment",
        lambda value: copy.deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        screen,
        "validate_terminal_snr_reassessment_validation",
        lambda value, **kwargs: copy.deepcopy(dict(value)),
    )
    return {
        "reassessment": reassessment,
        "reassessment_identity": reassessment_id,
        "reassessment_validation": validation,
        "reassessment_validation_identity": _identity("validation", "9"),
        "configs": {
            arm: json.loads(path.read_text(encoding="utf-8"))
            for arm, path in CONFIG_PATHS.items()
        },
        "config_identities": config_ids,
        "pair_validations": {
            condition: _pair_validation(condition)
            for condition in screen.CONDITION_NAMES
        },
        "pair_validation_identities": {
            condition: _identity(f"pair-{condition}", "c")
            for condition in screen.CONDITION_NAMES
        },
        "preparation_git": copy.deepcopy(GIT),
        "output_root": (
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "terminal_snr_endpoint_screen_v1"
        ),
    }


def test_builds_non_authorizing_matched_endpoint_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = screen.build_terminal_snr_screen_preparation(**_inputs(monkeypatch))

    assert report["selection"]["fresh_training_arms"] == list(screen.ARM_NAMES)
    assert report["selection"]["images_seen_per_arm"] == 640_000
    assert report["selection"]["conditions"] == {
        "control": {"cosine_endpoint_fraction": 1.0},
        "endpoint0975": {"cosine_endpoint_fraction": 0.975},
    }
    assert report["evaluation_contract"]["seed"] == 2027
    assert report["thresholds"] == SCREEN_THRESHOLDS
    assert report["authorization_boundary"] == screen.PREPARATION_BOUNDARY
    assert screen.validate_terminal_snr_screen_preparation_contract(report) == report


def test_endpoint_is_the_only_scientific_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = screen.build_terminal_snr_screen_preparation(**_inputs(monkeypatch))

    assert report["within_method_condition_differences"] == {
        method: ["diffusion.cosine_endpoint_fraction", "name"]
        for method in screen.METHOD_NAMES
    }


def test_rejects_config_not_bound_by_reassessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["config_identities"]["endpoint0975_cofitok"]["sha256"] = "f" * 64

    with pytest.raises(ValueError, match="differs from the reassessment"):
        screen.build_terminal_snr_screen_preparation(**inputs)


def test_rejects_second_scientific_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["configs"]["endpoint0975_dense_identity"]["optimization"][
        "weight_decay"
    ] = 0.02

    with pytest.raises(ValueError, match="endpoint intervention differs"):
        screen.build_terminal_snr_screen_preparation(**inputs)


def test_rejects_weakened_reassessment_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["reassessment"]["bounded_screen_contract"]["thresholds"][
        "both_methods_min_relative_fid_improvement"
    ] = 0.0

    with pytest.raises(ValueError, match="bounded-screen contract differs"):
        screen.build_terminal_snr_screen_preparation(**inputs)


def test_contract_rejects_execution_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = screen.build_terminal_snr_screen_preparation(**_inputs(monkeypatch))
    report["authorization_boundary"]["training_launch_allowed"] = True

    with pytest.raises(ValueError, match="preparation contract differs"):
        screen.validate_terminal_snr_screen_preparation_contract(report)
