from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import cofitok.generation.capacity_screen as capacity_screen
from cofitok.generation.capacity_screen import (
    ARM_NAMES,
    ARM_SPECS,
    CAPACITY_NAMES,
    PREPARATION_BOUNDARY,
    build_capacity_screen_preparation,
    validate_capacity_screen_preparation_contract,
)
from cofitok.generation.exposure_capacity_decision import (
    DECISION_ROLE,
    DECISION_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    "base128_cofitok": ROOT
    / "configs/generation/imagenet256_capacity_reference_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json",
    "base128_dense_identity": ROOT
    / "configs/generation/imagenet256_capacity_reference_"
    "rollout_x0_u2_ema_teacher_dense_100k.json",
    "base256_cofitok": ROOT
    / "configs/generation/imagenet256_capacity_qualification_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json",
    "base256_dense_identity": ROOT
    / "configs/generation/imagenet256_capacity_qualification_"
    "rollout_x0_u2_ema_teacher_dense_100k.json",
}
GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": len(name) + 1,
        "sha256": (name.encode().hex() + "0" * 64)[:64],
    }


def _decision() -> dict[str, object]:
    return {
        "schema_version": DECISION_SCHEMA,
        "role": DECISION_ROLE,
        "status": "completed",
        "decision": "capacity_screen",
        "scientific_status": "capacity_screen",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "decision_git": copy.deepcopy(GIT),
        "next_stage": {
            "capacity_screen_preparation_allowed": True,
        },
    }


def _pair_validation(capacity: str) -> dict[str, object]:
    return {
        "status": "pass",
        "mismatches": [],
        "relative_parameter_gap": 0.0001,
        "cofitok": {
            "parameter_count": ARM_SPECS[f"{capacity}_cofitok"][
                "parameter_count"
            ]
        },
        "dense": {
            "parameter_count": ARM_SPECS[f"{capacity}_dense_identity"][
                "parameter_count"
            ]
        },
        "training_recipe": {
            "valid": True,
            "stage": ARM_SPECS[f"{capacity}_cofitok"]["recipe_stage"],
            "schema": "cofitok_generation_training_recipe_v4",
            "effective_batches": {
                "cofitok": {
                    "micro_batch_size": 16,
                    "gradient_accumulation_steps": 4,
                    "effective_batch_size": 64,
                },
                "dense_identity": {
                    "micro_batch_size": 16,
                    "gradient_accumulation_steps": 4,
                    "effective_batch_size": 64,
                },
            },
        },
    }


def _inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(
        capacity_screen,
        "validate_decision_contract",
        lambda value: copy.deepcopy(dict(value)),
    )
    decision = _decision()
    decision_identity = _identity("decision")
    configs = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in CONFIG_PATHS.items()
    }
    return {
        "decision": decision,
        "decision_identity": decision_identity,
        "decision_validation": {
            "schema_version": capacity_screen.DECISION_VALIDATION_SCHEMA,
            "role": capacity_screen.DECISION_VALIDATION_ROLE,
            "status": "pass",
            "scientific_route": "capacity_screen",
            "decision": decision_identity,
            "generation_advantage_proven": False,
        },
        "decision_validation_identity": _identity("decision_validation"),
        "configs": configs,
        "config_identities": {arm: _identity(f"config_{arm}") for arm in ARM_NAMES},
        "pair_validations": {
            capacity: _pair_validation(capacity) for capacity in CAPACITY_NAMES
        },
        "pair_validation_identities": {
            capacity: _identity(f"pair_{capacity}") for capacity in CAPACITY_NAMES
        },
        "preparation_git": copy.deepcopy(GIT),
        "output_root": "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "capacity_qualification_v1",
    }


def test_builds_fresh_four_arm_capacity_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    report = build_capacity_screen_preparation(**_inputs(monkeypatch))

    assert report["selection"]["fresh_training_arms"] == list(ARM_NAMES)
    assert report["selection"]["images_seen_per_arm"] == 640_000
    assert report["evaluation_contract"]["sample_steps"] == 100
    assert report["evaluation_contract"]["samples_per_arm"] == 1_000
    assert report["evaluation_contract"]["class_fidelity_required"] is True
    assert report["decision_contract"]["confirmation_samples_per_arm"] == 10_000
    assert report["authorization_boundary"] == PREPARATION_BOUNDARY
    assert validate_capacity_screen_preparation_contract(report) == report


def test_capacity_intervention_is_only_base_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_capacity_screen_preparation(**_inputs(monkeypatch))

    assert report["within_method_capacity_differences"] == {
        "cofitok": ["model.base_channels", "name"],
        "dense_identity": ["model.base_channels", "name"],
    }


def test_rejects_a_nonfresh_or_nonmatched_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["configs"]["base128_cofitok"]["runtime"][
        "protected_checkpoint_steps"
    ] = [50_000, 100_000]

    with pytest.raises(ValueError, match="training selection differs"):
        build_capacity_screen_preparation(**inputs)


def test_rejects_an_extra_capacity_intervention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["configs"]["base256_cofitok"]["optimization"]["weight_decay"] = 0.02

    with pytest.raises(ValueError, match="capacity intervention differs"):
        build_capacity_screen_preparation(**inputs)


def test_rejects_a_decision_validation_for_another_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["decision_validation"]["decision"] = _identity("other")

    with pytest.raises(ValueError, match="decision validation differs"):
        build_capacity_screen_preparation(**inputs)


def test_contract_rejects_direct_300k_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_capacity_screen_preparation(**_inputs(monkeypatch))
    report["authorization_boundary"]["full_300k_launch_allowed"] = True

    with pytest.raises(ValueError, match="preparation contract differs"):
        validate_capacity_screen_preparation_contract(report)
