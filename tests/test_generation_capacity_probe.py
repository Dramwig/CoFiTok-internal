from __future__ import annotations

import json
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_recipe import (
    generation_training_recipe_contract,
    infer_generation_training_stage,
)
from scripts.validate_generation_configs import validate_pair


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "generation"
COFITOK_BRIDGE = (
    "imagenet256_stability_quality_bridge_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
)
DENSE_BRIDGE = (
    "imagenet256_stability_quality_bridge_"
    "rollout_x0_u2_ema_teacher_dense_100k.json"
)
COFITOK_CAPACITY = (
    "imagenet256_stability_capacity_probe_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
)
DENSE_CAPACITY = (
    "imagenet256_stability_capacity_probe_"
    "rollout_x0_u2_ema_teacher_dense_100k.json"
)


def _raw(name: str) -> dict:
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def _resolved(name: str) -> dict:
    return config_to_dict(load_config(CONFIG_ROOT / name))


def _differences(left: object, right: object, path: str = "") -> set[str]:
    if type(left) is not type(right):
        return {path}
    if isinstance(left, dict):
        differences: set[str] = set()
        for key in set(left) | set(right):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.add(child)
            else:
                differences.update(_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        return set() if left == right else {path}
    return set() if left == right else {path}


def test_capacity_probe_is_an_exact_base_channels_causal_intervention() -> None:
    allowed = {
        "name",
        "model.base_channels",
        "runtime.protected_checkpoint_steps",
    }
    for bridge_name, capacity_name in (
        (COFITOK_BRIDGE, COFITOK_CAPACITY),
        (DENSE_BRIDGE, DENSE_CAPACITY),
    ):
        bridge = _raw(bridge_name)
        capacity = _raw(capacity_name)

        assert _differences(bridge, capacity) == allowed
        assert bridge["model"]["base_channels"] == 128
        assert capacity["model"]["base_channels"] == 256
        assert bridge["runtime"]["steps"] == capacity["runtime"]["steps"] == 100_000
        assert capacity["runtime"]["protected_checkpoint_steps"] == [10_000]


def test_capacity_probe_pair_passes_recipe_and_parameter_contracts() -> None:
    cofitok = _resolved(COFITOK_CAPACITY)
    dense = _resolved(DENSE_CAPACITY)

    assert infer_generation_training_stage(cofitok, dense) == "stability_capacity_probe"
    recipe = generation_training_recipe_contract(
        cofitok,
        dense,
        stage="stability_capacity_probe",
    )
    assert recipe["valid"] is True, recipe["issues"]
    assert recipe["issues"] == []
    assert recipe["expected_shared"]["model.base_channels"] == 256
    assert recipe["expected_shared"]["runtime.steps"] == 100_000
    assert recipe["expected_shared"]["runtime.protected_checkpoint_steps"] == [
        10_000
    ]
    assert recipe["effective_batches"]["cofitok"]["effective_batch_size"] == 64

    report = validate_pair(
        load_config(CONFIG_ROOT / COFITOK_CAPACITY),
        load_config(CONFIG_ROOT / DENSE_CAPACITY),
        max_parameter_gap=0.02,
        stage="stability_capacity_probe",
    )
    assert report["status"] == "pass", report["mismatches"]
    assert report["cofitok"]["parameter_count"] == 250_153_763
    assert report["dense"]["parameter_count"] == 250_135_043
    assert report["relative_parameter_gap"] == pytest.approx(
        0.0000748395737577641
    )


def test_capacity_probe_rejects_any_schedule_or_objective_drift() -> None:
    cofitok = _resolved(COFITOK_CAPACITY)
    dense = _resolved(DENSE_CAPACITY)
    cofitok["loss"]["rollout_consistency_warmup_steps"] += 1

    report = generation_training_recipe_contract(
        cofitok,
        dense,
        stage="stability_capacity_probe",
    )

    assert report["valid"] is False
    assert any(
        "rollout_consistency_warmup_steps" in issue for issue in report["issues"]
    )
