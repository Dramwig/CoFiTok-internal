from __future__ import annotations

import copy
import json
from pathlib import Path

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract
from cofitok.generation_recipe import generation_training_recipe_contract


ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    return config_to_dict(load_config(ROOT / "configs/generation" / name))


def test_checked_in_scaling_and_full_recipes_pass() -> None:
    for stage, cofitok_name, dense_name in (
        (
            "scaling",
            "imagenet256_10pct_compressed_cofitok_k8_50k.json",
            "imagenet256_10pct_compressed_dense_50k.json",
        ),
        (
            "full",
            "imagenet256_cofitok_k8_300k.json",
            "imagenet256_dense_300k.json",
        ),
    ):
        contract = generation_training_recipe_contract(
            _config(cofitok_name),
            _config(dense_name),
            stage=stage,
        )

        assert contract["valid"] is True, contract["issues"]
        assert contract["issues"] == []
        assert contract["effective_batches"]["cofitok"]["effective_batch_size"] == 64
        assert contract["observed"]["cofitok"][
            "data.random_horizontal_flip_prob"
        ] == 0.5


def test_rank_recovery_probes_explicitly_disable_legacy_loss_defaults() -> None:
    expected_objectives = {
        "imagenet256_10pct_rankcomplete_denoise_path_k8_probe5k.json": {
            "denoise_path_prefix_weight": 0.05,
            "denoise_path_component_weight": 0.1,
        },
        "imagenet256_10pct_rankcomplete_epsilon_band_k8_probe5k.json": {
            "epsilon_band_prefix_weight": 0.05,
            "epsilon_band_component_weight": 0.1,
        },
        "imagenet256_10pct_rankcomplete_capacity_path_k8_probe5k.json": {
            "denoise_path_prefix_weight": 0.15,
            "denoise_path_component_weight": 0.3,
            "denoise_path_energy_weight": 0.5,
        },
        "imagenet256_10pct_rankcomplete_capacity_path_light_k8_probe5k.json": {
            "denoise_path_prefix_weight": 0.05,
            "denoise_path_component_weight": 0.1,
        },
        "imagenet256_10pct_rankcomplete_capacity_path_hellinger_k8_probe5k.json": {
            "denoise_path_prefix_weight": 0.05,
            "denoise_path_component_weight": 0.1,
            "denoise_path_energy_weight": 0.1,
        },
    }
    for name, expected in expected_objectives.items():
        raw = json.loads((ROOT / "configs/generation" / name).read_text(encoding="utf-8"))
        loss = raw["loss"]
        assert {key: loss[key] for key in ("prefix_weight", "monotonic_weight", "zero_token_weight")} == {
            "prefix_weight": 0.0,
            "monotonic_weight": 0.0,
            "zero_token_weight": 0.0,
        }
        resolved = _config(name)["loss"]
        nonzero_weights = {
            key: value
            for key, value in resolved.items()
            if key.endswith("_weight") and float(value) != 0.0
        }
        assert nonzero_weights == {"epsilon_weight": 1.0, **expected}
        if "capacity_path" in name:
            assert loss["denoise_path_progress_mode"] == "token_capacity"
        if "hellinger" in name:
            assert loss["denoise_path_energy_mode"] == "hellinger"


def test_hellinger_probe_only_changes_the_light_probe_energy_objective() -> None:
    light = json.loads(
        (
            ROOT
            / "configs/generation/imagenet256_10pct_rankcomplete_capacity_path_light_k8_probe5k.json"
        ).read_text(encoding="utf-8")
    )
    hellinger = json.loads(
        (
            ROOT
            / "configs/generation/imagenet256_10pct_rankcomplete_capacity_path_hellinger_k8_probe5k.json"
        ).read_text(encoding="utf-8")
    )
    expected = copy.deepcopy(light)
    expected["name"] = "imagenet256_10pct_rankcomplete_capacity_path_hellinger_k8_probe5k"
    expected["loss"]["denoise_path_energy_weight"] = 0.1
    expected["loss"]["denoise_path_energy_mode"] = "hellinger"

    assert hellinger == expected


def test_recipe_allows_selected_runtime_with_same_effective_batch() -> None:
    cofitok = _config("imagenet256_cofitok_k8_300k.json")
    dense = _config("imagenet256_dense_300k.json")
    for config in (cofitok, dense):
        config["data"]["batch_size"] = 32
        config["optimization"]["gradient_accumulation_steps"] = 2

    contract = generation_training_recipe_contract(cofitok, dense, stage="full")

    assert contract["valid"] is True, contract["issues"]


def test_recipe_rejects_unbenchmarked_runtime_with_same_effective_batch() -> None:
    cofitok = _config("imagenet256_cofitok_k8_300k.json")
    dense = _config("imagenet256_dense_300k.json")
    for config in (cofitok, dense):
        config["data"]["batch_size"] = 1
        config["optimization"]["gradient_accumulation_steps"] = 64

    contract = generation_training_recipe_contract(cofitok, dense, stage="full")

    assert contract["valid"] is False
    assert all("runtime_batch" in issue for issue in contract["issues"])


def test_scaling_recipe_accepts_pinned_legacy_implicit_defaults() -> None:
    cofitok = _config("imagenet256_10pct_cofitok_k8_50k.json")
    dense = _config("imagenet256_10pct_dense_50k.json")
    for config in (cofitok, dense):
        config["data"].pop("random_horizontal_flip_prob")
        config["runtime"].pop("protected_checkpoint_steps")
        config["model"].pop("token_channel_schedule")
        config["model"].pop("token_spatial_strides")

    contract = generation_training_recipe_contract(cofitok, dense, stage="legacy_scaling")

    assert contract["valid"] is True, contract["issues"]


def test_compressed_recipe_rejects_a_dense_sized_token() -> None:
    cofitok = _config("imagenet256_10pct_compressed_cofitok_k8_50k.json")
    dense = _config("imagenet256_10pct_compressed_dense_50k.json")
    cofitok["model"]["token_channel_schedule"][-1] = 3

    contract = generation_training_recipe_contract(cofitok, dense, stage="scaling")

    assert contract["valid"] is False
    assert any("not smaller than the dense field" in issue for issue in contract["issues"])


def test_compressed_recipe_rejects_full_resolution_rank_deficit() -> None:
    cofitok = _config("imagenet256_10pct_compressed_cofitok_k8_50k.json")
    dense = _config("imagenet256_10pct_compressed_dense_50k.json")
    cofitok["model"]["token_channel_schedule"][-2:] = [4, 2]
    cofitok["model"]["token_spatial_strides"][-2:] = [2, 1]

    contract = generation_training_recipe_contract(cofitok, dense, stage="scaling")

    assert contract["valid"] is False
    assert any("full-resolution token channels" in issue for issue in contract["issues"])


def test_recipe_rejects_identically_weakened_matched_pair() -> None:
    cofitok = _config("imagenet256_cofitok_k8_300k.json")
    dense = _config("imagenet256_dense_300k.json")
    for config in (cofitok, dense):
        config["model"]["class_dropout_prob"] = 0.0
        config["optimization"]["ema_decay"] = 0.9

    assert generation_pair_contract(cofitok, dense)["valid"] is True
    contract = generation_training_recipe_contract(cofitok, dense, stage="full")

    assert contract["valid"] is False
    assert any("class_dropout_prob" in issue for issue in contract["issues"])
    assert any("ema_decay" in issue for issue in contract["issues"])


def test_recipe_rejects_changed_factorization_objective() -> None:
    cofitok = _config("imagenet256_cofitok_k8_300k.json")
    dense = _config("imagenet256_dense_300k.json")
    cofitok = copy.deepcopy(cofitok)
    cofitok["loss"]["denoise_path_component_weight"] = 0.0

    contract = generation_training_recipe_contract(cofitok, dense, stage="full")

    assert contract["valid"] is False
    assert any("denoise_path_component_weight" in issue for issue in contract["issues"])
