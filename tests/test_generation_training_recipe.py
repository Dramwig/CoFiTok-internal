from __future__ import annotations

import copy
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
            "imagenet256_10pct_cofitok_k8_50k.json",
            "imagenet256_10pct_dense_50k.json",
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

    contract = generation_training_recipe_contract(cofitok, dense, stage="scaling")

    assert contract["valid"] is True, contract["issues"]


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
