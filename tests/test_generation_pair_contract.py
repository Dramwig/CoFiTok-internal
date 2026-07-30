from __future__ import annotations

import copy
from pathlib import Path

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return config_to_dict(load_config(ROOT / "configs/generation" / name))


def test_checked_in_generation_pairs_isolate_factorization_differences() -> None:
    for cofitok_name, dense_name in (
        (
            "imagenet256_10pct_fixed_basis_cofitok_k8_50k.json",
            "imagenet256_10pct_fixed_basis_dense_50k.json",
        ),
        (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_k8_probe1k.json",
            "imagenet256_10pct_stability_rollout_x0_u2_dense_probe1k.json",
        ),
        ("imagenet256_cofitok_k8_300k.json", "imagenet256_dense_300k.json"),
    ):
        report = generation_pair_contract(_read(cofitok_name), _read(dense_name))
        assert report["valid"] is True, report["issues"]
        assert report["mismatched_config_sections"] == []
        assert report["mismatched_shared_model_fields"] == []
        assert report["identities"] == {
            "cofitok_token_count": 8,
            "dense_token_count": 1,
            "cofitok_feedback": True,
            "dense_feedback": False,
            "cofitok_synthesis": "fixed_basis",
            "dense_synthesis": "dense_identity",
        }


def test_rollout_consistency_is_a_matched_training_loss() -> None:
    cofitok = _read("imagenet256_10pct_fixed_basis_cofitok_k8_50k.json")
    dense = _read("imagenet256_10pct_fixed_basis_dense_50k.json")
    for config in (cofitok, dense):
        config["loss"]["rollout_consistency_weight"] = 0.25
        config["loss"]["rollout_consistency_start_step"] = 100
        config["loss"]["rollout_consistency_warmup_steps"] = 100
        config["loss"]["rollout_consistency_unroll_steps"] = 2

    report = generation_pair_contract(cofitok, dense)

    assert report["valid"] is True, report["issues"]
    assert report["mismatched_shared_training_loss_fields"] == []
    assert "rollout_consistency_weight" not in report["dense_nonzero_auxiliary_losses"]

    mismatched = copy.deepcopy(dense)
    mismatched["loss"]["rollout_consistency_weight"] = 0.0
    report = generation_pair_contract(cofitok, mismatched)
    assert report["valid"] is False
    assert report["mismatched_shared_training_loss_fields"] == [
        "rollout_consistency_weight"
    ]

    mismatched = copy.deepcopy(dense)
    mismatched["loss"]["rollout_consistency_unroll_steps"] = 1
    report = generation_pair_contract(cofitok, mismatched)
    assert report["valid"] is False
    assert report["mismatched_shared_training_loss_fields"] == [
        "rollout_consistency_unroll_steps"
    ]
