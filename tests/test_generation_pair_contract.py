from __future__ import annotations

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
