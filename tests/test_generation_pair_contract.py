from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract
from scripts.validate_generation_configs import validate_pair


ROOT = Path(__file__).resolve().parents[1]

RESIDUAL_ALIGNMENT_FIELDS = {
    "class_conditioning_residual_alignment_weight": 0.05,
    "class_conditioning_residual_alignment_start_step": 100,
    "class_conditioning_residual_alignment_warmup_steps": 200,
    "class_conditioning_residual_alignment_batch_fraction": 0.0625,
    "class_conditioning_residual_alignment_margin": 0.1,
    "class_conditioning_residual_alignment_temperature": 0.1,
    "class_conditioning_residual_alignment_wrong_label_offsets": [1, 500],
    "class_conditioning_residual_alignment_min_timestep": 500,
    "class_conditioning_residual_alignment_pooling_factors": [8, 16, 32],
    "class_conditioning_residual_alignment_reconstruction_weight": 0.25,
}


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
        (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_k8_probe5k.json",
            "imagenet256_10pct_stability_rollout_x0_u2_dense_probe5k.json",
        ),
        (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe1k.json",
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe1k.json",
        ),
        (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classresidualalign_k8_probe1k.json",
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classresidualalign_dense_probe1k.json",
        ),
        (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe5k.json",
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe5k.json",
        ),
        (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json",
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_50k.json",
        ),
        (
            "imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json",
            "imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json",
        ),
        (
            "imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json",
            "imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json",
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


@pytest.mark.parametrize(
    ("control_name", "candidate_name"),
    [
        (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe1k.json",
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classresidualalign_k8_probe1k.json",
        ),
        (
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe1k.json",
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classresidualalign_dense_probe1k.json",
        ),
    ],
)
def test_residual_alignment_probe_changes_only_name_and_objective(
    control_name: str,
    candidate_name: str,
) -> None:
    config_dir = ROOT / "configs/generation"
    control = json.loads((config_dir / control_name).read_text(encoding="utf-8"))
    candidate = json.loads((config_dir / candidate_name).read_text(encoding="utf-8"))

    assert candidate["name"] == Path(candidate_name).stem
    assert not any(key.startswith("class_conditioning_ranking_") for key in candidate["loss"])
    assert {
        key: candidate["loss"][key] for key in RESIDUAL_ALIGNMENT_FIELDS
    } == RESIDUAL_ALIGNMENT_FIELDS

    candidate["name"] = control["name"]
    for key in RESIDUAL_ALIGNMENT_FIELDS:
        del candidate["loss"][key]
    assert candidate == control


def test_residual_alignment_probe_pair_is_matched() -> None:
    cofitok = _read(
        "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classresidualalign_k8_probe1k.json"
    )
    dense = _read(
        "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classresidualalign_dense_probe1k.json"
    )

    report = generation_pair_contract(cofitok, dense)

    assert report["valid"] is True, report["issues"]
    assert report["mismatched_shared_training_loss_fields"] == []
    assert report["dense_nonzero_auxiliary_losses"] == []


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


def test_ema_teacher_consistency_is_a_matched_training_loss() -> None:
    cofitok = _read("imagenet256_10pct_fixed_basis_cofitok_k8_50k.json")
    dense = _read("imagenet256_10pct_fixed_basis_dense_50k.json")
    for config in (cofitok, dense):
        config["loss"]["ema_teacher_consistency_weight"] = 0.25
        config["loss"]["ema_teacher_consistency_start_step"] = 500
        config["loss"]["ema_teacher_consistency_warmup_steps"] = 250
        config["loss"]["ema_teacher_consistency_batch_fraction"] = 0.0625

    report = generation_pair_contract(cofitok, dense)

    assert report["valid"] is True, report["issues"]
    assert report["mismatched_shared_training_loss_fields"] == []
    assert (
        "ema_teacher_consistency_weight"
        not in report["dense_nonzero_auxiliary_losses"]
    )

    mismatched = copy.deepcopy(dense)
    mismatched["loss"]["ema_teacher_consistency_weight"] = 0.0
    report = generation_pair_contract(cofitok, mismatched)
    assert report["valid"] is False
    assert report["mismatched_shared_training_loss_fields"] == [
        "ema_teacher_consistency_weight"
    ]


def test_class_conditioning_ranking_is_a_matched_training_loss() -> None:
    cofitok = _read("imagenet256_10pct_fixed_basis_cofitok_k8_50k.json")
    dense = _read("imagenet256_10pct_fixed_basis_dense_50k.json")
    shared = {
        "class_conditioning_ranking_weight": 0.05,
        "class_conditioning_ranking_start_step": 500,
        "class_conditioning_ranking_warmup_steps": 500,
        "class_conditioning_ranking_batch_fraction": 0.0625,
        "class_conditioning_ranking_margin": 0.01,
        "class_conditioning_ranking_wrong_label_offset": 500,
        "class_conditioning_ranking_min_timestep": 500,
    }
    for config in (cofitok, dense):
        config["loss"].update(shared)

    report = generation_pair_contract(cofitok, dense)

    assert report["valid"] is True, report["issues"]
    assert report["mismatched_shared_training_loss_fields"] == []
    assert (
        "class_conditioning_ranking_weight"
        not in report["dense_nonzero_auxiliary_losses"]
    )

    mismatched = copy.deepcopy(dense)
    mismatched["loss"]["class_conditioning_ranking_margin"] = 0.02
    report = generation_pair_contract(cofitok, mismatched)
    assert report["valid"] is False
    assert report["mismatched_shared_training_loss_fields"] == [
        "class_conditioning_ranking_margin"
    ]


def test_residual_alignment_is_a_matched_training_loss() -> None:
    cofitok = _read("imagenet256_10pct_fixed_basis_cofitok_k8_50k.json")
    dense = _read("imagenet256_10pct_fixed_basis_dense_50k.json")
    shared = {
        "class_conditioning_residual_alignment_weight": 0.05,
        "class_conditioning_residual_alignment_start_step": 500,
        "class_conditioning_residual_alignment_warmup_steps": 500,
        "class_conditioning_residual_alignment_batch_fraction": 0.0625,
        "class_conditioning_residual_alignment_margin": 0.1,
        "class_conditioning_residual_alignment_temperature": 0.1,
        "class_conditioning_residual_alignment_wrong_label_offsets": [1, 500],
        "class_conditioning_residual_alignment_min_timestep": 500,
        "class_conditioning_residual_alignment_pooling_factors": [8, 16],
        "class_conditioning_residual_alignment_reconstruction_weight": 0.25,
    }
    for config in (cofitok, dense):
        config["loss"].update(shared)

    report = generation_pair_contract(cofitok, dense)

    assert report["valid"] is True, report["issues"]
    assert report["mismatched_shared_training_loss_fields"] == []
    assert (
        "class_conditioning_residual_alignment_weight"
        not in report["dense_nonzero_auxiliary_losses"]
    )

    mismatched = copy.deepcopy(dense)
    mismatched["loss"]["class_conditioning_residual_alignment_temperature"] = 0.2
    report = generation_pair_contract(cofitok, mismatched)
    assert report["valid"] is False
    assert report["mismatched_shared_training_loss_fields"] == [
        "class_conditioning_residual_alignment_temperature"
    ]


def test_ema_teacher_5k_pair_targets_the_late_drift_window() -> None:
    cofitok = _read(
        "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe5k.json"
    )
    dense = _read(
        "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe5k.json"
    )

    for config in (cofitok, dense):
        assert config["runtime"]["steps"] == 5000
        assert config["runtime"]["checkpoint_interval"] == 1250
        assert config["runtime"]["evaluation_interval"] == 1250
        assert config["loss"]["ema_teacher_consistency_weight"] == 0.25
        assert config["loss"]["ema_teacher_consistency_start_step"] == 3000
        assert config["loss"]["ema_teacher_consistency_warmup_steps"] == 1000
        assert config["loss"]["ema_teacher_consistency_batch_fraction"] == 0.0625

    report = generation_pair_contract(cofitok, dense)
    assert report["valid"] is True, report["issues"]


def test_ema_teacher_50k_pair_scales_the_stability_windows() -> None:
    cofitok = _read(
        "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json"
    )
    dense = _read(
        "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_50k.json"
    )

    for config in (cofitok, dense):
        assert config["runtime"]["steps"] == 50_000
        assert config["runtime"]["checkpoint_interval"] == 5_000
        assert config["loss"]["rollout_consistency_warmup_steps"] == 10_000
        assert config["loss"]["ema_teacher_consistency_weight"] == 0.25
        assert config["loss"]["ema_teacher_consistency_start_step"] == 30_000
        assert config["loss"]["ema_teacher_consistency_warmup_steps"] == 10_000
        assert config["loss"]["ema_teacher_consistency_batch_fraction"] == 0.0625

    report = generation_pair_contract(cofitok, dense)
    assert report["valid"] is True, report["issues"]


def test_ema_teacher_300k_pair_scales_the_stability_windows() -> None:
    cofitok = _read(
        "imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
    )
    dense = _read(
        "imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json"
    )

    for config in (cofitok, dense):
        assert config["model"]["base_channels"] == 256
        assert config["runtime"]["steps"] == 300_000
        assert config["runtime"]["protected_checkpoint_steps"] == [
            50_000,
            100_000,
            200_000,
            300_000,
        ]
        assert config["loss"]["rollout_consistency_warmup_steps"] == 60_000
        assert config["loss"]["ema_teacher_consistency_start_step"] == 180_000
        assert config["loss"]["ema_teacher_consistency_warmup_steps"] == 60_000

    report = generation_pair_contract(cofitok, dense)
    assert report["valid"] is True, report["issues"]


def test_ema_teacher_300k_pair_has_exact_large_capacity_parameter_counts() -> None:
    report = validate_pair(
        load_config(
            ROOT
            / "configs/generation/"
            "imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
        ),
        load_config(
            ROOT
            / "configs/generation/"
            "imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json"
        ),
        max_parameter_gap=0.02,
        stage="stability_full",
    )

    assert report["status"] == "pass", report["mismatches"]
    assert report["cofitok"]["parameter_count"] == 250_153_763
    assert report["dense"]["parameter_count"] == 250_135_043
    assert report["relative_parameter_gap"] == pytest.approx(
        0.0000748395737577641
    )
