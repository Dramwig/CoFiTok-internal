from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from cofitok.generation.capacity_screen import ARM_NAMES
from cofitok.generation.capacity_screen_arm import (
    ARM_VALIDATION_ROLE,
    ARM_VALIDATION_SCHEMA,
)
from cofitok.generation.capacity_screen_result import (
    RESULT_BOUNDARY,
    build_capacity_screen_result,
    validate_capacity_screen_result_contract,
)


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/{name}.json",
        "bytes": 10,
        "sha256": sha256(name.encode("utf-8")).hexdigest(),
    }


def _arm(
    arm: str,
    *,
    fid: float,
    recall: float,
    precision: float = 0.70,
    top1: float = 0.005,
    top5: float = 0.02,
    predicted_class_fraction: float = 0.70,
    entropy: float = 0.75,
    endpoint: float = 0.03,
    validation: float = 0.03,
    reconstruction: float = 0.03,
    high_frequency: float = 0.20,
) -> dict[str, object]:
    method = "cofitok" if arm.endswith("cofitok") else "dense_identity"
    capacity = "base128" if arm.startswith("base128") else "base256"
    checkpoint: dict[str, object] = {
        "evaluated_images": 256,
        "timestep": 500,
        "order_count": 6 if method == "cofitok" else 1,
        "ordered_endpoint_clean_mse": endpoint,
    }
    if method == "cofitok":
        checkpoint.update(
            {
                "ordered_rank_by_path_auc": 1,
                "zero_token_max_abs": 0.0,
                "shuffled_to_ordered_endpoint_ratio": 10.0,
                "utilization": {
                    "coarse_token_energy_ratio": 0.50,
                    "tail_two_energy_ratio": 0.34,
                    "max_single_token_energy_ratio": 0.17,
                },
            }
        )
    launch_id = _identity("launch")
    return {
        "schema_version": ARM_VALIDATION_SCHEMA,
        "role": ARM_VALIDATION_ROLE,
        "status": "pass",
        "arm": arm,
        "capacity": capacity,
        "method": method,
        "execution_git": GIT,
        "sources": {"launch_receipt": launch_id},
        "training": {"validation_epsilon_mse": validation},
        "sampling": {"sample_set_sha256": (str(ARM_NAMES.index(arm) + 1) * 64)},
        "distribution": {
            "metrics": {
                "fid": fid,
                "precision": precision,
                "recall": recall,
                "inception_score_mean": 2.0,
                "inception_score_std": 0.1,
            },
            "real_set": {"sha256": "f" * 64},
        },
        "class_fidelity": {
            "metrics": {
                "top1_accuracy": top1,
                "top5_accuracy": top5,
                "predicted_class_fraction": predicted_class_fraction,
                "normalized_predicted_class_entropy": entropy,
            }
        },
        "checkpoint_evaluation": {"summary": checkpoint},
        "rollout": {
            "summary": {
                "final_reconstruction_x0_mse": reconstruction,
                "predicted_x0_high_frequency_ratio": {
                    "91": high_frequency,
                    "192": high_frequency,
                    "394": high_frequency,
                    "595": high_frequency,
                },
            }
        },
    }


def _inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    preparation_id = _identity("preparation")
    launch_id = _identity("launch")
    arms = {
        "base128_cofitok": _arm("base128_cofitok", fid=120.0, recall=0.01),
        "base128_dense_identity": _arm(
            "base128_dense_identity", fid=125.0, recall=0.012
        ),
        "base256_cofitok": _arm("base256_cofitok", fid=100.0, recall=0.03),
        "base256_dense_identity": _arm(
            "base256_dense_identity", fid=105.0, recall=0.032
        ),
    }
    preparation = {
        "scientific_status": "capacity_screen_prepared",
        "evaluation_contract": {"samples_per_arm": 1000},
    }
    launch = {
        "execution_checkout": GIT,
        "source_evidence": {"preparation": preparation_id},
    }
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_result.validate_capacity_screen_preparation_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_result.validate_capacity_screen_launch_receipt_contract",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_result.validate_capacity_screen_arm_validation",
        lambda value: value,
    )
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "launch_receipt": launch,
        "launch_receipt_identity": launch_id,
        "arm_validations": arms,
        "arm_validation_identities": {
            arm: _identity(f"{arm[0]}{index + 1}")
            for index, arm in enumerate(ARM_NAMES)
        },
        "result_git": GIT,
    }


def test_passing_screen_only_prepares_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_capacity_screen_result(**_inputs(monkeypatch))
    assert report["scientific_status"] == "screen_pass"
    assert report["failed_checks"] == []
    assert report["next_stage"]["capacity_confirmation_preparation_allowed"] is True
    assert report["next_stage"]["full_300k_launch_allowed"] is False
    assert report["authorization_boundary"] == RESULT_BOUNDARY
    assert validate_capacity_screen_result_contract(report) == report


def test_screen_holds_when_dense_does_not_improve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"]["base256_dense_identity"]["distribution"]["metrics"][
        "fid"
    ] = 126.0
    report = build_capacity_screen_result(**inputs)
    assert report["scientific_status"] == "hold"
    assert "dense_identity_fid_improves_with_capacity" in report["failed_checks"]
    assert report["next_stage"]["capacity_confirmation_preparation_allowed"] is False


def test_screen_holds_on_cofitok_mechanism_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"]["base256_cofitok"]["checkpoint_evaluation"]["summary"][
        "ordered_rank_by_path_auc"
    ] = 2
    report = build_capacity_screen_result(**inputs)
    assert report["scientific_status"] == "hold"
    assert "base256_ordered_rank" in report["failed_checks"]


def test_screen_holds_on_cofitok_specific_capacity_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"]["base256_cofitok"]["distribution"]["metrics"][
        "fid"
    ] = 119.0
    inputs["arm_validations"]["base256_dense_identity"]["distribution"]["metrics"][
        "fid"
    ] = 90.0
    report = build_capacity_screen_result(**inputs)
    assert report["scientific_status"] == "hold"
    assert "cofitok_relative_fid_capacity_interaction" in report["failed_checks"]


def test_rejects_arm_bound_to_another_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"]["base256_cofitok"]["sources"]["launch_receipt"] = _identity(
        "other"
    )
    with pytest.raises(ValueError, match="another launch receipt"):
        build_capacity_screen_result(**inputs)


def test_contract_rejects_direct_300k_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_capacity_screen_result(**_inputs(monkeypatch))
    altered = deepcopy(report)
    altered["authorization_boundary"] = dict(altered["authorization_boundary"])
    altered["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_capacity_screen_result_contract(altered)
