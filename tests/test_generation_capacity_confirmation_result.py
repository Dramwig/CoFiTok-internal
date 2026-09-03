from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from cofitok.generation.capacity_confirmation_result import (
    RESULT_BOUNDARY,
    build_capacity_confirmation_result,
    validate_capacity_confirmation_result_contract,
)
from cofitok.generation.capacity_screen import ARM_NAMES


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/{name}.json",
        "bytes": len(name) + 10,
        "sha256": sha256(name.encode("utf-8")).hexdigest(),
    }


def _arm(
    arm: str,
    *,
    fid: float,
    recall: float,
    precision: float,
    top1: float,
    top5: float,
    class_fraction: float,
    entropy: float,
) -> dict[str, object]:
    method = "cofitok" if arm.endswith("cofitok") else "dense_identity"
    capacity = "base128" if arm.startswith("base128") else "base256"
    checkpoint: dict[str, object] = {
        "evaluated_images": 256,
        "timestep": 500,
        "order_count": 6 if method == "cofitok" else 1,
        "ordered_endpoint_clean_mse": 0.03,
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
    return {
        "arm": arm,
        "capacity": capacity,
        "method": method,
        "execution_git": GIT,
        "sources": {"launch_receipt": _identity("launch")},
        "frozen_training": {
            "training_performed": False,
            "checkpoint": {"sha256": sha256(arm.encode()).hexdigest()},
            "screen_training": {"validation_epsilon_mse": 0.03},
        },
        "sampling": {
            "sample_set_sha256": sha256(f"samples-{arm}".encode()).hexdigest(),
            "checkpoint_sha256": sha256(arm.encode()).hexdigest(),
        },
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
                "predicted_class_fraction": class_fraction,
                "normalized_predicted_class_entropy": entropy,
            },
            "classifier": {"name": "torchvision_resnet50_imagenet1k_v2"},
        },
        "screen_checkpoint_evaluation": {"summary": checkpoint},
        "screen_rollout": {
            "summary": {
                "final_reconstruction_x0_mse": 0.03,
                "predicted_x0_high_frequency_ratio": {
                    "91": 0.2,
                    "192": 0.2,
                    "394": 0.2,
                    "595": 0.2,
                },
            }
        },
    }


def _inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    preparation_id = _identity("preparation")
    launch_id = _identity("launch")
    arms = {
        "base128_cofitok": _arm(
            "base128_cofitok",
            fid=120.0,
            recall=0.05,
            precision=0.55,
            top1=0.005,
            top5=0.03,
            class_fraction=0.60,
            entropy=0.70,
        ),
        "base128_dense_identity": _arm(
            "base128_dense_identity",
            fid=125.0,
            recall=0.05,
            precision=0.55,
            top1=0.005,
            top5=0.03,
            class_fraction=0.60,
            entropy=0.70,
        ),
        "base256_cofitok": _arm(
            "base256_cofitok",
            fid=90.0,
            recall=0.20,
            precision=0.60,
            top1=0.030,
            top5=0.10,
            class_fraction=0.70,
            entropy=0.80,
        ),
        "base256_dense_identity": _arm(
            "base256_dense_identity",
            fid=95.0,
            recall=0.21,
            precision=0.61,
            top1=0.035,
            top5=0.11,
            class_fraction=0.71,
            entropy=0.81,
        ),
    }
    preparation = {
        "scientific_status": "capacity_confirmation_prepared",
        "evaluation_contract": {"samples_per_arm": 10_000},
    }
    launch = {
        "execution_checkout": GIT,
        "source_evidence": {"preparation": preparation_id},
    }
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation_result.validate_capacity_confirmation_preparation_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation_result.validate_capacity_confirmation_launch_receipt_contract",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation_result.validate_capacity_confirmation_arm_validation",
        lambda value: value,
    )
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "launch_receipt": launch,
        "launch_receipt_identity": launch_id,
        "arm_validations": arms,
        "arm_validation_identities": {
            arm: _identity(f"confirmation-{index}-{arm}")
            for index, arm in enumerate(ARM_NAMES)
        },
        "result_git": GIT,
    }


def test_passing_confirmation_only_prepares_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_capacity_confirmation_result(**_inputs(monkeypatch))
    assert report["scientific_status"] == "confirmation_pass"
    assert report["failed_checks"] == []
    assert report["next_stage"]["support_collapse_resolved"] is True
    assert report["next_stage"][
        "large_capacity_readiness_preparation_allowed"
    ] is True
    assert report["next_stage"]["full_300k_launch_allowed"] is False
    assert report["authorization_boundary"] == RESULT_BOUNDARY
    assert validate_capacity_confirmation_result_contract(report) == report


def test_confirmation_holds_when_base256_recall_stays_collapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"]["base256_cofitok"]["distribution"]["metrics"][
        "recall"
    ] = 0.02
    report = build_capacity_confirmation_result(**inputs)
    assert report["scientific_status"] == "hold"
    assert "base256_cofitok_recall_absolute" in report["failed_checks"]
    assert report["next_stage"][
        "large_capacity_readiness_preparation_allowed"
    ] is False


def test_confirmation_holds_when_class_conditioning_stays_collapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"]["base256_dense_identity"]["class_fidelity"][
        "metrics"
    ]["top1_accuracy"] = 0.001
    report = build_capacity_confirmation_result(**inputs)
    assert "base256_dense_identity_top1_absolute" in report["failed_checks"]


def test_confirmation_holds_on_mechanism_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"]["base256_cofitok"][
        "screen_checkpoint_evaluation"
    ]["summary"]["ordered_rank_by_path_auc"] = 2
    report = build_capacity_confirmation_result(**inputs)
    assert "base256_ordered_rank" in report["failed_checks"]


def test_confirmation_rejects_another_launch_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"]["base256_cofitok"]["sources"][
        "launch_receipt"
    ] = _identity("other")
    with pytest.raises(ValueError, match="another launch receipt"):
        build_capacity_confirmation_result(**inputs)


def test_contract_rejects_direct_300k_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_capacity_confirmation_result(**_inputs(monkeypatch))
    altered = deepcopy(report)
    altered["authorization_boundary"] = dict(altered["authorization_boundary"])
    altered["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_capacity_confirmation_result_contract(altered)
