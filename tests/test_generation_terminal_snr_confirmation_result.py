from __future__ import annotations

import copy
from hashlib import sha256

import pytest

from cofitok.generation.terminal_snr_confirmation import EVALUATION_CONTRACT
from cofitok.generation.terminal_snr_confirmation_result import (
    CONFIRMATION_THRESHOLDS,
    RESULT_BOUNDARY,
    build_terminal_snr_confirmation_result,
    validate_terminal_snr_confirmation_result_contract,
)
from cofitok.generation.terminal_snr_screen import ARM_NAMES, ARM_SPECS


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/terminal-snr-confirmation",
    "tracked_dirty": False,
}
SCREEN_GIT = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "analysis/terminal-snr-screen",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": len(name) + 1,
        "sha256": sha256(name.encode()).hexdigest(),
    }


def _sources(*, low_support: bool = False, evaluator_mismatch: bool = False):
    preparation_id = _identity("preparation")
    launch_id = _identity("launch")
    preparation = {
        "scientific_status": "terminal_snr_confirmation_prepared",
        "evaluation_contract": copy.deepcopy(EVALUATION_CONTRACT),
        "screen_execution_checkout": SCREEN_GIT,
    }
    launch = {
        "execution_checkout": GIT,
        "screen_execution_checkout": SCREEN_GIT,
        "evaluation_contract": copy.deepcopy(preparation["evaluation_contract"]),
        "runtime_environment_sha256": "1" * 64,
        "source_evidence": {"preparation": preparation_id},
    }
    arms = {}
    arm_ids = {}
    for index, arm in enumerate(ARM_NAMES):
        spec = ARM_SPECS[arm]
        checkpoint_sha = sha256(f"checkpoint-{arm}".encode()).hexdigest()
        endpoint = spec["condition"] == "endpoint0975"
        fid = 70.0 if endpoint else 80.0
        recall = 0.16 if endpoint else 0.15
        if low_support and spec["method"] == "cofitok":
            recall = 0.05
        diagnostic = {
            "ordered_rank_by_path_auc": 1,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 3.0,
            "utilization": {
                "coarse_token_energy_ratio": 0.2,
                "tail_two_energy_ratio": 0.4,
            },
        }
        evaluator_sha = "2" * 64
        if evaluator_mismatch and arm == ARM_NAMES[-1]:
            evaluator_sha = "9" * 64
        arms[arm] = {
            "condition": spec["condition"],
            "method": spec["method"],
            "endpoint_fraction": spec["endpoint_fraction"],
            "execution_git": GIT,
            "screen_execution_git": SCREEN_GIT,
            "sources": {"launch_receipt": launch_id},
            "frozen_training": {
                "checkpoint": {
                    "path": f"/tmp/{arm}.pt",
                    "bytes": 100 + index,
                    "sha256": checkpoint_sha,
                    "integrity_manifest": _identity(f"sidecar-{arm}"),
                },
                "screen_training": {
                    "validation_epsilon_mse": 0.2,
                    "validation": {"runtime_environment_sha256": "1" * 64},
                },
            },
            "sampling": {
                "sample_count": 10_000,
                "sample_set_sha256": sha256(f"confirmation-{arm}".encode()).hexdigest(),
                "runtime_environment_sha256": "1" * 64,
            },
            "sampling_preflight": {
                "report": _identity(f"preflight-{arm}"),
                "status": "passed",
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_step": 10_000,
                "runtime_environment_sha256": "1" * 64,
                "execution_git": {
                    "revision": GIT["revision"],
                    "branch": GIT["branch"],
                    "tracked_dirty": False,
                },
                "checkpoint_screen_git": {
                    "revision": SCREEN_GIT["revision"],
                    "branch": SCREEN_GIT["branch"],
                    "tracked_dirty": False,
                },
                "source_git": None,
                "request": {"batch_size": 4},
                "output_finite": True,
                "elapsed_seconds": 1.0,
                "peak_allocated_bytes": 1024,
                "device_total_memory_bytes": 2048,
            },
            "screen_sample_set_sha256": sha256(f"screen-{arm}".encode()).hexdigest(),
            "distribution": {
                "metrics": {
                    "fid": fid,
                    "inception_score_mean": 4.0,
                    "precision": 0.20,
                    "recall": recall,
                },
                "real_set": {"sha256": "3" * 64},
                "runtime_environment_sha256": evaluator_sha,
            },
            "class_fidelity": {
                "metrics": {
                    "top1_accuracy": 0.03,
                    "top5_accuracy": 0.10,
                    "predicted_class_fraction": 0.40,
                    "normalized_predicted_class_entropy": 0.70,
                },
                "classifier": {"sha256": "4" * 64},
                "runtime_environment_sha256": evaluator_sha,
            },
            "screen_checkpoint_evaluation": {"summary": diagnostic},
            "screen_rollout": {
                "terminal_raw_x0_clipping": {
                    "raw_x0_clip_fraction": 0.90 if endpoint else 0.99
                }
            },
        }
        arm_ids[arm] = _identity(f"confirmation-{arm}")
    return preparation, preparation_id, launch, launch_id, arms, arm_ids


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_result.validate_terminal_snr_confirmation_preparation_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_result.validate_terminal_snr_confirmation_launch_receipt_contract",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_confirmation_result.validate_terminal_snr_confirmation_arm_validation",
        lambda value: value,
    )


def _build(monkeypatch: pytest.MonkeyPatch, **options):
    _patch(monkeypatch)
    preparation, prep_id, launch, launch_id, arms, arm_ids = _sources(**options)
    return build_terminal_snr_confirmation_result(
        preparation=preparation,
        preparation_identity=prep_id,
        launch_receipt=launch,
        launch_receipt_identity=launch_id,
        arm_validations=arms,
        arm_validation_identities=arm_ids,
        result_git=GIT,
    )


def test_pass_requires_directional_absolute_matched_and_mechanism_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    assert report["confirmation_pass"] is True
    assert report["support_collapse_resolved"] is True
    assert report["failed_checks"] == []
    assert report["next_stage"]["large_capacity_readiness_preparation_allowed"] is True
    assert report["next_stage"]["full_300k_launch_allowed"] is False
    assert report["authorization_boundary"] == RESULT_BOUNDARY
    assert report["thresholds"] == CONFIRMATION_THRESHOLDS
    assert validate_terminal_snr_confirmation_result_contract(report) == report


def test_holds_when_absolute_support_is_too_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch, low_support=True)
    assert report["confirmation_pass"] is False
    assert report["support_collapse_resolved"] is False
    assert "endpoint0975_cofitok.recall_absolute" in report["failed_checks"]
    assert report["next_stage"]["large_capacity_readiness_preparation_allowed"] is False


def test_rejects_cross_arm_evaluator_runtime_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="evaluator runtime differs"):
        _build(monkeypatch, evaluator_mismatch=True)


def test_contract_rejects_implicit_300k_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered["next_stage"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_terminal_snr_confirmation_result_contract(altered)
