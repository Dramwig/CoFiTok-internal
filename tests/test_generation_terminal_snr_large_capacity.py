from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from cofitok.generation.terminal_snr_large_capacity import (
    BASE_CHANNELS,
    CONFIG_FILENAMES,
    CONFIG_NAMES,
    EFFECTIVE_BATCH_SIZE,
    ENDPOINT_FRACTION,
    EXPECTED_PARAMETER_COUNTS,
    FORMAL_EVALUATION_CONTRACT,
    FULL_STAGE_DIRNAME,
    MILESTONE_STEPS,
    PREPARATION_BOUNDARY,
    TREND_EVALUATION_CONTRACT,
    build_terminal_snr_large_capacity_preparation,
    validate_terminal_snr_large_capacity_config_pair,
    validate_terminal_snr_large_capacity_preparation_contract,
)
from cofitok.generation_recipe import GENERATION_TRAINING_RECIPE_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/generation"
OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    f"{FULL_STAGE_DIRNAME}"
)
CONFIRMATION_GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/generation-terminal-snr-confirmation-v1",
    "tracked_dirty": False,
}
PREPARATION_GIT = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "scale/generation-terminal-snr-capacity-full-v1",
    "tracked_dirty": False,
}


def _read_config(method: str) -> dict:
    return json.loads(
        (CONFIG_ROOT / CONFIG_FILENAMES[method]).read_text(encoding="utf-8")
    )


def _identity(path: str) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(path) + 1,
        "sha256": sha256(path.encode()).hexdigest(),
    }


def _source_objects(*, passed: bool = True, adjacent: bool = True):
    result_path = "/tmp/confirmation/terminal_snr_confirmation_result.json"
    result_id = _identity(result_path)
    validation_path = (
        "/tmp/confirmation/terminal_snr_confirmation_result.validation.json"
        if adjacent
        else "/tmp/other/terminal_snr_confirmation_result.validation.json"
    )
    validation_id = _identity(validation_path)
    result = {
        "scientific_status": "confirmation_pass" if passed else "hold",
        "confirmation_pass": passed,
        "support_collapse_resolved": passed,
        "failed_checks": [] if passed else ["endpoint0975_cofitok.recall_absolute"],
        "decision": (
            "prepare_separate_large_capacity_readiness"
            if passed
            else "hold_terminal_snr_intervention"
        ),
        "result_git": CONFIRMATION_GIT,
        "checks": [{"name": "support", "pass": passed}],
        "next_stage": {
            "route": "large_capacity_readiness_preparation" if passed else "hold",
            "support_collapse_resolved": passed,
            "large_capacity_readiness_preparation_allowed": passed,
            "large_capacity_readiness_launch_allowed": False,
            "separate_source_bound_authorization_required": True,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "claim_policy": {
            "confirmation_is_large_capacity_qualification_only": True,
            "support_collapse_resolved": passed,
        },
    }
    validation = {
        "status": "pass",
        "scientific_status": result["scientific_status"],
        "confirmation_pass": passed,
        "support_collapse_resolved": passed,
        "result": result_id,
        "result_git": CONFIRMATION_GIT,
        "validator_git": CONFIRMATION_GIT,
        "failed_checks": copy.deepcopy(result["failed_checks"]),
    }
    return result, result_id, validation, validation_id


def _config_validation() -> dict:
    return {
        "status": "pass",
        "recipe_schema": GENERATION_TRAINING_RECIPE_SCHEMA,
        "recipe_stage": "stability_full",
        "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
        "relative_parameter_gap": (
            EXPECTED_PARAMETER_COUNTS["cofitok"]
            - EXPECTED_PARAMETER_COUNTS["dense_identity"]
        )
        / EXPECTED_PARAMETER_COUNTS["dense_identity"],
        "base_channels": BASE_CHANNELS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "diffusion": {
            "num_train_timesteps": 1_000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_type": "cosine",
            "prediction_target": "epsilon",
            "cosine_endpoint_fraction": ENDPOINT_FRACTION,
        },
        "fresh_initialization_required": True,
        "resume_checkpoint_allowed": False,
        "cofitok_synthesis": {
            "mode": "fixed_basis",
            "kernel_size": 1,
            "gamma_mode": "fixed_one",
            "current_token_only": True,
            "zero_token_maps_to_zero": True,
        },
        "pair_contract_sha256": "e" * 64,
        "recipe_contract_sha256": "f" * 64,
    }


def _patch_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_large_capacity."
        "validate_terminal_snr_confirmation_result_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_large_capacity."
        "replay_terminal_snr_confirmation_result",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_large_capacity."
        "validate_terminal_snr_confirmation_validation_receipt",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_large_capacity."
        "validate_terminal_snr_large_capacity_config_pair",
        lambda **_: _config_validation(),
    )


def _build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    passed: bool = True,
    adjacent: bool = True,
) -> dict:
    _patch_replay(monkeypatch)
    result, result_id, validation, validation_id = _source_objects(
        passed=passed, adjacent=adjacent
    )
    return build_terminal_snr_large_capacity_preparation(
        confirmation_result=result,
        confirmation_result_identity=result_id,
        confirmation_validation=validation,
        confirmation_validation_identity=validation_id,
        cofitok_config=_read_config("cofitok"),
        cofitok_config_identity=_identity(
            f"/tmp/configs/{CONFIG_FILENAMES['cofitok']}"
        ),
        dense_config=_read_config("dense_identity"),
        dense_config_identity=_identity(
            f"/tmp/configs/{CONFIG_FILENAMES['dense_identity']}"
        ),
        preparation_git=PREPARATION_GIT,
        output_root=OUTPUT_ROOT,
    )


def test_endpoint0975_full_configs_only_add_the_selected_diffusion_endpoint() -> None:
    source_names = {
        "cofitok": (
            "imagenet256_stability_rgbtail3_rollout_x0_u2_"
            "ema_teacher_k8_300k.json"
        ),
        "dense_identity": (
            "imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json"
        ),
    }
    for method in ("cofitok", "dense_identity"):
        source = json.loads(
            (CONFIG_ROOT / source_names[method]).read_text(encoding="utf-8")
        )
        expected = copy.deepcopy(source)
        expected["name"] = CONFIG_NAMES[method]
        expected["diffusion"] = {
            "num_train_timesteps": 1_000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_type": "cosine",
            "prediction_target": "epsilon",
            "cosine_endpoint_fraction": 0.975,
        }
        assert _read_config(method) == expected


def test_endpoint0975_full_pair_is_matched_fresh_250m_300k() -> None:
    report = validate_terminal_snr_large_capacity_config_pair(
        cofitok_config=_read_config("cofitok"),
        dense_config=_read_config("dense_identity"),
    )
    assert report["status"] == "pass"
    assert report["parameter_counts"] == EXPECTED_PARAMETER_COUNTS
    assert report["base_channels"] == 256
    assert report["effective_batch_size"] == 64
    assert report["diffusion"]["cosine_endpoint_fraction"] == 0.975
    assert report["fresh_initialization_required"] is True
    assert report["resume_checkpoint_allowed"] is False
    assert report["cofitok_synthesis"]["current_token_only"] is True
    assert report["cofitok_synthesis"]["zero_token_maps_to_zero"] is True


def test_preparation_predeclares_milestones_and_formal_50k_ddim250(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    assert [row["step"] for row in report["milestone_contract"]] == list(
        MILESTONE_STEPS
    )
    assert all(
        row["trend_evaluation"] == TREND_EVALUATION_CONTRACT
        for row in report["milestone_contract"]
    )
    assert report["formal_evaluation_contract"] == FORMAL_EVALUATION_CONTRACT
    assert report["formal_evaluation_contract"]["samples_per_method"] == 50_000
    assert report["formal_evaluation_contract"]["sample_steps"] == 250
    assert report["selection"]["fresh_initialization_required"] is True
    assert report["selection"]["resume_checkpoint_allowed"] is False
    assert report["authorization_boundary"] == PREPARATION_BOUNDARY
    assert report["next_stage"]["execution_ready"] is False
    assert report["next_stage"]["full_300k_launch_allowed"] is False
    assert (
        validate_terminal_snr_large_capacity_preparation_contract(report)
        == report
    )


def test_preparation_rejects_a_nonpassing_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="exact passing confirmation"):
        _build(monkeypatch, passed=False)


def test_preparation_requires_adjacent_validation_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="not adjacent"):
        _build(monkeypatch, adjacent=False)


def test_preparation_requires_same_exact_result_and_validator_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_replay(monkeypatch)
    result, result_id, validation, validation_id = _source_objects()
    validation["validator_git"] = {
        **CONFIRMATION_GIT,
        "revision": "e" * 40,
        "tree": "f" * 40,
    }
    with pytest.raises(ValueError, match="exact passing confirmation"):
        build_terminal_snr_large_capacity_preparation(
            confirmation_result=result,
            confirmation_result_identity=result_id,
            confirmation_validation=validation,
            confirmation_validation_identity=validation_id,
            cofitok_config=_read_config("cofitok"),
            cofitok_config_identity=_identity(
                f"/tmp/configs/{CONFIG_FILENAMES['cofitok']}"
            ),
            dense_config=_read_config("dense_identity"),
            dense_config_identity=_identity(
                f"/tmp/configs/{CONFIG_FILENAMES['dense_identity']}"
            ),
            preparation_git=PREPARATION_GIT,
            output_root=OUTPUT_ROOT,
        )


def test_contract_rejects_implicit_execution_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered["next_stage"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_terminal_snr_large_capacity_preparation_contract(altered)


@pytest.mark.parametrize(
    "section",
    [
        "source_evidence",
        "qualification",
        "selection",
        "next_stage",
        "claim_policy",
    ],
)
def test_contract_rejects_unexpected_nested_fields(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
) -> None:
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered[section]["unexpected_authority"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_terminal_snr_large_capacity_preparation_contract(altered)


def test_contract_rejects_unexpected_config_validation_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered["selection"]["config_validation"]["execution_ready"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_terminal_snr_large_capacity_preparation_contract(altered)
