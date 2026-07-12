from __future__ import annotations

import copy

import pytest

from scripts.validate_generation_training_pair import validate_training_pair


REVISION = "a" * 40


def _report(*, parameters: int, token_count: int) -> dict:
    return {
        "training_complete": True,
        "completed_steps": 50_000,
        "target_steps": 50_000,
        "parameter_count": parameters,
        "git": {
            "dirty": False,
            "revision": REVISION,
            "branch": "scale/generative-system",
        },
        "latest_checkpoint": {
            "checkpoint": "checkpoint_step_00050000.pt",
            "step": 50_000,
        },
        "config": {
            "data": {"dataset": "imagenet_256_10pct", "batch_size": 16},
            "diffusion": {"schedule_type": "cosine"},
            "runtime": {"steps": 50_000},
            "optimization": {"gradient_accumulation_steps": 4},
            "model": {
                "image_channels": 3,
                "image_size": 256,
                "token_count": token_count,
                "token_channels": 64 if token_count > 1 else 3,
                "base_channels": 128,
                "predictor_type": "scalable_unet",
                "predictor_use_feedback": token_count > 1,
                "predictor_channel_multipliers": [1, 2, 3, 4],
                "predictor_num_res_blocks": 2,
                "predictor_attention_resolutions": [32, 16],
                "predictor_num_heads": 8,
                "predictor_dropout": 0.0,
                "predictor_gradient_checkpointing": True,
                "num_classes": 1000,
                "class_dropout_prob": 0.1,
                "synthesis_mode": "restricted" if token_count > 1 else "dense_identity",
            },
            "loss": {
                "epsilon_weight": 1.0,
                "denoise_path_component_weight": 0.1 if token_count > 1 else 0.0,
            },
        },
    }


def test_validator_accepts_completed_matched_pair() -> None:
    report = validate_training_pair(
        _report(parameters=100_500, token_count=8),
        _report(parameters=100_000, token_count=1),
        expected_steps=50_000,
        expected_revision=REVISION,
    )

    assert report["status"] == "pass"
    assert report["relative_parameter_gap"] == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["git"].update(revision="b" * 40), "revision"),
        (lambda report: report["git"].update(dirty=True), "dirty"),
        (lambda report: report.update(training_complete=False), "incomplete"),
        (lambda report: report["latest_checkpoint"].update(step=49_999), "checkpoint step"),
        (lambda report: report["config"]["data"].update(dataset="imagenet_256"), "dataset"),
    ],
)
def test_validator_rejects_invalid_dense_report(mutation, message: str) -> None:
    dense = _report(parameters=100_000, token_count=1)
    mutation(dense)

    with pytest.raises(ValueError, match=message):
        validate_training_pair(
            _report(parameters=100_500, token_count=8),
            dense,
            expected_steps=50_000,
            expected_revision=REVISION,
        )


def test_validator_rejects_protocol_or_parameter_mismatch() -> None:
    dense = _report(parameters=90_000, token_count=1)
    dense["config"]["optimization"]["gradient_accumulation_steps"] = 2

    with pytest.raises(ValueError, match="mismatched config sections"):
        validate_training_pair(
            _report(parameters=100_500, token_count=8),
            dense,
            expected_steps=50_000,
            expected_revision=REVISION,
        )

    matched_dense = copy.deepcopy(dense)
    matched_dense["config"]["optimization"]["gradient_accumulation_steps"] = 4
    with pytest.raises(ValueError, match="parameter gap"):
        validate_training_pair(
            _report(parameters=100_500, token_count=8),
            matched_dense,
            expected_steps=50_000,
            expected_revision=REVISION,
        )


def test_validator_rejects_shared_model_or_dense_auxiliary_drift() -> None:
    dense = _report(parameters=100_000, token_count=1)
    dense["config"]["model"]["class_dropout_prob"] = 0.2
    with pytest.raises(ValueError, match="mismatched shared model fields"):
        validate_training_pair(
            _report(parameters=100_500, token_count=8),
            dense,
            expected_steps=50_000,
            expected_revision=REVISION,
        )

    dense = _report(parameters=100_000, token_count=1)
    dense["config"]["loss"]["denoise_path_component_weight"] = 0.1
    with pytest.raises(ValueError, match="dense baseline has nonzero"):
        validate_training_pair(
            _report(parameters=100_500, token_count=8),
            dense,
            expected_steps=50_000,
            expected_revision=REVISION,
        )
