from __future__ import annotations

import argparse

import torch

from cofitok.configs import (
    DataConfig,
    DiffusionConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    config_to_dict,
)
from cofitok.generation import GenerationRequest, GenerationSession
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import checkpoint_integrity_path
from scripts.infer_generation import run_inference


def _checkpoint(tmp_path):
    config = ExperimentConfig(
        name="generation_session_cpu",
        data=DataConfig(image_size=8, channels=3),
        diffusion=DiffusionConfig(num_train_timesteps=4, schedule_type="cosine"),
        model=ModelConfig(
            image_channels=3,
            image_size=8,
            token_count=2,
            token_channels=4,
            base_channels=8,
            predictor_type="scalable_unet",
            predictor_channel_multipliers=[1],
            predictor_num_res_blocks=1,
            predictor_attention_resolutions=[],
            predictor_num_heads=1,
            num_classes=5,
            synthesis_active_token_channels=[2, 4],
        ),
        runtime=RuntimeConfig(device="cpu", precision="fp32"),
        optimization=OptimizationConfig(ema_warmup_steps=0),
    )
    model = CoFiTokTiny(config.model)
    ema = ExponentialMovingAverage(model, warmup_steps=0)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "config": config_to_dict(config),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "step": 23,
        },
        path,
    )
    write_json_report(
        checkpoint_integrity_path(path),
        {
            "schema_version": 1,
            "checkpoint": path.name,
            "checkpoint_bytes": path.stat().st_size,
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_format_version": 1,
            "step": 23,
        },
    )
    return path


def test_generation_session_reuses_loaded_checkpoint_with_provenance(tmp_path) -> None:
    session = GenerationSession.from_checkpoint(_checkpoint(tmp_path), weights="ema")
    request = GenerationRequest(
        seeds=(11, 12),
        class_labels=(1, 2),
        sample_steps=2,
        prefix_budget=2,
        guidance_scale=1.5,
        precision="fp32",
    )

    first = session.generate(request)
    second = session.generate(request)

    torch.testing.assert_close(first.images, second.images, rtol=0.0, atol=0.0)
    assert first.images.device.type == "cpu"
    assert first.images.shape == (2, 3, 8, 8)
    assert first.metadata["checkpoint_step"] == 23
    assert len(first.metadata["checkpoint_sha256"]) == 64
    assert first.metadata["weights"] == "ema"
    assert first.metadata["request"]["seeds"] == [11, 12]
    assert first.metadata["request"]["class_labels"] == [1, 2]
    assert first.metadata["request"]["prefix_budget"] == 2


def test_generation_session_validates_model_specific_request(tmp_path) -> None:
    session = GenerationSession.from_checkpoint(_checkpoint(tmp_path), weights="ema")

    try:
        session.generate(
            GenerationRequest(
                seeds=(1,),
                class_labels=(5,),
                sample_steps=1,
                guidance_scale=1.0,
                precision="fp32",
            )
        )
    except ValueError as error:
        assert "class range" in str(error)
    else:
        raise AssertionError("out-of-range class label was accepted")

    try:
        session.generate(
            GenerationRequest(
                seeds=(1,),
                class_labels=None,
                sample_steps=1,
                guidance_scale=1.0,
                precision="fp32",
            )
        )
    except ValueError as error:
        assert "requires class labels" in str(error)
    else:
        raise AssertionError("missing class labels were accepted")


def test_inference_cli_core_writes_atomic_provenance_report(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "inference"
    report = run_inference(
        argparse.Namespace(
            checkpoint=str(checkpoint),
            output_dir=str(output_dir),
            class_ids="3",
            seeds="7,9",
            seed=0,
            num_images=1,
            prefix_budgets="1,2",
            batch_size=2,
            sample_steps=1,
            guidance_scale=1.0,
            guidance_rescale=0.0,
            cfg_batch_mode="batched",
            eta=0.0,
            weights="ema",
            precision="fp32",
            overwrite=False,
        )
    )

    assert report["status"] == "completed"
    assert report["output_count"] == 4
    assert report["checkpoint"]["checkpoint_step"] == 23
    assert report["request"]["seeds"] == [7, 9]
    assert report["request"]["class_ids"] == [3, 3]
    assert report["request"]["prefix_budgets"] == [1, 2]
    assert (output_dir / "inference_report.json").is_file()
    for output in report["outputs"]:
        assert len(output["sha256"]) == 64
        assert (output_dir / output["filename"]).is_file()
