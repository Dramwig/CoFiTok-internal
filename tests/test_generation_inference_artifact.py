from __future__ import annotations

import pytest
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
from cofitok.generation import (
    GenerationRequest,
    GenerationSession,
    export_ema_inference_artifact,
    verify_inference_artifact,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import checkpoint_integrity_path


def _training_checkpoint(tmp_path):
    config = ExperimentConfig(
        name="inference_export_cpu",
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
    with torch.no_grad():
        for value in ema.shadow.values():
            if torch.is_floating_point(value):
                value.add_(0.01)
    path = tmp_path / "training.pt"
    torch.save(
        {
            "format_version": 1,
            "config": config_to_dict(config),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "step": 31,
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
            "step": 31,
        },
    )
    return path


def test_ema_export_is_smaller_verified_and_sample_equivalent(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"

    report = export_ema_inference_artifact(source, artifact)
    reused = export_ema_inference_artifact(source, artifact)

    assert report["status"] == "completed"
    assert report["weights"] == "ema_export"
    assert report["verified"] is True
    assert report["artifact_bytes"] < report["source_checkpoint_bytes"]
    assert reused["reused"] is True
    assert verify_inference_artifact(artifact)["artifact_sha256"] == report[
        "artifact_sha256"
    ]

    request = GenerationRequest(
        seeds=(9,),
        class_labels=(2,),
        sample_steps=1,
        guidance_scale=1.0,
        precision="fp32",
    )
    source_result = GenerationSession.from_checkpoint(source, weights="ema").generate(
        request
    )
    export_session = GenerationSession.from_checkpoint(artifact, weights="ema")
    export_result = export_session.generate(request)
    torch.testing.assert_close(
        source_result.images,
        export_result.images,
        rtol=0.0,
        atol=0.0,
    )
    assert export_result.metadata["weights"] == "ema_export"
    assert export_result.metadata["source_checkpoint_sha256"] == report[
        "source_checkpoint_sha256"
    ]


def test_inference_artifact_rejects_model_weights_and_tampering(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)

    with pytest.raises(ValueError, match="only exported EMA"):
        GenerationSession.from_checkpoint(artifact, weights="model")

    payload = bytearray(artifact.read_bytes())
    payload[len(payload) // 2] ^= 1
    artifact.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")
