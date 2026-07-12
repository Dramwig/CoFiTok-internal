from __future__ import annotations

import torch

from cofitok.configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    config_to_dict,
)
from cofitok.models import CoFiTokTiny
from cofitok.training import ExponentialMovingAverage
from scripts.preflight_generation_sampling import run_sampling_preflight


def _write_cpu_checkpoint(path) -> None:
    config = ExperimentConfig(
        name="sampling_preflight_cpu",
        data=DataConfig(image_size=8, channels=3),
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
    torch.save(
        {
            "config": config_to_dict(config),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "step": 17,
        },
        path,
    )


def test_sampling_preflight_runs_shared_ema_cfg_path(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_cpu_checkpoint(checkpoint)

    report = run_sampling_preflight(
        checkpoint,
        batch_size=3,
        prefix_budget=2,
        guidance_scale=1.5,
        cfg_batch_mode="batched",
        weights="ema",
        precision="fp32",
    )

    assert report["status"] == "passed"
    assert report["checkpoint_step"] == 17
    assert len(report["checkpoint_sha256"]) == 64
    assert report["request"]["effective_model_batch_size"] == 6
    assert report["request"]["forward_passes"] == 1
    assert report["result"]["output_shape"] == [3, 3, 8, 8]
    assert report["result"]["output_finite"] is True
    assert report["result"]["cuda_memory_after_forward"] is None


def test_sampling_preflight_rejects_invalid_prefix_before_forward(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_cpu_checkpoint(checkpoint)

    try:
        run_sampling_preflight(checkpoint, batch_size=1, prefix_budget=3, precision="fp32")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("invalid prefix budget was accepted")


def test_sampling_preflight_retains_checkpoint_provenance_on_forward_failure(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_cpu_checkpoint(checkpoint)

    def fail_forward(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("scripts.preflight_generation_sampling.predict_epsilon", fail_forward)
    report = run_sampling_preflight(checkpoint, batch_size=2, precision="fp32")

    assert report["status"] == "failed"
    assert report["checkpoint_step"] == 17
    assert len(report["checkpoint_sha256"]) == 64
    assert report["error_type"] == "RuntimeError"
    assert report["error"] == "CUDA out of memory"
