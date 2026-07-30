from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from cofitok.configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    config_to_dict,
)
from cofitok.environment import runtime_environment_sha256
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import checkpoint_integrity_path
from scripts.preflight_generation_sampling import run_sampling_preflight


ROOT = Path(__file__).resolve().parents[1]


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
            "format_version": 1,
            "config": config_to_dict(config),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "step": 17,
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
            "step": 17,
        },
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
    assert report["git"]["revision"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert isinstance(report["git"]["tracked_dirty"], bool)
    assert report["runtime_environment"]["device"]["type"] == "cpu"
    assert report["runtime_environment_sha256"] == runtime_environment_sha256(
        report["runtime_environment"]
    )
    assert report["checkpoint_step"] == 17
    assert report["release_authorization_required"] is False
    assert len(report["checkpoint_sha256"]) == 64
    assert report["checkpoint_integrity_manifest"].endswith("checkpoint.pt.integrity.json")
    assert report["request"]["effective_model_batch_size"] == 6
    assert report["request"]["forward_passes"] == 1
    assert report["result"]["output_shape"] == [3, 3, 8, 8]
    assert report["result"]["output_finite"] is True
    assert report["result"]["cuda_memory_after_forward"] is None


def test_sampling_preflight_measures_warm_and_repeated_forwards(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_cpu_checkpoint(checkpoint)

    report = run_sampling_preflight(
        checkpoint,
        batch_size=2,
        guidance_scale=1.5,
        precision="fp32",
        warmup_forwards=1,
        measured_forwards=2,
    )

    assert report["status"] == "passed"
    assert report["request"]["warmup_forwards"] == 1
    assert report["request"]["measured_forwards"] == 2
    assert len(report["result"]["durations_seconds"]) == 2
    assert report["result"]["mean_forward_seconds"] > 0.0
    assert report["result"]["output_images_per_second"] > 0.0


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


def test_sampling_preflight_rejects_checkpoint_bytes_that_fail_integrity(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_cpu_checkpoint(checkpoint)
    payload = bytearray(checkpoint.read_bytes())
    payload[len(payload) // 2] ^= 1
    checkpoint.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_sampling_preflight(checkpoint, batch_size=1, precision="fp32")


def test_sampling_preflight_release_policy_precedes_deserialization(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_cpu_checkpoint(checkpoint)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("checkpoint was deserialized before policy rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="release-authorized inference artifact"):
        run_sampling_preflight(
            checkpoint,
            batch_size=1,
            precision="fp32",
            require_release_authorization=True,
        )
