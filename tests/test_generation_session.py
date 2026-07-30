from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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
from cofitok.generation import GenerationRequest, GenerationSession
from cofitok.generation.protocol import INFERENCE_API, sampling_protocol_contract
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import checkpoint_integrity_path
from scripts.infer_generation import run_inference


ROOT = Path(__file__).resolve().parents[1]


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
    assert first.metadata["inference_api"] == INFERENCE_API
    assert first.metadata["sampling"]["protocol_schema"] == "cofitok_ddim_sampling_v1"
    assert sampling_protocol_contract(first.metadata["sampling"])["valid"] is True


def test_generation_session_random_stream_is_batch_size_invariant(tmp_path) -> None:
    session = GenerationSession.from_checkpoint(_checkpoint(tmp_path), weights="ema")
    combined = session.generate(
        GenerationRequest(
            seeds=(11, 12),
            class_labels=(1, 2),
            sample_steps=2,
            prefix_budget=2,
            guidance_scale=1.5,
            precision="fp32",
        )
    )
    split = torch.cat(
        [
            session.generate(
                GenerationRequest(
                    seeds=(seed,),
                    class_labels=(label,),
                    sample_steps=2,
                    prefix_budget=2,
                    guidance_scale=1.5,
                    precision="fp32",
                )
            ).images
            for seed, label in ((11, 1), (12, 2))
        ]
    )

    torch.testing.assert_close(combined.images, split, rtol=0.0, atol=0.0)
    assert combined.metadata["sampling"]["random_stream"] == {
        "scope": "per_request_seed",
        "seed_formula": "explicit_seed",
        "batch_size_invariant": True,
    }


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


def test_generation_session_production_mode_rejects_training_checkpoint(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="release-authorized inference artifact"):
        GenerationSession.from_checkpoint(
            _checkpoint(tmp_path),
            weights="ema",
            require_release_authorization=True,
        )


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
    assert report["inference_api"] == INFERENCE_API
    assert report["sampling_protocol_schema"] == "cofitok_ddim_sampling_v1"
    assert report["request"]["seeds"] == [7, 9]
    assert report["request"]["class_ids"] == [3, 3]
    assert report["request"]["prefix_budgets"] == [1, 2]
    assert (output_dir / "inference_report.json").is_file()
    for output in report["outputs"]:
        assert len(output["sha256"]) == 64
        assert (output_dir / output["filename"]).is_file()


def test_formal_sampling_cli_runs_checkpoint_to_png_and_report(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "formal_samples"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_samples.py"),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--num-samples",
            "1",
            "--batch-size",
            "1",
            "--sample-steps",
            "1",
            "--prefix-budgets",
            "2",
            "--guidance-scale",
            "1.0",
            "--weights",
            "ema",
            "--precision",
            "fp32",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output_dir / "sampling_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["sampling"]["actual_timesteps"] == [0]
    assert report["schema_version"] == 6
    assert report["sampling"]["protocol_schema"] == "cofitok_ddim_sampling_v1"
    assert report["sampling"]["sampler"] == "ddim"
    assert report["sampling"]["num_train_timesteps"] == 4
    assert report["sampling"]["clip_x0"] is True
    manifest = json.loads((output_dir / "sampling_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert report["runtime_environment"]["schema_version"] == 1
    assert len(report["runtime_environment_sha256"]) == 64
    assert report["sample_sets"]["2"]["count"] == 1
    assert len(report["sample_sets"]["2"]["sha256"]) == 64
    assert (output_dir / "prefix_2/000000.png").is_file()

    drifted_environment = dict(environment)
    drifted_environment["PYTHONHASHSEED"] = "314159"
    resumed = subprocess.run(
        [*result.args, "--resume"],
        cwd=ROOT,
        env=drifted_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode != 0
    assert "sampling manifest does not match" in resumed.stderr
