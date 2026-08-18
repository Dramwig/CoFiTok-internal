from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from cofitok.configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    config_to_dict,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import checkpoint_integrity_path
from cofitok.utils.seed import seed_everything
from scripts.evaluate_generation_conditioning_sensitivity import (
    parameter_update_statistics,
    summarize_timestep_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(root: Path, num_classes: int) -> Path:
    dataset_root = root / "imagenet_256"
    mapping = {str(index): f"n{index:08d}" for index in range(num_classes)}
    metadata = dataset_root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "label_to_wnid.json").write_text(
        json.dumps({"label_to_wnid": mapping}),
        encoding="utf-8",
    )
    for index, wnid in enumerate(mapping.values()):
        class_dir = dataset_root / "extracted" / "val" / wnid
        class_dir.mkdir(parents=True)
        image = Image.new("RGB", (8, 8), color=(index * 30, 20, 40))
        image.save(class_dir / f"sample_{index}.png")
    return dataset_root


def _write_checkpoint(path: Path, dataset_parent: Path) -> None:
    config = ExperimentConfig(
        name="conditioning_sensitivity_cpu",
        data=DataConfig(
            dataset="imagenet_256",
            root=str(dataset_parent),
            image_size=8,
            channels=3,
            batch_size=2,
            class_conditional=True,
        ),
        model=ModelConfig(
            image_channels=3,
            image_size=8,
            token_count=1,
            token_channels=3,
            base_channels=8,
            predictor_type="scalable_unet",
            predictor_use_feedback=False,
            predictor_channel_multipliers=[1],
            predictor_num_res_blocks=1,
            predictor_attention_resolutions=[],
            predictor_num_heads=1,
            predictor_gradient_checkpointing=False,
            num_classes=4,
            class_dropout_prob=0.1,
            synthesis_mode="dense_identity",
        ),
        runtime=RuntimeConfig(device="cuda", precision="bf16", seed=17),
        optimization=OptimizationConfig(ema_warmup_steps=0),
    )
    seed_everything(config.runtime.seed)
    model = CoFiTokTiny(config.model)
    ema = ExponentialMovingAverage(model, warmup_steps=0)
    with torch.no_grad():
        model.predictor.class_embed.weight.add_(0.05)
        model.predictor.input_proj.weight.add_(0.01)
    ema.update(model)
    torch.save(
        {
            "format_version": 1,
            "config": config_to_dict(config),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "step": 11,
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
            "step": 11,
        },
    )


def _run(checkpoint: Path, output: Path, *, resume: bool = False, check: bool = True):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_generation_conditioning_sensitivity.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--num-samples",
        "2",
        "--wrong-label-offset",
        "1",
        "--timesteps",
        "1",
        "2",
        "--threads",
        "1",
    ]
    if resume:
        command.append("--resume")
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def test_parameter_update_statistics_handles_zero_initialization() -> None:
    result = parameter_update_statistics(torch.zeros(2), torch.ones(2))

    assert result["initial_zero"] is True
    assert result["relative_update_rms"] is None
    assert result["cosine_initial_current"] is None
    assert result["update_rms"] == pytest.approx(1.0)


def test_summarize_timestep_rows_preserves_paired_counts() -> None:
    rows = []
    for index, improvement in enumerate((0.1, -0.2)):
        rows.append(
            {
                "timestep": 5,
                "conditions": {
                    name: {
                        "epsilon_mse_to_noise": 1.0 + index,
                        "x0_mse_to_clean": 2.0 + index,
                        "epsilon_rms": 3.0 + index,
                    }
                    for name in ("correct", "wrong", "null")
                },
                "relative_delta_to_correct_rms": {
                    "correct_vs_wrong": 0.01,
                    "correct_vs_null": 0.02,
                    "wrong_vs_null": 0.03,
                },
                "correct_relative_mse_improvement": {
                    "versus_wrong": improvement,
                    "versus_null": improvement / 2,
                },
                "correct_better": {
                    "than_wrong": improvement > 0,
                    "than_null": improvement > 0,
                },
            }
        )

    summary = summarize_timestep_rows(rows, [5])[0]

    assert summary["sample_count"] == 2
    assert summary["correct_better_count"] == {"than_wrong": 1, "than_null": 1}
    assert summary["correct_relative_mse_improvement"]["versus_wrong"][
        "mean"
    ] == pytest.approx(-0.05)


def test_conditioning_sensitivity_cli_is_cpu_only_and_resumable(tmp_path: Path) -> None:
    _write_dataset(tmp_path, num_classes=4)
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "conditioning"
    _write_checkpoint(checkpoint, tmp_path)

    completed = _run(checkpoint, output)
    assert completed.returncode == 0
    report = json.loads(
        (output / "conditioning_sensitivity_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "completed"
    assert report["runtime"]["device"] == "cpu"
    assert report["request"]["num_samples"] == 2
    assert len(report["sample_rows"]) == 4
    assert len(report["timesteps"]) == 2
    assert report["claim_boundary"]["authorizes_training"] is False
    class_update = report["parameter_update_audit"]["raw"][
        "predictor.class_embed.weight"
    ]
    assert class_update["relative_update_rms"] > 0.0

    resumed = _run(checkpoint, output, resume=True)
    assert resumed.returncode == 0


def test_conditioning_sensitivity_cli_rejects_existing_output_without_resume(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, num_classes=4)
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "conditioning"
    _write_checkpoint(checkpoint, tmp_path)
    _run(checkpoint, output)

    repeated = _run(checkpoint, output, check=False)

    assert repeated.returncode != 0
    assert "pass --resume" in repeated.stderr
