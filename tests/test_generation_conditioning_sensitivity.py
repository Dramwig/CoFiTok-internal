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
from scripts.build_generation_conditioning_sensitivity_comparison import (
    build_comparison,
)
from scripts.audit_generation_conditioning_path import (
    audit_conditioning_path,
    summarize_path_rows,
)
from scripts.evaluate_generation_conditioning_gain_sweep import (
    apply_class_embedding_gain,
    summarize_gain_rows,
)
from scripts.build_generation_conditioning_gain_sweep_comparison import (
    build_comparison as build_gain_comparison,
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


def _run_gain(
    checkpoint: Path,
    output: Path,
    *,
    resume: bool = False,
    check: bool = True,
):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src")]
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_generation_conditioning_gain_sweep.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--num-samples",
        "2",
        "--wrong-label-offset",
        "1",
        "--timesteps",
        "1",
        "--gains",
        "0",
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


def test_conditioning_gain_sweep_cli_is_cpu_only_and_resumable(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, num_classes=4)
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "conditioning_gain_sweep.json"
    _write_checkpoint(checkpoint, tmp_path)

    completed = _run_gain(checkpoint, output)
    assert completed.returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["runtime"]["device"] == "cpu"
    assert report["request"]["gains"] == [0.0, 1.0, 2.0]
    assert set(report["gain_summary"]) == {"0", "1", "2"}
    assert report["gain_summary"]["0"]["correct_better_count"] == {
        "than_wrong": 0,
        "than_null": 0,
    }
    assert report["claim_boundary"]["authorizes_training"] is False

    resumed = _run_gain(checkpoint, output, resume=True)
    assert resumed.returncode == 0


def _comparison_source(checkpoint_sha: str, *, synthesis_mode: str) -> dict:
    rows = []
    for index in range(2):
        rows.append(
            {
                "sample_index": index,
                "timestep": 5,
                "correct_label": index,
                "wrong_label": index + 1,
                "noise_seed": 100 + index,
                "relative_delta_to_correct_rms": {
                    "correct_vs_wrong": 0.01 + index * 0.001,
                    "correct_vs_null": 0.009 + index * 0.001,
                    "wrong_vs_null": 0.008,
                },
                "correct_relative_mse_improvement": {
                    "versus_wrong": 0.001 if index == 0 else -0.001,
                    "versus_null": -0.002,
                },
                "correct_better": {
                    "than_wrong": index == 0,
                    "than_null": False,
                },
            }
        )
    return {
        "git": {"revision": "a" * 40, "branch": "diagnostic", "tracked_dirty": False},
        "request": {
            "weights": "ema",
            "num_samples": 2,
            "start_label": 0,
            "wrong_label_offset": 1,
            "timesteps": [5],
            "noise_seed": 100,
            "threads": 1,
        },
        "dataset": {"alias": "imagenet_256", "samples": [1, 2]},
        "weights": "ema",
        "checkpoint": {"sha256": checkpoint_sha},
        "model": {
            "predictor_type": "scalable_unet",
            "synthesis_mode": synthesis_mode,
            "num_classes": 4,
        },
        "sample_rows": rows,
        "parameter_update_audit": {
            "ema": {
                "predictor.class_embed.weight": {"relative_update_rms": 0.03}
            }
        },
    }


def test_build_conditioning_comparison_reports_matched_sign_tests() -> None:
    cofitok = _comparison_source("a" * 64, synthesis_mode="fixed_basis")
    dense = _comparison_source("b" * 64, synthesis_mode="dense_identity")

    report = build_comparison(
        cofitok=cofitok,
        dense=dense,
        sources={"cofitok_k8": {"sha256": "c" * 64}, "dense_identity": {"sha256": "d" * 64}},
        git={"revision": "e" * 40, "branch": "comparison", "tracked_dirty": False},
    )

    assert report["status"] == "completed"
    assert report["paired_differences"]["row_count"] == 2
    assert report["methods"]["cofitok_k8"]["correct_better_count"] == {
        "than_wrong": 1,
        "than_null": 0,
    }
    assert report["diagnostic_interpretation"][
        "shared_conditioning_weakness_supported"
    ] is True


def test_conditioning_path_audit_reports_every_block_and_timestep() -> None:
    config = ModelConfig(
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
        num_classes=4,
        synthesis_mode="dense_identity",
    )
    model = CoFiTokTiny(config).eval()

    audit = audit_conditioning_path(
        model=model,
        labels=[0, 1],
        wrong_label_offset=1,
        timesteps=[1, 2],
    )

    assert audit["row_count"] == 4
    assert audit["block_count"] == 4
    assert len(audit["summary"]) == 2
    assert audit["summary"][0]["label_count"] == 2
    assert (
        audit["summary"][0]["block_modulation_relative_delta"]
        ["correct_vs_wrong"]["mean"]
        > 0.0
    )


def test_summarize_path_rows_rejects_missing_timestep() -> None:
    with pytest.raises(ValueError, match="No conditioning path rows"):
        summarize_path_rows([], [5])


def test_apply_class_embedding_gain_preserves_null_and_scales_displacement() -> None:
    config = ModelConfig(
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
        num_classes=4,
        synthesis_mode="dense_identity",
    )
    model = CoFiTokTiny(config)
    base = model.predictor.class_embed.weight.detach().clone()
    null = base[4].clone()

    apply_class_embedding_gain(model, base, 2.0)

    updated = model.predictor.class_embed.weight.detach()
    assert torch.equal(updated[4], null)
    assert torch.allclose(updated[0] - null, 2.0 * (base[0] - null))


def test_summarize_gain_rows_tracks_correct_label_advantage() -> None:
    rows = [
        {
            "correct_better": {"than_wrong": True, "than_null": False},
            "correct_relative_mse_improvement": {
                "versus_wrong": 0.1,
                "versus_null": -0.1,
            },
            "relative_delta_to_correct_rms": {
                "correct_vs_wrong": 0.2,
                "correct_vs_null": 0.3,
                "wrong_vs_null": 0.4,
            },
        },
        {
            "correct_better": {"than_wrong": False, "than_null": True},
            "correct_relative_mse_improvement": {
                "versus_wrong": -0.1,
                "versus_null": 0.1,
            },
            "relative_delta_to_correct_rms": {
                "correct_vs_wrong": 0.4,
                "correct_vs_null": 0.5,
                "wrong_vs_null": 0.6,
            },
        },
    ]

    summary = summarize_gain_rows({"2": rows})["2"]

    assert summary["correct_better_count"] == {"than_wrong": 1, "than_null": 1}
    assert summary["correct_relative_mse_improvement_mean"][
        "versus_wrong"
    ] == pytest.approx(0.0)
    assert summary["relative_delta_to_correct_rms_mean"][
        "correct_vs_wrong"
    ] == pytest.approx(0.3)


def _gain_comparison_source(
    checkpoint_sha: str,
    *,
    synthesis_mode: str,
    recovered: bool,
) -> dict:
    def gain_summary(
        *,
        count: int,
        improvement: float,
        delta: float,
    ) -> dict:
        return {
            "row_count": 8,
            "correct_better_count": {
                "than_wrong": count,
                "than_null": count,
            },
            "correct_relative_mse_improvement_mean": {
                "versus_wrong": improvement,
                "versus_null": improvement,
            },
            "relative_delta_to_correct_rms_mean": {
                "correct_vs_wrong": delta,
                "correct_vs_null": delta,
                "wrong_vs_null": delta,
            },
        }

    return {
        "git": {
            "revision": "a" * 40,
            "branch": "conditioning-gain",
            "tracked_dirty": False,
        },
        "request": {
            "weights": "ema",
            "num_samples": 8,
            "start_label": 0,
            "wrong_label_offset": 2,
            "timesteps": [5],
            "gains": [1.0, 2.0],
            "noise_seed": 100,
            "threads": 1,
        },
        "dataset": {"alias": "imagenet_256", "samples": list(range(8))},
        "checkpoint": {"sha256": checkpoint_sha},
        "model": {
            "predictor_type": "scalable_unet",
            "synthesis_mode": synthesis_mode,
            "num_classes": 1000,
        },
        "gain_summary": {
            "1": gain_summary(count=4, improvement=0.0, delta=0.1),
            "2": gain_summary(
                count=8 if recovered else 4,
                improvement=0.1 if recovered else 0.0,
                delta=0.2,
            ),
        },
    }


@pytest.mark.parametrize(
    ("recovered", "expected_action"),
    [
        (True, "validate_shared_conditioning_gain_in_sampling"),
        (
            False,
            "develop_matched_training_time_label_ranking_or_contrastive_"
            "denoising_loss",
        ),
    ],
)
def test_build_gain_comparison_selects_evidence_based_next_action(
    recovered: bool,
    expected_action: str,
) -> None:
    cofitok = _gain_comparison_source(
        "a" * 64,
        synthesis_mode="fixed_basis",
        recovered=recovered,
    )
    dense = _gain_comparison_source(
        "b" * 64,
        synthesis_mode="dense_identity",
        recovered=recovered,
    )

    report = build_gain_comparison(
        cofitok=cofitok,
        dense=dense,
        sources={"cofitok_k8": {"sha256": "c" * 64}, "dense_identity": {"sha256": "d" * 64}},
        git={"revision": "e" * 40, "branch": "comparison", "tracked_dirty": False},
    )

    interpretation = report["diagnostic_interpretation"]
    assert interpretation["shared_inference_gain_recovery_supported"] is recovered
    assert interpretation["recommended_next_action"] == expected_action
    assert report["claim_boundary"]["authorizes_training"] is False
