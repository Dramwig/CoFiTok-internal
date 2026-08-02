from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import checkpoint_integrity_path
from scripts.evaluate_generation_checkpoint import (
    component_energy_statistics,
    component_orders,
    prefix_tensors,
    spatial_prefix_targets,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_cpu_checkpoint(path: Path) -> None:
    config = ExperimentConfig(
        name="checkpoint_evaluation_cpu",
        data=DataConfig(image_size=8, channels=3, batch_size=2),
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


def _run_checkpoint_evaluator(
    checkpoint: Path,
    output: Path,
    *,
    resume: bool = False,
    timestep: int = 1,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_generation_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--num-images",
        "2",
        "--timestep",
        str(timestep),
        "--random-orders",
        "1",
        "--weights",
        "ema",
        "--precision",
        "fp32",
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


def test_component_orders_are_deterministic_and_deduplicated() -> None:
    first = component_orders(token_count=4, random_orders=16, seed=7)
    second = component_orders(token_count=4, random_orders=16, seed=7)

    assert first == second
    assert first["ordered"] == [0, 1, 2, 3]
    assert first["reverse"] == [3, 2, 1, 0]
    assert len({tuple(order) for order in first.values()}) == len(first)


def test_prefix_tensors_respect_requested_order() -> None:
    components = [torch.full((1, 1, 1, 1), float(value)) for value in (1, 2, 4)]

    prefixes = prefix_tensors(components, [2, 0, 1])

    assert [float(value) for value in prefixes] == [4.0, 5.0, 7.0]


def test_component_energy_statistics_preserve_per_sample_collapse() -> None:
    first = torch.stack([torch.ones(1, 2, 2), torch.zeros(1, 2, 2)])
    second = torch.stack([torch.zeros(1, 2, 2), torch.ones(1, 2, 2)])

    statistics = component_energy_statistics([first, second])

    assert torch.equal(statistics["ratios"], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(statistics["uniform_mse"], torch.tensor([0.25, 0.25]))


def test_spatial_prefix_targets_end_at_clean_image() -> None:
    clean = torch.randn(2, 3, 16, 16)

    targets = spatial_prefix_targets(clean, count=4)

    assert len(targets) == 4
    assert torch.equal(targets[-1], clean)
    assert targets[0].shape == clean.shape


def test_checkpoint_evaluator_cli_records_git_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "evaluation"
    _write_cpu_checkpoint(checkpoint)
    _run_checkpoint_evaluator(checkpoint, output)
    report = json.loads(
        (output / "checkpoint_evaluation_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output / "checkpoint_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert report["schema_version"] == 2
    assert report["status"] == "completed"
    assert report["git"]["revision"] == revision
    assert isinstance(report["git"]["tracked_dirty"], bool)
    assert report["request"] == manifest["request"]
    assert report["manifest"]["path"] == (
        output / "checkpoint_evaluation_manifest.json"
    ).resolve().as_posix()
    assert len(report["manifest"]["sha256"]) == 64
    assert len(report["metrics"]["component_energy_ratio"]) == 2
    assert len(report["metrics"]["target_component_energy_ratio"]) == 2
    assert len(
        report["metrics"]["objective_target_component_energy_ratio_per_sample_mean"]
    ) == 2
    assert report["metrics"]["component_energy_uniform_mse_per_sample_mean"] >= 0.0
    assert report["metrics"]["target_component_energy_uniform_mse_per_sample_mean"] >= 0.0


def test_checkpoint_evaluator_resume_reuses_completed_report(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "evaluation"
    _write_cpu_checkpoint(checkpoint)
    _run_checkpoint_evaluator(checkpoint, output)
    report_path = output / "checkpoint_evaluation_report.json"
    original_bytes = report_path.read_bytes()
    original_mtime = report_path.stat().st_mtime_ns

    resumed = _run_checkpoint_evaluator(checkpoint, output, resume=True)

    assert "reused completed checkpoint evaluation" in resumed.stdout
    assert report_path.read_bytes() == original_bytes
    assert report_path.stat().st_mtime_ns == original_mtime


def test_checkpoint_evaluator_resume_can_start_fresh(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "evaluation"
    _write_cpu_checkpoint(checkpoint)

    completed = _run_checkpoint_evaluator(checkpoint, output, resume=True)

    assert "wrote" in completed.stdout
    assert (output / "checkpoint_evaluation_manifest.json").is_file()
    assert (output / "checkpoint_evaluation_report.json").is_file()


def test_checkpoint_evaluator_resume_reruns_incomplete_manifest(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "evaluation"
    _write_cpu_checkpoint(checkpoint)
    _run_checkpoint_evaluator(checkpoint, output)
    (output / "checkpoint_evaluation_report.json").unlink()

    completed = _run_checkpoint_evaluator(checkpoint, output, resume=True)

    assert "wrote" in completed.stdout
    assert (output / "checkpoint_evaluation_report.json").is_file()


def test_checkpoint_evaluator_resume_rejects_request_drift(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "evaluation"
    _write_cpu_checkpoint(checkpoint)
    _run_checkpoint_evaluator(checkpoint, output)
    report_path = output / "checkpoint_evaluation_report.json"
    original_bytes = report_path.read_bytes()

    resumed = _run_checkpoint_evaluator(
        checkpoint,
        output,
        resume=True,
        timestep=2,
        check=False,
    )

    assert resumed.returncode != 0
    assert "resume manifest does not match the request" in resumed.stderr
    assert report_path.read_bytes() == original_bytes


def test_checkpoint_evaluator_resume_rejects_inconsistent_report(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "evaluation"
    _write_cpu_checkpoint(checkpoint)
    _run_checkpoint_evaluator(checkpoint, output)
    report_path = output / "checkpoint_evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metrics"]["evaluated_images"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    resumed = _run_checkpoint_evaluator(checkpoint, output, resume=True, check=False)

    assert resumed.returncode != 0
    assert "evidence is inconsistent" in resumed.stderr


def test_checkpoint_evaluator_requires_resume_for_existing_evidence(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "evaluation"
    _write_cpu_checkpoint(checkpoint)
    _run_checkpoint_evaluator(checkpoint, output)

    repeated = _run_checkpoint_evaluator(checkpoint, output, check=False)

    assert repeated.returncode != 0
    assert "pass --resume to validate it" in repeated.stderr
