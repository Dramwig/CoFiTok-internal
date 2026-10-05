from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import scripts.infer_generation as inference_cli
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
from cofitok.inference_replay import (
    INFERENCE_REPORT_SCHEMA_VERSION,
    validate_completed_inference_evidence,
)
from cofitok.models import CoFiTokTiny
from cofitok.output_lock import OutputLockError, exclusive_output_lock
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
    monkeypatch,
) -> None:
    checkpoint = _checkpoint(tmp_path)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("training checkpoint was deserialized before policy rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="release-authorized inference artifact"):
        GenerationSession.from_checkpoint(
            checkpoint,
            weights="ema",
            require_release_authorization=True,
        )


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows symlink creation requires elevated privileges",
)
def test_generation_loader_rejects_symlinked_checkpoint_and_sidecar(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    checkpoint_alias = tmp_path / "checkpoint_alias.pt"
    checkpoint_alias.symlink_to(checkpoint)

    with pytest.raises(ValueError, match="must not contain a symlink"):
        GenerationSession.from_checkpoint(checkpoint_alias, weights="ema")

    integrity_path = checkpoint_integrity_path(checkpoint)
    integrity_target = tmp_path / "integrity_target.json"
    integrity_target.write_bytes(integrity_path.read_bytes())
    integrity_path.unlink()
    integrity_path.symlink_to(integrity_target)

    with pytest.raises(ValueError, match="must not contain a symlink"):
        GenerationSession.from_checkpoint(checkpoint, weights="ema")


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
    assert report["schema_version"] == INFERENCE_REPORT_SCHEMA_VERSION
    assert report["output_count"] == 4
    assert report["checkpoint"]["checkpoint_step"] == 23
    assert report["inference_api"] == INFERENCE_API
    assert report["sampling_protocol_schema"] == "cofitok_ddim_sampling_v1"
    assert report["git"]["tracked_dirty"] is not None
    assert report["runtime_environment"]["schema_version"] == 1
    assert len(report["runtime_environment_sha256"]) == 64
    assert report["request"]["seeds"] == [7, 9]
    assert report["request"]["class_ids"] == [3, 3]
    assert report["request"]["prefix_budgets"] == [1, 2]
    assert (output_dir / "inference_report.json").is_file()
    assert (output_dir / "inference_manifest.json").is_file()
    assert (output_dir / "inference_progress.json").is_file()
    progress = json.loads(
        (output_dir / "inference_progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "completed"
    assert progress["completed_output_count"] == 4
    replay = validate_completed_inference_evidence(
        report,
        expected_root=output_dir,
    )
    assert replay["status"] == "verified"
    assert replay["output_count"] == 4
    for output in report["outputs"]:
        assert len(output["sha256"]) == 64
        assert (output_dir / output["filename"]).is_file()


def _inference_args(checkpoint: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(checkpoint),
        output_dir=str(output_dir),
        report="",
        class_ids="3",
        seeds="7,9",
        seed=0,
        num_images=1,
        prefix_budgets="1,2",
        batch_size=1,
        sample_steps=1,
        guidance_scale=1.0,
        guidance_rescale=0.0,
        cfg_batch_mode="batched",
        eta=0.0,
        weights="ema",
        precision="fp32",
        require_release_authorization=False,
        resume=False,
        overwrite=False,
    )


def test_inference_cli_locks_output_before_loading_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "inference"
    args = _inference_args(tmp_path / "not_loaded.pt", output_dir)
    monkeypatch.setattr(
        inference_cli.GenerationSession,
        "from_checkpoint",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded before output lock"),
    )

    with exclusive_output_lock(output_dir, role="test_holder"):
        with pytest.raises(OutputLockError, match="already locked"):
            run_inference(args)
    assert not output_dir.exists()


def test_inference_cli_resume_reuses_completed_outputs_after_failure(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "recoverable_inference"
    args = _inference_args(checkpoint, output_dir)
    original_save = inference_cli.save_tensor_png
    save_calls = 0

    def interrupt_second_save(image, path, *, overwrite=False):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise RuntimeError("simulated inference interruption")
        return original_save(image, path, overwrite=overwrite)

    monkeypatch.setattr(inference_cli, "save_tensor_png", interrupt_second_save)
    with pytest.raises(RuntimeError, match="simulated inference interruption"):
        run_inference(args)

    failed_progress = json.loads(
        (output_dir / "inference_progress.json").read_text(encoding="utf-8")
    )
    assert failed_progress["status"] == "failed"
    assert failed_progress["completed_output_count"] == 1
    retained = Path(failed_progress["outputs"][0]["path"])
    retained_sha = file_sha256(retained)
    retained_mtime = retained.stat().st_mtime_ns

    monkeypatch.setattr(inference_cli, "save_tensor_png", original_save)
    args.resume = True
    completed = run_inference(args)

    assert completed["status"] == "completed"
    assert completed["attempt_count"] == 2
    assert completed["output_count"] == 4
    assert file_sha256(retained) == retained_sha
    assert retained.stat().st_mtime_ns == retained_mtime


def test_inference_cli_completed_resume_is_read_only(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "completed_inference"
    args = _inference_args(checkpoint, output_dir)
    first = run_inference(args)
    evidence = [
        output_dir / "inference_manifest.json",
        output_dir / "inference_progress.json",
        output_dir / "inference_report.json",
        *[Path(row["path"]) for row in first["outputs"]],
    ]
    identities = {
        path.as_posix(): (file_sha256(path), path.stat().st_mtime_ns)
        for path in evidence
    }

    args.resume = True
    reused = run_inference(args)

    assert reused["status"] == "completed"
    assert reused["reused"] is True
    assert {
        path.as_posix(): (file_sha256(path), path.stat().st_mtime_ns)
        for path in evidence
    } == identities


def test_inference_cli_resume_rejects_request_drift(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "drifted_inference"
    args = _inference_args(checkpoint, output_dir)
    first = run_inference(args)
    output = Path(first["outputs"][0]["path"])
    identity = (file_sha256(output), output.stat().st_mtime_ns)

    args.resume = True
    args.sample_steps = 2
    with pytest.raises(ValueError, match="manifest does not match"):
        run_inference(args)

    assert (file_sha256(output), output.stat().st_mtime_ns) == identity


def test_inference_cli_resume_regenerates_only_corrupt_output(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "corrupt_inference"
    args = _inference_args(checkpoint, output_dir)
    first = run_inference(args)
    target = Path(first["outputs"][0]["path"])
    expected_sha = first["outputs"][0]["sha256"]
    untouched = Path(first["outputs"][1]["path"])
    untouched_identity = (file_sha256(untouched), untouched.stat().st_mtime_ns)
    target.write_bytes(b"corrupt")

    args.resume = True
    recovered = run_inference(args)

    assert recovered["status"] == "completed"
    assert recovered["attempt_count"] == 2
    assert file_sha256(target) == expected_sha
    assert (file_sha256(untouched), untouched.stat().st_mtime_ns) == untouched_identity


def test_inference_cli_completed_resume_rejects_progress_tamper(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "tampered_progress"
    args = _inference_args(checkpoint, output_dir)
    run_inference(args)
    progress_path = output_dir / "inference_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["cumulative_elapsed_seconds"] += 1.0
    write_json_report(progress_path, progress)

    args.resume = True
    with pytest.raises(ValueError, match="progress identity differs"):
        run_inference(args)


def test_inference_cli_rejects_unexpected_png_before_control_writes(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "mixed_inference"
    output_dir.mkdir()
    unexpected = output_dir / "leftover_from_another_request.png"
    unexpected.write_bytes(b"owned-existing-output")
    identity = (file_sha256(unexpected), unexpected.stat().st_mtime_ns)
    args = _inference_args(checkpoint, output_dir)
    args.overwrite = True

    with pytest.raises(ValueError, match="unexpected PNG files"):
        run_inference(args)

    assert (file_sha256(unexpected), unexpected.stat().st_mtime_ns) == identity
    assert not (output_dir / "inference_manifest.json").exists()
    assert not (output_dir / "inference_progress.json").exists()
    assert not (output_dir / "inference_report.json").exists()


def test_inference_cli_resume_rejects_missing_progress_evidence(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "missing_progress"
    args = _inference_args(checkpoint, output_dir)
    run_inference(args)
    report_path = output_dir / "inference_report.json"
    report_identity = (file_sha256(report_path), report_path.stat().st_mtime_ns)
    (output_dir / "inference_progress.json").unlink()

    args.resume = True
    with pytest.raises(ValueError, match="missing its progress evidence"):
        run_inference(args)

    assert (file_sha256(report_path), report_path.stat().st_mtime_ns) == report_identity
    assert not (output_dir / "inference_progress.json").exists()


def test_completed_inference_evidence_rehashes_physical_pngs(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "tampered_output"
    args = _inference_args(checkpoint, output_dir)
    report = run_inference(args)
    Path(report["outputs"][0]["path"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="digest differs from physical PNG"):
        validate_completed_inference_evidence(report, expected_root=output_dir)


def test_formal_sampling_cli_runs_checkpoint_to_png_and_report(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output_dir = tmp_path / "formal_samples"
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(ROOT / "src"), inherited_pythonpath) if path
    )

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
