from __future__ import annotations

import json

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
from scripts.preflight_generation_sampling import run_sampling_preflight


SOURCE_ENVIRONMENT_SHA = "e" * 64
SOURCE_GIT = {
    "revision": "a" * 40,
    "branch": "scale/generative-system",
    "dirty": False,
}


def _training_authorization(tmp_path) -> dict:
    return {
        "schema_version": 1,
        "status": "pass",
        "stage": "scaling",
        "decision": "promote_to_full_imagenet256",
        "gate_path": (tmp_path / "promotion_gate.json").resolve().as_posix(),
        "gate_bytes": 12_345,
        "gate_sha256": "b" * 64,
        "gate_identity_sha256": "c" * 64,
        "validated_thresholds": {
            "min_samples": 10_000.0,
            "max_fid_regression": 0.05,
            "max_absolute_fid": 100.0,
            "max_endpoint_regression": 0.05,
        },
    }


def _training_checkpoint(
    tmp_path,
    *,
    include_provenance: bool = True,
    include_authorization: bool = False,
):
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
    payload = {
        "format_version": 1,
        "config": config_to_dict(config),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "step": 31,
    }
    if include_provenance:
        payload["extra_state"] = {
            "runtime_environment_sha256": SOURCE_ENVIRONMENT_SHA,
            "git": SOURCE_GIT,
        }
        if include_authorization:
            payload["extra_state"]["training_authorization"] = (
                _training_authorization(tmp_path)
            )
    torch.save(payload, path)
    integrity = {
        "schema_version": 1,
        "checkpoint": path.name,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_format_version": 1,
        "step": 31,
    }
    if include_provenance:
        integrity.update(
            runtime_environment_sha256=SOURCE_ENVIRONMENT_SHA,
            git_revision=SOURCE_GIT["revision"],
            git_branch=SOURCE_GIT["branch"],
            git_dirty=SOURCE_GIT["dirty"],
        )
        if include_authorization:
            authorization = _training_authorization(tmp_path)
            integrity.update(
                authorization_stage=authorization["stage"],
                authorization_decision=authorization["decision"],
                authorization_gate_bytes=authorization["gate_bytes"],
                authorization_gate_sha256=authorization["gate_sha256"],
                authorization_gate_identity_sha256=authorization[
                    "gate_identity_sha256"
                ],
            )
    write_json_report(
        checkpoint_integrity_path(path),
        integrity,
    )
    return path


def test_ema_export_is_smaller_verified_and_sample_equivalent(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"

    report = export_ema_inference_artifact(source, artifact)
    reused = export_ema_inference_artifact(source, artifact)

    assert report["status"] == "completed"
    assert report["schema_version"] == 3
    assert report["weights"] == "ema_export"
    assert report["verified"] is True
    assert report["artifact_bytes"] < report["source_checkpoint_bytes"]
    assert reused["reused"] is True
    verified = verify_inference_artifact(artifact)
    assert verified["schema_version"] == 3
    assert verified["artifact_format_version"] == 3
    assert verified["artifact_sha256"] == report["artifact_sha256"]
    assert report["source_runtime_environment_sha256"] == SOURCE_ENVIRONMENT_SHA
    assert report["source_git"] == SOURCE_GIT
    assert report["source_training_authorization"] == _training_authorization(
        tmp_path
    )
    assert verified["source_training_authorization"] == _training_authorization(
        tmp_path
    )

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
    assert (
        export_result.metadata["source_runtime_environment_sha256"]
        == SOURCE_ENVIRONMENT_SHA
    )
    assert export_result.metadata["source_git"] == SOURCE_GIT
    assert export_result.metadata["training_authorization"] == (
        _training_authorization(tmp_path)
    )
    preflight = run_sampling_preflight(
        artifact,
        batch_size=1,
        prefix_budget=2,
        guidance_scale=1.0,
        precision="fp32",
    )
    assert preflight["status"] == "passed"
    assert preflight["training_authorization"] == _training_authorization(
        tmp_path
    )


def test_ema_export_rejects_source_without_deployment_provenance(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_provenance=False)

    with pytest.raises(ValueError, match="runtime environment provenance"):
        export_ema_inference_artifact(
            source,
            tmp_path / "unprovenanced_inference.pt",
        )


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


def test_inference_artifact_rejects_source_sidecar_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    integrity_path = checkpoint_integrity_path(artifact)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["source_git_revision"] = "b" * 40
    write_json_report(integrity_path, integrity)

    with pytest.raises(ValueError, match="source Git provenance mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_inference_artifact_rejects_training_authorization_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    integrity_path = checkpoint_integrity_path(artifact)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["source_training_authorization"][
        "gate_identity_sha256"
    ] = "d" * 64
    write_json_report(integrity_path, integrity)

    with pytest.raises(ValueError, match="training authorization mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_export_reuse_rejects_source_authorization_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    source_integrity_path = checkpoint_integrity_path(source)
    source_integrity = json.loads(
        source_integrity_path.read_text(encoding="utf-8")
    )
    source_integrity["authorization_gate_identity_sha256"] = "d" * 64
    write_json_report(source_integrity_path, source_integrity)

    with pytest.raises(ValueError, match="authorization differs"):
        export_ema_inference_artifact(source, artifact)
