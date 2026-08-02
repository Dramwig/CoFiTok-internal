from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cofitok.configs import ExperimentConfig, config_from_dict
from cofitok.generation.artifact import (
    INFERENCE_ARTIFACT_FORMAT_VERSION,
    INFERENCE_ARTIFACT_TYPE,
    verify_inference_artifact,
)
from cofitok.generation.release import verify_generation_release_receipt
from cofitok.generation_authorization import validate_generation_gate_binding
from cofitok.models import CoFiTokTiny
from cofitok.training import ExponentialMovingAverage
from cofitok.training.authorization import (
    validate_checkpoint_training_authorization,
    validate_generation_training_authorization,
)
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


@dataclass
class LoadedGenerationModel:
    model: nn.Module
    config: ExperimentConfig
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_integrity_manifest: Path
    checkpoint_step: int
    weights: str
    artifact_type: str
    source_checkpoint_sha256: str | None
    source_runtime_environment_sha256: str | None
    source_git_provenance: dict[str, Any] | None
    training_authorization: dict[str, Any] | None
    release_authorization: dict[str, Any] | None
    release_authorization_required: bool
    completion_authorization: dict[str, Any] | None
    completion_authorization_required: bool


def load_generation_model(
    checkpoint_path: str | Path,
    *,
    weights: str = "ema",
    require_release_authorization: bool = False,
    completion_receipt: str | Path | None = None,
    require_completion_authorization: bool = False,
) -> LoadedGenerationModel:
    """Load the exact model path shared by formal sampling and its preflight."""
    if weights not in {"ema", "model"}:
        raise ValueError("weights must be ema or model")

    path = Path(checkpoint_path)
    integrity_path = checkpoint_integrity_path(path)
    with integrity_path.open("r", encoding="utf-8") as handle:
        integrity_hint = json.load(handle)
    is_inference_artifact = (
        integrity_hint.get("artifact_type") == INFERENCE_ARTIFACT_TYPE
    )
    integrity = (
        verify_inference_artifact(path)
        if is_inference_artifact
        else verify_training_checkpoint(path)
    )
    effective_release_requirement = (
        require_release_authorization or require_completion_authorization
    )
    if require_completion_authorization and not completion_receipt:
        raise ValueError(
            "Production inference completion authorization requires a release receipt"
        )
    if completion_receipt and not is_inference_artifact:
        raise ValueError("A generation release receipt only authorizes inference artifacts")
    if effective_release_requirement and (
        not is_inference_artifact
        or integrity.get("release_authorization") is None
    ):
        raise ValueError(
            "Production inference requires a release-authorized inference artifact"
        )
    completion_authorization = (
        verify_generation_release_receipt(
            completion_receipt,
            path,
            artifact_integrity=integrity,
        )
        if completion_receipt
        else None
    )
    checkpoint_sha256 = str(
        integrity["artifact_sha256"]
        if is_inference_artifact
        else integrity["checkpoint_sha256"]
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_format = (
        INFERENCE_ARTIFACT_FORMAT_VERSION
        if is_inference_artifact
        else int(integrity["checkpoint_format_version"])
    )
    if int(checkpoint.get("format_version", -1)) != expected_format:
        raise ValueError("Checkpoint payload format does not match integrity metadata")
    if int(checkpoint.get("step", -1)) != int(integrity["step"]):
        raise ValueError("Checkpoint payload step does not match integrity metadata")
    training_authorization = None
    release_authorization = None
    if not is_inference_artifact:
        training_authorization = validate_checkpoint_training_authorization(
            checkpoint,
            integrity,
        )
    config = config_from_dict(checkpoint["config"])
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint sampling requested but CUDA is unavailable")

    model = CoFiTokTiny(config.model)
    model.load_state_dict(checkpoint["model"], strict=True)
    if is_inference_artifact:
        if weights != "ema" or checkpoint.get("weights") != "ema_export":
            raise ValueError("Inference artifact exposes only exported EMA weights")
        effective_weights = "ema_export"
        source_checkpoint_sha256 = str(checkpoint["source"]["checkpoint_sha256"])
        if source_checkpoint_sha256 != integrity["source_checkpoint_sha256"]:
            raise ValueError("Inference artifact source provenance mismatch")
        source_runtime_environment_sha256 = str(
            checkpoint["source"].get("runtime_environment_sha256", "")
        )
        if (
            source_runtime_environment_sha256
            != integrity["source_runtime_environment_sha256"]
        ):
            raise ValueError("Inference artifact source environment mismatch")
        source_git = checkpoint["source"].get("git")
        if not isinstance(source_git, Mapping):
            raise ValueError("Inference artifact source Git provenance is missing")
        source_git_provenance = dict(source_git)
        integrity_git = {
            "revision": integrity["source_git_revision"],
            "branch": integrity["source_git_branch"],
            "dirty": integrity["source_git_dirty"],
        }
        if source_git_provenance != integrity_git:
            raise ValueError("Inference artifact source Git provenance mismatch")
        payload_authorization = checkpoint["source"].get(
            "training_authorization"
        )
        integrity_authorization = integrity.get("source_training_authorization")
        if payload_authorization != integrity_authorization:
            raise ValueError(
                "Inference artifact source training authorization mismatch"
            )
        if payload_authorization is not None:
            if not isinstance(payload_authorization, Mapping):
                raise ValueError(
                    "Inference artifact source training authorization is malformed"
                )
            validate_generation_training_authorization(payload_authorization)
            training_authorization = dict(payload_authorization)
        payload_release = checkpoint.get("release_authorization")
        integrity_release = integrity.get("release_authorization")
        if payload_release != integrity_release:
            raise ValueError(
                "Inference artifact release authorization mismatch"
            )
        if payload_release is not None:
            if not isinstance(payload_release, Mapping):
                raise ValueError(
                    "Inference artifact release authorization is malformed"
                )
            validate_generation_gate_binding(
                payload_release,
                expected_stage="full",
            )
            release_authorization = dict(payload_release)
    elif weights == "ema":
        if "ema" not in checkpoint:
            raise KeyError("EMA weights are missing from the checkpoint")
        ema = ExponentialMovingAverage(
            model,
            decay=config.optimization.ema_decay,
            warmup_steps=config.optimization.ema_warmup_steps,
        )
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
        effective_weights = "ema"
        source_checkpoint_sha256 = None
        source_runtime_environment_sha256 = None
        source_git_provenance = None
    else:
        effective_weights = "model"
        source_checkpoint_sha256 = None
        source_runtime_environment_sha256 = None
        source_git_provenance = None

    checkpoint_step = int(checkpoint["step"])
    del checkpoint
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32

    return LoadedGenerationModel(
        model=model,
        config=config,
        device=device,
        checkpoint_path=path.resolve(),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_integrity_manifest=checkpoint_integrity_path(path).resolve(),
        checkpoint_step=checkpoint_step,
        weights=effective_weights,
        artifact_type=(INFERENCE_ARTIFACT_TYPE if is_inference_artifact else "training_checkpoint"),
        source_checkpoint_sha256=source_checkpoint_sha256,
        source_runtime_environment_sha256=source_runtime_environment_sha256,
        source_git_provenance=source_git_provenance,
        training_authorization=training_authorization,
        release_authorization=release_authorization,
        release_authorization_required=effective_release_requirement,
        completion_authorization=completion_authorization,
        completion_authorization_required=require_completion_authorization,
    )
