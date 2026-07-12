from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from cofitok.configs import ExperimentConfig, config_from_dict
from cofitok.generation.artifact import (
    INFERENCE_ARTIFACT_FORMAT_VERSION,
    INFERENCE_ARTIFACT_TYPE,
    verify_inference_artifact,
)
from cofitok.models import CoFiTokTiny
from cofitok.training import ExponentialMovingAverage
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


def load_generation_model(
    checkpoint_path: str | Path,
    *,
    weights: str = "ema",
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
    else:
        effective_weights = "model"
        source_checkpoint_sha256 = None

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
    )
