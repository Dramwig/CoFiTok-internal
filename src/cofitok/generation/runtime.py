from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from cofitok.configs import ExperimentConfig, config_from_dict
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


def load_generation_model(
    checkpoint_path: str | Path,
    *,
    weights: str = "ema",
) -> LoadedGenerationModel:
    """Load the exact model path shared by formal sampling and its preflight."""
    if weights not in {"ema", "model"}:
        raise ValueError("weights must be ema or model")

    path = Path(checkpoint_path)
    integrity = verify_training_checkpoint(path)
    checkpoint_sha256 = str(integrity["checkpoint_sha256"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("format_version", -1)) != int(
        integrity["checkpoint_format_version"]
    ):
        raise ValueError("Checkpoint payload format does not match integrity metadata")
    if int(checkpoint.get("step", -1)) != int(integrity["step"]):
        raise ValueError("Checkpoint payload step does not match integrity metadata")
    config = config_from_dict(checkpoint["config"])
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint sampling requested but CUDA is unavailable")

    model = CoFiTokTiny(config.model)
    model.load_state_dict(checkpoint["model"], strict=True)
    if weights == "ema":
        if "ema" not in checkpoint:
            raise KeyError("EMA weights are missing from the checkpoint")
        ema = ExponentialMovingAverage(
            model,
            decay=config.optimization.ema_decay,
            warmup_steps=config.optimization.ema_warmup_steps,
        )
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)

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
        weights=weights,
    )
