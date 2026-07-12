from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from cofitok.configs import ExperimentConfig, config_from_dict
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256
from cofitok.training import ExponentialMovingAverage


@dataclass
class LoadedGenerationModel:
    model: nn.Module
    config: ExperimentConfig
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
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
    checkpoint_sha256 = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
        checkpoint_step=checkpoint_step,
        weights=weights,
    )
