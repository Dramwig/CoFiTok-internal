from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cofitok.training.ema import ExponentialMovingAverage


CHECKPOINT_FORMAT_VERSION = 1


def unwrap_model(model: nn.Module) -> nn.Module:
    unwrapped = model
    while True:
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        return unwrapped


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    step: int,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "config": dict(config),
        "model": unwrap_model(model).state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "metrics": dict(metrics or {}),
        "extra_state": dict(extra_state or {}),
    }
    temporary = target.with_suffix(f"{target.suffix}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    latest = target.parent / "latest.json"
    latest_temporary = latest.with_suffix(f".tmp-{os.getpid()}")
    latest_temporary.write_text(
        json.dumps({"checkpoint": target.name, "step": step}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(latest_temporary, latest)
    return target


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    if ema is not None:
        ema.load_state_dict(checkpoint["ema"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def prune_checkpoints(directory: str | Path, keep_last: int) -> list[Path]:
    if keep_last < 1:
        raise ValueError("keep_last must be positive")
    paths = sorted(Path(directory).glob("checkpoint_step_*.pt"))
    removed = paths[:-keep_last]
    for path in removed:
        path.unlink()
    return removed
