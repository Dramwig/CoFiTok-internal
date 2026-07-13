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

from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.ema import ExponentialMovingAverage


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_INTEGRITY_VERSION = 1


def _config_mismatch_paths(
    expected: Any,
    actual: Any,
    *,
    path: str = "config",
) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        mismatches = []
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}"
            if key not in expected or key not in actual:
                mismatches.append(child)
                continue
            mismatches.extend(
                _config_mismatch_paths(expected[key], actual[key], path=child)
            )
        return mismatches
    return [] if expected == actual else [path]


def checkpoint_integrity_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint.with_name(f"{checkpoint.name}.integrity.json")


def verify_training_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    integrity_path = checkpoint_integrity_path(checkpoint)
    if not integrity_path.is_file():
        raise FileNotFoundError(f"Checkpoint integrity manifest is missing: {integrity_path}")
    with integrity_path.open("r", encoding="utf-8") as handle:
        integrity = json.load(handle)
    if integrity.get("schema_version") != CHECKPOINT_INTEGRITY_VERSION:
        raise ValueError("Unsupported checkpoint integrity manifest version")
    if integrity.get("checkpoint") != checkpoint.name:
        raise ValueError("Checkpoint integrity manifest names another file")
    expected_bytes = int(integrity.get("checkpoint_bytes", -1))
    actual_bytes = checkpoint.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Checkpoint size mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    expected_sha256 = str(integrity.get("checkpoint_sha256", ""))
    if len(expected_sha256) != 64:
        raise ValueError("Checkpoint integrity SHA256 is malformed")
    actual_sha256 = file_sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    return integrity


def backfill_training_checkpoint_integrity(
    path: str | Path,
    *,
    update_latest: bool = False,
) -> dict[str, Any]:
    """Add integrity metadata to a legacy atomic checkpoint without changing its bytes."""
    checkpoint_path = Path(path)
    integrity_path = checkpoint_integrity_path(checkpoint_path)
    if integrity_path.is_file():
        integrity = verify_training_checkpoint(checkpoint_path)
    else:
        checkpoint_sha256 = file_sha256(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {
            "format_version",
            "step",
            "config",
            "model",
            "ema",
            "optimizer",
            "scheduler",
            "rng_state",
            "extra_state",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Legacy checkpoint is missing exact-resume state: {missing}")
        if int(payload["format_version"]) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint format: {payload['format_version']}")
        sampler = payload["extra_state"].get("sampler")
        if sampler is None:
            raise ValueError("Legacy production checkpoint is missing sampler state")
        integrity = {
            "schema_version": CHECKPOINT_INTEGRITY_VERSION,
            "checkpoint": checkpoint_path.name,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_format_version": int(payload["format_version"]),
            "step": int(payload["step"]),
        }
        del payload
        write_json_report(integrity_path, integrity)
        verify_training_checkpoint(checkpoint_path)

    if update_latest:
        latest_path = checkpoint_path.parent / "latest.json"
        if latest_path.is_file():
            with latest_path.open("r", encoding="utf-8") as handle:
                previous_latest = json.load(handle)
            if previous_latest.get("checkpoint") != checkpoint_path.name:
                raise ValueError("Refusing to repoint latest.json to a non-latest checkpoint")
            if int(previous_latest.get("step", -1)) != int(integrity["step"]):
                raise ValueError("Legacy latest.json step does not match checkpoint payload")
        write_json_report(
            latest_path,
            {**integrity, "integrity_manifest": integrity_path.name},
        )
    return integrity


def resolve_latest_checkpoint(directory: str | Path) -> Path:
    root = Path(directory)
    latest_path = root / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"No automatic resume pointer at {latest_path}")
    with latest_path.open("r", encoding="utf-8") as handle:
        latest = json.load(handle)
    checkpoint_name = str(latest.get("checkpoint", ""))
    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("latest.json checkpoint must be a local filename")
    checkpoint = root / checkpoint_name
    integrity = verify_training_checkpoint(checkpoint)
    expected_pointer = {
        "step": integrity["step"],
        "checkpoint_bytes": integrity["checkpoint_bytes"],
        "checkpoint_sha256": integrity["checkpoint_sha256"],
        "integrity_manifest": checkpoint_integrity_path(checkpoint).name,
    }
    for key, expected in expected_pointer.items():
        if latest.get(key) != expected:
            raise ValueError(f"latest.json {key} does not match checkpoint integrity metadata")
    return checkpoint


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
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    integrity = {
        "schema_version": CHECKPOINT_INTEGRITY_VERSION,
        "checkpoint": target.name,
        "checkpoint_bytes": target.stat().st_size,
        "checkpoint_sha256": file_sha256(target),
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
    }
    integrity_path = checkpoint_integrity_path(target)
    write_json_report(integrity_path, integrity)
    write_json_report(
        target.parent / "latest.json",
        {**integrity, "integrity_manifest": integrity_path.name},
    )
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
    verify_integrity: bool = True,
    expected_config: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    integrity = None
    if verify_integrity:
        integrity = verify_training_checkpoint(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    if expected_config is not None:
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, Mapping):
            raise ValueError("Checkpoint is missing its exact-resume config")
        mismatches = _config_mismatch_paths(expected_config, checkpoint_config)
        if mismatches:
            preview = ", ".join(mismatches[:8])
            if len(mismatches) > 8:
                preview += f", ... ({len(mismatches)} fields)"
            raise ValueError(f"Checkpoint config mismatch at: {preview}")
    if integrity is not None:
        if int(checkpoint.get("step", -1)) != int(integrity["step"]):
            raise ValueError("Checkpoint payload step does not match integrity metadata")
        if int(checkpoint.get("format_version", -1)) != int(
            integrity["checkpoint_format_version"]
        ):
            raise ValueError("Checkpoint payload format does not match integrity metadata")
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


def prune_checkpoints(
    directory: str | Path,
    keep_last: int,
    *,
    protected_steps: list[int] | tuple[int, ...] = (),
) -> list[Path]:
    if keep_last < 1:
        raise ValueError("keep_last must be positive")
    if any(step < 1 for step in protected_steps):
        raise ValueError("protected checkpoint steps must be positive")
    paths = sorted(Path(directory).glob("checkpoint_step_*.pt"))
    protected = {int(step) for step in protected_steps}
    recent = set(paths[-keep_last:])
    retained = recent | {
        path
        for path in paths
        if int(path.stem.removeprefix("checkpoint_step_")) in protected
    }
    removed = [path for path in paths if path not in retained]
    for path in removed:
        path.unlink()
        checkpoint_integrity_path(path).unlink(missing_ok=True)
    return removed
