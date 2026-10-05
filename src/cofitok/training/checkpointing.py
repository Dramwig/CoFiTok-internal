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

from cofitok.data.provenance import validate_dataset_provenance
from cofitok.environment import (
    runtime_environment_mismatch_paths,
    runtime_environment_sha256,
)
from cofitok.path_security import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.authorization import (
    validate_checkpoint_training_authorization,
    validate_generation_training_authorization,
)
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


def _validate_git_provenance(
    provenance: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    required = {"revision", "branch", "dirty"}
    if set(provenance) != required:
        raise ValueError(f"{label} must contain exactly: {sorted(required)}")
    resolved = dict(provenance)
    if (
        not isinstance(resolved["revision"], str)
        or not resolved["revision"]
        or not isinstance(resolved["branch"], str)
        or not isinstance(resolved["dirty"], bool)
    ):
        raise ValueError(f"{label} is malformed")
    return resolved


def checkpoint_integrity_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint.with_name(f"{checkpoint.name}.integrity.json")


def verify_training_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = reject_symlink_chain(path, name="training checkpoint")
    integrity_path = reject_symlink_chain(
        checkpoint_integrity_path(checkpoint),
        name="checkpoint integrity manifest",
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training checkpoint is missing: {checkpoint}")
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
    runtime_environment_sha = integrity.get("runtime_environment_sha256")
    if runtime_environment_sha is not None and len(str(runtime_environment_sha)) != 64:
        raise ValueError("Checkpoint runtime environment SHA256 is malformed")
    dataset_identity_sha = integrity.get("dataset_identity_sha256")
    if dataset_identity_sha is not None and len(str(dataset_identity_sha)) != 64:
        raise ValueError("Checkpoint dataset identity SHA256 is malformed")
    git_keys = {"git_revision", "git_branch", "git_dirty"}
    present_git_keys = git_keys & integrity.keys()
    if present_git_keys and present_git_keys != git_keys:
        raise ValueError("Checkpoint Git integrity metadata is incomplete")
    if present_git_keys and (
        not isinstance(integrity["git_revision"], str)
        or not integrity["git_revision"]
        or not isinstance(integrity["git_branch"], str)
        or not isinstance(integrity["git_dirty"], bool)
    ):
        raise ValueError("Checkpoint Git integrity metadata is malformed")
    authorization_keys = {
        "authorization_stage",
        "authorization_decision",
        "authorization_gate_bytes",
        "authorization_gate_sha256",
        "authorization_gate_identity_sha256",
    }
    present_authorization_keys = authorization_keys & integrity.keys()
    if present_authorization_keys and present_authorization_keys != authorization_keys:
        raise ValueError("Checkpoint training-authorization metadata is incomplete")
    if present_authorization_keys and (
        integrity["authorization_stage"] != "scaling"
        or integrity["authorization_decision"] != "promote_to_full_imagenet256"
        or int(integrity["authorization_gate_bytes"]) < 1
        or len(str(integrity["authorization_gate_sha256"])) != 64
        or len(str(integrity["authorization_gate_identity_sha256"])) != 64
    ):
        raise ValueError("Checkpoint training-authorization metadata is malformed")
    return integrity


def backfill_training_checkpoint_integrity(
    path: str | Path,
    *,
    update_latest: bool = False,
) -> dict[str, Any]:
    """Add integrity metadata to a legacy atomic checkpoint without changing its bytes."""
    checkpoint_path = reject_symlink_chain(path, name="training checkpoint")
    integrity_path = reject_symlink_chain(
        checkpoint_integrity_path(checkpoint_path),
        name="checkpoint integrity manifest",
    )
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
        latest_path = reject_symlink_chain(
            checkpoint_path.parent / "latest.json",
            name="checkpoint latest pointer",
        )
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
    root = reject_symlink_chain(directory, name="checkpoint directory")
    latest_path = reject_symlink_chain(
        root / "latest.json",
        name="checkpoint latest pointer",
    )
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
    if "runtime_environment_sha256" in integrity:
        expected_pointer["runtime_environment_sha256"] = integrity[
            "runtime_environment_sha256"
        ]
    for key in ("git_revision", "git_branch", "git_dirty"):
        if key in integrity:
            expected_pointer[key] = integrity[key]
    for key in (
        "authorization_stage",
        "authorization_decision",
        "authorization_gate_bytes",
        "authorization_gate_sha256",
        "authorization_gate_identity_sha256",
    ):
        if key in integrity:
            expected_pointer[key] = integrity[key]
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
    torch_cpu = state["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8:
        raise TypeError("CPU RNG state must be a torch.uint8 tensor")
    # A CUDA map_location also remaps the serialized CPU RNG tensor. PyTorch's
    # CPU and CUDA generators both require their state tensors on the CPU.
    torch.set_rng_state(torch_cpu.detach().cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch_cuda = state["torch_cuda"]
        if not isinstance(torch_cuda, (list, tuple)) or any(
            not isinstance(item, torch.Tensor) or item.dtype != torch.uint8
            for item in torch_cuda
        ):
            raise TypeError("CUDA RNG states must be torch.uint8 tensors")
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in torch_cuda])


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
    target = reject_symlink_chain(path, name="training checkpoint")
    integrity_path = reject_symlink_chain(
        checkpoint_integrity_path(target),
        name="checkpoint integrity manifest",
    )
    latest_path = reject_symlink_chain(
        target.parent / "latest.json",
        name="checkpoint latest pointer",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_extra_state = dict(extra_state or {})
    git_provenance = resolved_extra_state.get("git")
    if git_provenance is not None and not isinstance(git_provenance, Mapping):
        raise ValueError("Git checkpoint state must be a mapping")
    if git_provenance is not None:
        git_provenance = _validate_git_provenance(
            git_provenance,
            label="Git checkpoint state",
        )
        resolved_extra_state["git"] = git_provenance
    runtime_environment = resolved_extra_state.get("runtime_environment")
    runtime_environment_sha = None
    if runtime_environment is not None:
        if not isinstance(runtime_environment, Mapping):
            raise ValueError("runtime environment checkpoint state must be a mapping")
        runtime_environment_sha = runtime_environment_sha256(runtime_environment)
        declared_sha = resolved_extra_state.get("runtime_environment_sha256")
        if declared_sha is not None and declared_sha != runtime_environment_sha:
            raise ValueError("runtime environment SHA256 differs from checkpoint state")
        resolved_extra_state["runtime_environment_sha256"] = runtime_environment_sha
    dataset_provenance = resolved_extra_state.get("dataset_provenance")
    dataset_identity_sha = None
    if dataset_provenance is not None:
        if not isinstance(dataset_provenance, Mapping):
            raise ValueError("dataset checkpoint provenance must be a mapping")
        if dataset_provenance.get("formal") is True:
            config_data = config.get("data")
            if not isinstance(config_data, Mapping) or not config_data.get("dataset"):
                raise ValueError("formal dataset checkpoint lacks a config dataset alias")
            dataset_identity_sha = validate_dataset_provenance(
                dataset_provenance,
                expected_dataset=str(config_data["dataset"]),
            )["identity_sha256"]
    training_authorization = resolved_extra_state.get("training_authorization")
    authorization_evidence = None
    if training_authorization is not None:
        if not isinstance(training_authorization, Mapping):
            raise ValueError("generation training authorization must be a mapping")
        authorization_evidence = validate_generation_training_authorization(
            training_authorization
        )
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
        "extra_state": resolved_extra_state,
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
    if runtime_environment_sha is not None:
        integrity["runtime_environment_sha256"] = runtime_environment_sha
    if dataset_identity_sha is not None:
        integrity["dataset_identity_sha256"] = dataset_identity_sha
    if git_provenance is not None:
        integrity.update(
            {
                "git_revision": git_provenance.get("revision"),
                "git_branch": git_provenance.get("branch"),
                "git_dirty": git_provenance.get("dirty"),
            }
        )
    if authorization_evidence is not None:
        integrity.update(
            {
                "authorization_stage": authorization_evidence["stage"],
                "authorization_decision": authorization_evidence["decision"],
                "authorization_gate_bytes": authorization_evidence["gate_bytes"],
                "authorization_gate_sha256": authorization_evidence["gate_sha256"],
                "authorization_gate_identity_sha256": authorization_evidence[
                    "gate_identity_sha256"
                ],
            }
        )
    write_json_report(integrity_path, integrity)
    write_json_report(
        latest_path,
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
    expected_runtime_environment: Mapping[str, Any] | None = None,
    expected_git_provenance: Mapping[str, Any] | None = None,
    expected_dataset_provenance: Mapping[str, Any] | None = None,
    expected_training_authorization: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = reject_symlink_chain(path, name="training checkpoint")
    integrity = None
    if verify_integrity:
        integrity = verify_training_checkpoint(path)
    if expected_git_provenance is not None:
        expected_git_provenance = _validate_git_provenance(
            expected_git_provenance,
            label="Expected Git provenance",
        )
        if integrity is not None:
            integrity_git = {
                "revision": integrity.get("git_revision"),
                "branch": integrity.get("git_branch"),
                "dirty": integrity.get("git_dirty"),
            }
            if expected_git_provenance != integrity_git:
                raise ValueError("Checkpoint Git provenance differs from expected revision")
    expected_dataset_identity_sha = None
    if expected_dataset_provenance is not None and expected_dataset_provenance.get(
        "formal"
    ) is True:
        expected_config_data = (
            expected_config.get("data") if isinstance(expected_config, Mapping) else None
        )
        if not isinstance(expected_config_data, Mapping) or not expected_config_data.get(
            "dataset"
        ):
            raise ValueError("formal expected dataset provenance requires a config alias")
        expected_dataset_identity_sha = validate_dataset_provenance(
            expected_dataset_provenance,
            expected_dataset=str(expected_config_data["dataset"]),
        )["identity_sha256"]
        if integrity is None or integrity.get(
            "dataset_identity_sha256"
        ) != expected_dataset_identity_sha:
            raise ValueError(
                "Checkpoint dataset identity differs from expected provenance"
            )
    expected_authorization_evidence = None
    if expected_training_authorization is not None:
        expected_authorization_evidence = validate_generation_training_authorization(
            expected_training_authorization
        )
        if integrity is None:
            raise ValueError("Checkpoint training authorization requires integrity metadata")
        expected_integrity_authorization = {
            "authorization_stage": expected_authorization_evidence["stage"],
            "authorization_decision": expected_authorization_evidence["decision"],
            "authorization_gate_bytes": expected_authorization_evidence["gate_bytes"],
            "authorization_gate_sha256": expected_authorization_evidence["gate_sha256"],
            "authorization_gate_identity_sha256": expected_authorization_evidence[
                "gate_identity_sha256"
            ],
        }
        if any(
            integrity.get(key) != value
            for key, value in expected_integrity_authorization.items()
        ):
            raise ValueError(
                "Checkpoint training authorization differs from the expected promotion gate"
            )
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    if integrity is not None:
        validate_checkpoint_training_authorization(checkpoint, integrity)
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
    if expected_runtime_environment is not None:
        extra_state = checkpoint.get("extra_state")
        if not isinstance(extra_state, Mapping):
            raise ValueError("Checkpoint is missing exact-resume extra state")
        checkpoint_environment = extra_state.get("runtime_environment")
        if not isinstance(checkpoint_environment, Mapping):
            raise ValueError("Checkpoint is missing its exact-resume runtime environment")
        checkpoint_environment_sha = runtime_environment_sha256(checkpoint_environment)
        if extra_state.get("runtime_environment_sha256") != checkpoint_environment_sha:
            raise ValueError("Checkpoint runtime environment SHA256 is inconsistent")
        if integrity is not None and integrity.get(
            "runtime_environment_sha256"
        ) != checkpoint_environment_sha:
            raise ValueError(
                "Checkpoint runtime environment differs from integrity metadata"
            )
        mismatches = runtime_environment_mismatch_paths(
            expected_runtime_environment, checkpoint_environment
        )
        if mismatches:
            preview = ", ".join(mismatches[:8])
            if len(mismatches) > 8:
                preview += f", ... ({len(mismatches)} fields)"
            raise ValueError(f"Checkpoint runtime environment mismatch at: {preview}")
    if expected_git_provenance is not None:
        extra_state = checkpoint.get("extra_state")
        if not isinstance(extra_state, Mapping):
            raise ValueError("Checkpoint is missing exact-resume extra state")
        checkpoint_git = extra_state.get("git")
        if not isinstance(checkpoint_git, Mapping):
            raise ValueError("Checkpoint is missing its exact-resume Git provenance")
        mismatches = _config_mismatch_paths(
            expected_git_provenance,
            checkpoint_git,
            path="git",
        )
        if mismatches:
            preview = ", ".join(mismatches[:8])
            raise ValueError(f"Checkpoint Git provenance mismatch at: {preview}")
        if integrity is not None:
            integrity_git = {
                "revision": integrity.get("git_revision"),
                "branch": integrity.get("git_branch"),
                "dirty": integrity.get("git_dirty"),
            }
            if dict(checkpoint_git) != integrity_git:
                raise ValueError("Checkpoint Git provenance differs from integrity metadata")
    if expected_dataset_provenance is not None:
        extra_state = checkpoint.get("extra_state")
        if not isinstance(extra_state, Mapping):
            raise ValueError("Checkpoint is missing exact-resume extra state")
        checkpoint_dataset = extra_state.get("dataset_provenance")
        if not isinstance(checkpoint_dataset, Mapping):
            raise ValueError("Checkpoint is missing its dataset provenance")
        mismatches = _config_mismatch_paths(
            expected_dataset_provenance,
            checkpoint_dataset,
            path="dataset_provenance",
        )
        if mismatches:
            preview = ", ".join(mismatches[:8])
            raise ValueError(f"Checkpoint dataset provenance mismatch at: {preview}")
        if expected_dataset_identity_sha is not None:
            checkpoint_config = checkpoint.get("config")
            checkpoint_data = (
                checkpoint_config.get("data")
                if isinstance(checkpoint_config, Mapping)
                else None
            )
            if not isinstance(checkpoint_data, Mapping):
                raise ValueError("Checkpoint formal dataset config is missing")
            checkpoint_identity_sha = validate_dataset_provenance(
                checkpoint_dataset,
                expected_dataset=str(checkpoint_data.get("dataset", "")),
            )["identity_sha256"]
            if checkpoint_identity_sha != expected_dataset_identity_sha:
                raise ValueError(
                    "Checkpoint payload dataset identity differs from expected provenance"
                )
            if integrity is not None and integrity.get(
                "dataset_identity_sha256"
            ) != checkpoint_identity_sha:
                raise ValueError(
                    "Checkpoint dataset identity differs from integrity metadata"
                )
    if expected_training_authorization is not None:
        extra_state = checkpoint.get("extra_state")
        if not isinstance(extra_state, Mapping):
            raise ValueError("Checkpoint is missing exact-resume extra state")
        checkpoint_authorization = extra_state.get("training_authorization")
        if not isinstance(checkpoint_authorization, Mapping):
            raise ValueError("Checkpoint is missing its generation training authorization")
        mismatches = _config_mismatch_paths(
            expected_training_authorization,
            checkpoint_authorization,
            path="training_authorization",
        )
        if mismatches:
            preview = ", ".join(mismatches[:8])
            raise ValueError(
                f"Checkpoint generation training authorization mismatch at: {preview}"
            )
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
    root = reject_symlink_chain(directory, name="checkpoint directory")
    paths = sorted(root.glob("checkpoint_step_*.pt"))
    paths = [
        reject_symlink_chain(path, name="training checkpoint") for path in paths
    ]
    protected = {int(step) for step in protected_steps}
    recent = set(paths[-keep_last:])
    retained = recent | {
        path
        for path in paths
        if int(path.stem.removeprefix("checkpoint_step_")) in protected
    }
    removed = [path for path in paths if path not in retained]
    for path in removed:
        integrity_path = reject_symlink_chain(
            checkpoint_integrity_path(path),
            name="checkpoint integrity manifest",
        )
        path.unlink()
        integrity_path.unlink(missing_ok=True)
    return removed
