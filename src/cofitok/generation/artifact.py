from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from cofitok.configs import config_from_dict
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.authorization import validate_checkpoint_training_authorization
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


INFERENCE_ARTIFACT_TYPE = "cofitok_generation_inference"
INFERENCE_ARTIFACT_FORMAT_VERSION = 2
INFERENCE_ARTIFACT_INTEGRITY_VERSION = 2


def _source_checkpoint_provenance(
    source_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_environment_sha = str(
        source_integrity.get("runtime_environment_sha256", "")
    )
    if len(runtime_environment_sha) != 64:
        raise ValueError("Source checkpoint lacks runtime environment provenance")
    git = {
        "revision": source_integrity.get("git_revision"),
        "branch": source_integrity.get("git_branch"),
        "dirty": source_integrity.get("git_dirty"),
    }
    if (
        not isinstance(git["revision"], str)
        or not git["revision"]
        or not isinstance(git["branch"], str)
        or not isinstance(git["dirty"], bool)
    ):
        raise ValueError("Source checkpoint lacks Git provenance")
    return {
        "runtime_environment_sha256": runtime_environment_sha,
        "git": git,
    }


def verify_inference_artifact(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    integrity_path = checkpoint_integrity_path(artifact)
    if not integrity_path.is_file():
        raise FileNotFoundError(f"Inference artifact integrity manifest is missing: {integrity_path}")
    with integrity_path.open("r", encoding="utf-8") as handle:
        integrity = json.load(handle)
    if integrity.get("schema_version") != INFERENCE_ARTIFACT_INTEGRITY_VERSION:
        raise ValueError("Unsupported inference artifact integrity version")
    if integrity.get("artifact_type") != INFERENCE_ARTIFACT_TYPE:
        raise ValueError("Integrity manifest is not a CoFiTok inference artifact")
    if integrity.get("artifact") != artifact.name:
        raise ValueError("Inference artifact integrity manifest names another file")
    expected_bytes = int(integrity.get("artifact_bytes", -1))
    actual_bytes = artifact.stat().st_size
    if expected_bytes != actual_bytes:
        raise ValueError(
            f"Inference artifact size mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    expected_sha256 = str(integrity.get("artifact_sha256", ""))
    if len(expected_sha256) != 64 or file_sha256(artifact) != expected_sha256:
        raise ValueError("Inference artifact SHA256 mismatch")
    if int(integrity.get("artifact_format_version", -1)) != INFERENCE_ARTIFACT_FORMAT_VERSION:
        raise ValueError("Unsupported inference artifact format version")
    if int(integrity.get("step", -1)) < 1:
        raise ValueError("Inference artifact step is invalid")
    if len(str(integrity.get("source_checkpoint_sha256", ""))) != 64:
        raise ValueError("Inference artifact source checkpoint SHA256 is malformed")
    if len(str(integrity.get("source_runtime_environment_sha256", ""))) != 64:
        raise ValueError("Inference artifact source environment SHA256 is malformed")
    if (
        not isinstance(integrity.get("source_git_revision"), str)
        or not integrity["source_git_revision"]
        or not isinstance(integrity.get("source_git_branch"), str)
        or not isinstance(integrity.get("source_git_dirty"), bool)
    ):
        raise ValueError("Inference artifact source Git provenance is malformed")
    return integrity


def export_ema_inference_artifact(
    training_checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    source_path = Path(training_checkpoint)
    target = Path(output)
    source_integrity = verify_training_checkpoint(source_path)
    source_provenance = _source_checkpoint_provenance(source_integrity)
    if target.exists() or checkpoint_integrity_path(target).exists():
        existing = verify_inference_artifact(target)
        if existing["source_checkpoint_sha256"] != source_integrity["checkpoint_sha256"]:
            raise ValueError("Existing inference artifact comes from another checkpoint")
        if (
            existing["source_runtime_environment_sha256"]
            != source_provenance["runtime_environment_sha256"]
            or {
                "revision": existing["source_git_revision"],
                "branch": existing["source_git_branch"],
                "dirty": existing["source_git_dirty"],
            }
            != source_provenance["git"]
        ):
            raise ValueError("Existing inference artifact source provenance differs")
        return {
            "schema_version": 2,
            "status": "completed",
            "reused": True,
            "source_checkpoint": source_path.resolve().as_posix(),
            "source_checkpoint_sha256": source_integrity["checkpoint_sha256"],
            "source_checkpoint_bytes": int(source_integrity["checkpoint_bytes"]),
            "source_runtime_environment_sha256": source_provenance[
                "runtime_environment_sha256"
            ],
            "source_git": source_provenance["git"],
            "artifact": target.resolve().as_posix(),
            "artifact_integrity_manifest": checkpoint_integrity_path(target).resolve().as_posix(),
            "artifact_sha256": existing["artifact_sha256"],
            "artifact_bytes": int(existing["artifact_bytes"]),
            "checkpoint_step": int(existing["step"]),
            "weights": "ema_export",
            "verified": True,
        }

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != int(source_integrity["step"]):
        raise ValueError("Training checkpoint step differs from integrity metadata")
    validate_checkpoint_training_authorization(checkpoint, source_integrity)
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, Mapping):
        raise ValueError("Training checkpoint lacks deployment provenance state")
    if (
        extra_state.get("runtime_environment_sha256")
        != source_provenance["runtime_environment_sha256"]
        or extra_state.get("git") != source_provenance["git"]
    ):
        raise ValueError("Training checkpoint deployment provenance is inconsistent")
    config = config_from_dict(checkpoint["config"])
    model = CoFiTokTiny(config.model)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema = ExponentialMovingAverage(
        model,
        decay=config.optimization.ema_decay,
        warmup_steps=config.optimization.ema_warmup_steps,
    )
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(model)
    payload = {
        "artifact_type": INFERENCE_ARTIFACT_TYPE,
        "format_version": INFERENCE_ARTIFACT_FORMAT_VERSION,
        "step": int(checkpoint["step"]),
        "config": checkpoint["config"],
        "model": model.state_dict(),
        "weights": "ema_export",
        "source": {
            "checkpoint": source_path.resolve().as_posix(),
            "checkpoint_sha256": source_integrity["checkpoint_sha256"],
            "checkpoint_format_version": int(
                source_integrity["checkpoint_format_version"]
            ),
            "runtime_environment_sha256": source_provenance[
                "runtime_environment_sha256"
            ],
            "git": source_provenance["git"],
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    del checkpoint, payload, ema, model

    integrity = {
        "schema_version": INFERENCE_ARTIFACT_INTEGRITY_VERSION,
        "artifact_type": INFERENCE_ARTIFACT_TYPE,
        "artifact": target.name,
        "artifact_bytes": target.stat().st_size,
        "artifact_sha256": file_sha256(target),
        "artifact_format_version": INFERENCE_ARTIFACT_FORMAT_VERSION,
        "step": int(source_integrity["step"]),
        "source_checkpoint_sha256": source_integrity["checkpoint_sha256"],
        "source_runtime_environment_sha256": source_provenance[
            "runtime_environment_sha256"
        ],
        "source_git_revision": source_provenance["git"]["revision"],
        "source_git_branch": source_provenance["git"]["branch"],
        "source_git_dirty": source_provenance["git"]["dirty"],
    }
    write_json_report(checkpoint_integrity_path(target), integrity)
    verified = verify_inference_artifact(target)
    return {
        "schema_version": 2,
        "status": "completed",
        "reused": False,
        "source_checkpoint": source_path.resolve().as_posix(),
        "source_checkpoint_sha256": source_integrity["checkpoint_sha256"],
        "source_checkpoint_bytes": int(source_integrity["checkpoint_bytes"]),
        "source_runtime_environment_sha256": source_provenance[
            "runtime_environment_sha256"
        ],
        "source_git": source_provenance["git"],
        "artifact": target.resolve().as_posix(),
        "artifact_integrity_manifest": checkpoint_integrity_path(target).resolve().as_posix(),
        "artifact_sha256": verified["artifact_sha256"],
        "artifact_bytes": int(verified["artifact_bytes"]),
        "checkpoint_step": int(verified["step"]),
        "weights": "ema_export",
        "verified": True,
    }
