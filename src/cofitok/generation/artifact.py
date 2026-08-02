from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from cofitok.configs import config_from_dict
from cofitok.environment import runtime_environment_sha256
from cofitok.generation_authorization import (
    capture_generation_gate_binding,
    validate_generation_gate_binding,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.models import CoFiTokTiny
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.authorization import (
    validate_checkpoint_training_authorization,
    validate_generation_training_authorization,
)
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


INFERENCE_ARTIFACT_TYPE = "cofitok_generation_inference"
INFERENCE_ARTIFACT_FORMAT_VERSION = 4
INFERENCE_ARTIFACT_INTEGRITY_VERSION = 4
INFERENCE_EXPORT_MANIFEST_SCHEMA_VERSION = 1
INFERENCE_EXPORT_MANIFEST_ROLE = "cofitok_generation_inference_export_manifest"
INFERENCE_EXPORT_LOCK_ROLE = "generation_inference_artifact_export"


def inference_export_manifest_path(path: str | Path) -> Path:
    artifact = Path(path)
    return artifact.with_name(f"{artifact.name}.export_manifest.json")


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _source_training_authorization_fields(
    source_integrity: Mapping[str, Any],
) -> dict[str, Any] | None:
    names = (
        "authorization_stage",
        "authorization_decision",
        "authorization_gate_bytes",
        "authorization_gate_sha256",
        "authorization_gate_identity_sha256",
    )
    present = [name in source_integrity for name in names]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Source checkpoint training authorization is incomplete")
    result = {name: source_integrity[name] for name in names}
    if (
        not isinstance(result["authorization_stage"], str)
        or not result["authorization_stage"]
        or not isinstance(result["authorization_decision"], str)
        or not result["authorization_decision"]
        or int(result["authorization_gate_bytes"]) < 1
        or len(str(result["authorization_gate_sha256"])) != 64
        or len(str(result["authorization_gate_identity_sha256"])) != 64
    ):
        raise ValueError("Source checkpoint training authorization is malformed")
    return result


def _validated_export_execution(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Inference export execution provenance is malformed")
    if set(value) != {
        "git",
        "runtime_environment",
        "runtime_environment_sha256",
    }:
        raise ValueError("Inference export execution provenance fields differ")
    git = value.get("git")
    if (
        not isinstance(git, Mapping)
        or set(git) != {"revision", "branch", "tracked_dirty"}
        or not isinstance(git.get("revision"), str)
        or len(str(git["revision"])) != 40
        or not isinstance(git.get("branch"), str)
        or not git["branch"]
        or not isinstance(git.get("tracked_dirty"), bool)
    ):
        raise ValueError("Inference export execution Git provenance is malformed")
    environment = value.get("runtime_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("Inference export runtime environment is malformed")
    environment_sha = runtime_environment_sha256(environment)
    if value.get("runtime_environment_sha256") != environment_sha:
        raise ValueError("Inference export runtime environment SHA256 differs")
    return _json_copy(value)


def _expected_export_manifest(
    *,
    source_path: Path,
    target: Path,
    source_integrity: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    release_authorization: Mapping[str, Any] | None,
    execution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": INFERENCE_EXPORT_MANIFEST_SCHEMA_VERSION,
        "role": INFERENCE_EXPORT_MANIFEST_ROLE,
        "artifact_type": INFERENCE_ARTIFACT_TYPE,
        "artifact_format_version": INFERENCE_ARTIFACT_FORMAT_VERSION,
        "weights": "ema_export",
        "source": {
            "path": source_path.resolve().as_posix(),
            "bytes": int(source_integrity["checkpoint_bytes"]),
            "sha256": source_integrity["checkpoint_sha256"],
            "step": int(source_integrity["step"]),
            "checkpoint_format_version": int(
                source_integrity["checkpoint_format_version"]
            ),
            "integrity_manifest": file_identity(
                checkpoint_integrity_path(source_path)
            ),
            "runtime_environment_sha256": source_provenance[
                "runtime_environment_sha256"
            ],
            "git": _json_copy(source_provenance["git"]),
            "training_authorization": _source_training_authorization_fields(
                source_integrity
            ),
        },
        "target": {
            "artifact": target.resolve().as_posix(),
            "integrity_manifest": checkpoint_integrity_path(target)
            .resolve()
            .as_posix(),
        },
        "release_authorization": (
            None
            if release_authorization is None
            else _json_copy(release_authorization)
        ),
        "execution": None if execution is None else _json_copy(execution),
    }


def _read_exact_export_manifest(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    payload = read_json_object(path, name="inference export manifest")
    if payload != expected:
        raise ValueError("Existing inference export manifest request differs")
    return payload


def verify_inference_export_manifest(
    path: str | Path,
    *,
    expected_artifact: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = reject_symlink_chain(path, name="inference export manifest")
    payload = read_json_object(
        manifest_path,
        name="inference export manifest",
    )
    if set(payload) != {
        "schema_version",
        "role",
        "artifact_type",
        "artifact_format_version",
        "weights",
        "source",
        "target",
        "release_authorization",
        "execution",
    }:
        raise ValueError("Inference export manifest fields differ")
    if (
        payload.get("schema_version") != INFERENCE_EXPORT_MANIFEST_SCHEMA_VERSION
        or payload.get("role") != INFERENCE_EXPORT_MANIFEST_ROLE
        or payload.get("artifact_type") != INFERENCE_ARTIFACT_TYPE
        or payload.get("artifact_format_version")
        != INFERENCE_ARTIFACT_FORMAT_VERSION
        or payload.get("weights") != "ema_export"
    ):
        raise ValueError("Inference export manifest identity is invalid")
    source = payload.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "path",
        "bytes",
        "sha256",
        "step",
        "checkpoint_format_version",
        "integrity_manifest",
        "runtime_environment_sha256",
        "git",
        "training_authorization",
    }:
        raise ValueError("Inference export manifest source is malformed")
    source_path = reject_symlink_chain(
        str(source.get("path", "")),
        name="inference export source checkpoint",
    )
    source_integrity = verify_training_checkpoint(source_path)
    source_provenance = _source_checkpoint_provenance(source_integrity)
    actual_source = {
        "path": source_path.resolve().as_posix(),
        "bytes": int(source_integrity["checkpoint_bytes"]),
        "sha256": source_integrity["checkpoint_sha256"],
        "step": int(source_integrity["step"]),
        "checkpoint_format_version": int(
            source_integrity["checkpoint_format_version"]
        ),
        "integrity_manifest": file_identity(
            checkpoint_integrity_path(source_path)
        ),
        "runtime_environment_sha256": source_provenance[
            "runtime_environment_sha256"
        ],
        "git": source_provenance["git"],
        "training_authorization": _source_training_authorization_fields(
            source_integrity
        ),
    }
    if dict(source) != actual_source:
        raise ValueError("Inference export manifest source identity differs")
    target = payload.get("target")
    if not isinstance(target, Mapping) or set(target) != {
        "artifact",
        "integrity_manifest",
    }:
        raise ValueError("Inference export manifest target is malformed")
    artifact = reject_symlink_chain(
        str(target.get("artifact", "")),
        name="inference export artifact",
    )
    expected_target = {
        "artifact": artifact.resolve().as_posix(),
        "integrity_manifest": checkpoint_integrity_path(artifact)
        .resolve()
        .as_posix(),
    }
    if dict(target) != expected_target:
        raise ValueError("Inference export manifest target identity differs")
    if manifest_path.resolve() != inference_export_manifest_path(artifact).resolve():
        raise ValueError("Inference export manifest path is not canonical")
    if expected_artifact is not None:
        expected_path = reject_symlink_chain(
            expected_artifact,
            name="expected inference export artifact",
        )
        if artifact.resolve() != expected_path.resolve():
            raise ValueError("Inference export manifest names another artifact")
    release_authorization = payload.get("release_authorization")
    formal_source = source["training_authorization"] is not None
    if formal_source:
        if not isinstance(release_authorization, Mapping):
            raise ValueError("Formal inference export manifest lacks release authorization")
        validate_generation_gate_binding(
            release_authorization,
            expected_stage="full",
        )
        actual_release = capture_generation_gate_binding(
            str(release_authorization["gate_path"]),
            expected_stage="full",
        )
        if dict(release_authorization) != actual_release:
            raise ValueError("Inference export manifest release authorization differs")
    elif release_authorization is not None:
        raise ValueError("Non-formal inference export manifest has release authorization")
    _validated_export_execution(payload.get("execution"))
    return payload


def _remove_stale_artifact_temporaries(target: Path) -> int:
    removed = 0
    for candidate in target.parent.glob(f"{target.name}.tmp-*"):
        reject_symlink_chain(candidate, name="stale inference export temporary")
        if not candidate.is_file():
            raise ValueError(
                f"Stale inference export temporary is not a regular file: {candidate}"
            )
        candidate.unlink()
        removed += 1
    return removed


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


def _validated_training_authorization(
    value: Any,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Inference artifact training authorization is malformed")
    validate_generation_training_authorization(value)
    return dict(value)


def _authorization_integrity_fields(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = validate_generation_training_authorization(authorization)
    return {
        "authorization_stage": evidence["stage"],
        "authorization_decision": evidence["decision"],
        "authorization_gate_bytes": evidence["gate_bytes"],
        "authorization_gate_sha256": evidence["gate_sha256"],
        "authorization_gate_identity_sha256": evidence[
            "gate_identity_sha256"
        ],
    }


def _validated_release_authorization(
    value: Any,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Inference artifact release authorization is malformed")
    validate_generation_gate_binding(value, expected_stage="full")
    return dict(value)


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
    source_training_authorization = _validated_training_authorization(
        integrity.get("source_training_authorization")
    )
    release_authorization = _validated_release_authorization(
        integrity.get("release_authorization")
    )
    if (source_training_authorization is None) != (
        release_authorization is None
    ):
        raise ValueError(
            "Formal inference artifact must bind both training and release authorization"
        )
    return integrity


def export_ema_inference_artifact(
    training_checkpoint: str | Path,
    output: str | Path,
    *,
    release_gate: str | Path | None = None,
    resume: bool = False,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_path = reject_symlink_chain(
        training_checkpoint,
        name="inference export source checkpoint",
    )
    target = reject_symlink_chain(output, name="inference export artifact")
    if source_path.resolve() == target.resolve():
        raise ValueError("Inference export source and target must differ")
    validated_execution = _validated_export_execution(execution)
    with exclusive_output_lock(target, role=INFERENCE_EXPORT_LOCK_ROLE):
        return _export_ema_inference_artifact_locked(
            source_path,
            target,
            release_gate=release_gate,
            resume=bool(resume),
            execution=validated_execution,
        )


def _completed_export_report(
    *,
    source_path: Path,
    target: Path,
    source_integrity: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    source_training_authorization: Mapping[str, Any] | None,
    release_authorization: Mapping[str, Any] | None,
    execution: Mapping[str, Any] | None,
    verified: Mapping[str, Any],
    reused: bool,
    resume_requested: bool,
    partial_outputs_recovered: bool,
) -> dict[str, Any]:
    manifest_path = inference_export_manifest_path(target)
    return {
        "schema_version": 4,
        "status": "completed",
        "reused": reused,
        "resume_requested": resume_requested,
        "partial_outputs_recovered": partial_outputs_recovered,
        "source_checkpoint": source_path.resolve().as_posix(),
        "source_checkpoint_sha256": source_integrity["checkpoint_sha256"],
        "source_checkpoint_bytes": int(source_integrity["checkpoint_bytes"]),
        "source_runtime_environment_sha256": source_provenance[
            "runtime_environment_sha256"
        ],
        "source_git": _json_copy(source_provenance["git"]),
        "source_training_authorization": (
            None
            if source_training_authorization is None
            else _json_copy(source_training_authorization)
        ),
        "release_authorization": (
            None
            if release_authorization is None
            else _json_copy(release_authorization)
        ),
        "execution": None if execution is None else _json_copy(execution),
        "export_manifest": file_identity(manifest_path),
        "artifact": target.resolve().as_posix(),
        "artifact_integrity_manifest": checkpoint_integrity_path(target)
        .resolve()
        .as_posix(),
        "artifact_sha256": verified["artifact_sha256"],
        "artifact_bytes": int(verified["artifact_bytes"]),
        "checkpoint_step": int(verified["step"]),
        "weights": "ema_export",
        "verified": True,
    }


def _export_ema_inference_artifact_locked(
    source_path: Path,
    target: Path,
    *,
    release_gate: str | Path | None,
    resume: bool,
    execution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_integrity = verify_training_checkpoint(source_path)
    source_provenance = _source_checkpoint_provenance(source_integrity)
    formal_source = "authorization_stage" in source_integrity
    if formal_source and not release_gate:
        raise ValueError(
            "Formal inference artifact export requires a full release gate"
        )
    if not formal_source and release_gate:
        raise ValueError(
            "A release gate may only authorize a formal full-training checkpoint"
        )
    release_authorization = (
        capture_generation_gate_binding(
            release_gate,
            expected_stage="full",
        )
        if release_gate
        else None
    )
    target_integrity = reject_symlink_chain(
        checkpoint_integrity_path(target),
        name="inference export integrity manifest",
    )
    manifest_path = reject_symlink_chain(
        inference_export_manifest_path(target),
        name="inference export manifest",
    )
    expected_manifest = _expected_export_manifest(
        source_path=source_path,
        target=target,
        source_integrity=source_integrity,
        source_provenance=source_provenance,
        release_authorization=release_authorization,
        execution=execution,
    )
    for path, name in (
        (target, "inference export artifact"),
        (target_integrity, "inference export integrity manifest"),
        (manifest_path, "inference export manifest"),
    ):
        if path.exists() and not path.is_file():
            raise ValueError(f"{name} is not a regular file: {path}")

    artifact_exists = target.is_file()
    integrity_exists = target_integrity.is_file()
    manifest_exists = manifest_path.is_file()
    if artifact_exists and integrity_exists:
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
        existing_authorization = _validated_training_authorization(
            existing.get("source_training_authorization")
        )
        source_has_authorization = "authorization_stage" in source_integrity
        if (existing_authorization is None) != (not source_has_authorization):
            raise ValueError("Existing inference artifact authorization differs")
        if existing_authorization is not None:
            expected_fields = _authorization_integrity_fields(
                existing_authorization
            )
            if any(
                source_integrity.get(key) != value
                for key, value in expected_fields.items()
            ):
                raise ValueError("Existing inference artifact authorization differs")
        if existing.get("release_authorization") != release_authorization:
            raise ValueError("Existing inference artifact release authorization differs")
        if not manifest_exists:
            raise ValueError("Completed inference artifact lacks its export manifest")
        _read_exact_export_manifest(manifest_path, expected=expected_manifest)
        return _completed_export_report(
            source_path=source_path,
            target=target,
            source_integrity=source_integrity,
            source_provenance=source_provenance,
            source_training_authorization=existing_authorization,
            release_authorization=release_authorization,
            execution=execution,
            verified=existing,
            reused=True,
            resume_requested=resume,
            partial_outputs_recovered=False,
        )

    partial_outputs_recovered = False
    if artifact_exists != integrity_exists:
        if not resume:
            raise ValueError(
                "Partial inference export requires explicit resume"
            )
        if not manifest_exists:
            raise ValueError(
                "Partial inference export lacks a matching export manifest"
            )
        _read_exact_export_manifest(manifest_path, expected=expected_manifest)
        if artifact_exists:
            target.unlink()
        if integrity_exists:
            target_integrity.unlink()
        partial_outputs_recovered = True
    elif manifest_exists:
        _read_exact_export_manifest(manifest_path, expected=expected_manifest)
        if not resume:
            raise ValueError(
                "Interrupted inference export requires explicit resume"
            )
        partial_outputs_recovered = True
    else:
        write_json_report(manifest_path, expected_manifest)

    if partial_outputs_recovered:
        _remove_stale_artifact_temporaries(target)

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != int(source_integrity["step"]):
        raise ValueError("Training checkpoint step differs from integrity metadata")
    source_training_authorization = validate_checkpoint_training_authorization(
        checkpoint,
        source_integrity,
    )
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
        "release_authorization": release_authorization,
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
            "training_authorization": source_training_authorization,
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
        "source_training_authorization": source_training_authorization,
        "release_authorization": release_authorization,
    }
    write_json_report(target_integrity, integrity)
    verified = verify_inference_artifact(target)
    return _completed_export_report(
        source_path=source_path,
        target=target,
        source_integrity=source_integrity,
        source_provenance=source_provenance,
        source_training_authorization=source_training_authorization,
        release_authorization=release_authorization,
        execution=execution,
        verified=verified,
        reused=False,
        resume_requested=resume,
        partial_outputs_recovered=partial_outputs_recovered,
    )
