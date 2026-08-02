from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation.artifact import verify_inference_artifact
from cofitok.reporting import file_sha256, write_json_report


GENERATION_RELEASE_RECEIPT_SCHEMA_VERSION = 1
GENERATION_RELEASE_RECEIPT_TYPE = "cofitok_generation_release_receipt"
_COMPLETION_PROFILES = {
    "large_scale_generation_v1": {
        "status": "complete",
        "check": "deployable_ema_inference_artifacts",
    },
    "stability_generation_system_v1": {
        "status": "pass",
        "check": "stability_release_authorized_inference",
    },
}
_METHODS = ("cofitok", "dense_identity")


def _read_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"release source is missing: {source}")
    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _completion_profile(audit: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    raw_profile = audit.get("profile")
    profile = (
        "large_scale_generation_v1"
        if raw_profile is None
        else str(raw_profile)
    )
    contract = _COMPLETION_PROFILES.get(profile)
    if contract is None:
        raise ValueError(f"unsupported generation completion profile: {profile}")
    if audit.get("schema_version") != 1:
        raise ValueError("unsupported generation completion audit schema")
    if (
        audit.get("status") != contract["status"]
        or audit.get("complete") is not True
        or audit.get("failed_checks") != []
        or audit.get("missing_checks") != []
    ):
        raise ValueError("generation completion audit did not pass")
    return profile, contract


def _completion_inference_evidence(
    audit: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    profile, contract = _completion_profile(audit)
    checks = audit.get("checks")
    if not isinstance(checks, list):
        raise ValueError("generation completion audit check list is missing")
    matches = [
        row
        for row in checks
        if isinstance(row, Mapping) and row.get("name") == contract["check"]
    ]
    if len(matches) != 1 or matches[0].get("status") != "pass":
        raise ValueError("generation completion audit lacks passing inference evidence")
    evidence = matches[0].get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(_METHODS):
        raise ValueError("generation completion inference evidence is incomplete")
    return profile, dict(evidence)


def _artifact_receipt_row(method: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(evidence.get("artifact_path", "")))
    if not path.is_absolute():
        raise ValueError(f"{method} completion artifact path is not absolute")
    sha256 = str(evidence.get("artifact_sha256", ""))
    smoke_sha256 = evidence.get("smoke_output_sha256")
    if (
        len(sha256) != 64
        or int(evidence.get("artifact_bytes", 0)) < 1
        or int(evidence.get("source_checkpoint_bytes", 0)) < 1
        or not isinstance(evidence.get("source_git"), Mapping)
        or not isinstance(evidence.get("training_authorization"), Mapping)
        or not isinstance(evidence.get("release_authorization"), Mapping)
        or int(evidence.get("smoke_output_count", 0)) < 1
        or not isinstance(smoke_sha256, list)
        or len(smoke_sha256) != int(evidence["smoke_output_count"])
        or any(len(str(value)) != 64 for value in smoke_sha256)
    ):
        raise ValueError(f"{method} completion artifact evidence is malformed")
    return {
        "path": path.resolve().as_posix(),
        "artifact_sha256": sha256,
        "artifact_bytes": int(evidence["artifact_bytes"]),
        "checkpoint_step": 300_000,
        "source_checkpoint_bytes": int(evidence["source_checkpoint_bytes"]),
        "source_runtime_environment_sha256": evidence.get(
            "source_runtime_environment_sha256"
        ),
        "source_git": dict(evidence["source_git"]),
        "execution_git": evidence.get("execution_git"),
        "export_runtime_environment_sha256": evidence.get(
            "export_runtime_environment_sha256"
        ),
        "execution_runtime_environment_sha256": evidence.get(
            "execution_runtime_environment_sha256"
        ),
        "training_authorization": dict(evidence["training_authorization"]),
        "release_authorization": dict(evidence["release_authorization"]),
        "smoke_output_count": int(evidence["smoke_output_count"]),
        "smoke_output_sha256": list(smoke_sha256),
    }


def _receipt_payload(
    completion_audit_path: str | Path,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    profile, evidence = _completion_inference_evidence(audit)
    expectations = (
        audit.get("expectations")
        if profile == "stability_generation_system_v1"
        else audit.get("expected_revisions")
    )
    if not isinstance(expectations, Mapping):
        raise ValueError("generation completion expectations are missing")
    artifacts = {
        method: _artifact_receipt_row(method, evidence[method])
        for method in _METHODS
    }
    if len({row["path"] for row in artifacts.values()}) != len(_METHODS):
        raise ValueError("generation completion artifacts must use distinct paths")
    return {
        "schema_version": GENERATION_RELEASE_RECEIPT_SCHEMA_VERSION,
        "receipt_type": GENERATION_RELEASE_RECEIPT_TYPE,
        "status": "completed",
        "completion_profile": profile,
        "completion_audit": _file_identity(completion_audit_path),
        "completion_expectations": dict(expectations),
        "artifacts": artifacts,
    }


def _validate_artifact_against_row(
    method: str,
    row: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> None:
    source_git = {
        "revision": integrity.get("source_git_revision"),
        "branch": integrity.get("source_git_branch"),
        "dirty": integrity.get("source_git_dirty"),
    }
    if (
        row.get("artifact_sha256") != integrity.get("artifact_sha256")
        or int(row.get("artifact_bytes", -1))
        != int(integrity.get("artifact_bytes", -2))
        or int(row.get("checkpoint_step", -1)) != int(integrity.get("step", -2))
        or row.get("source_runtime_environment_sha256")
        != integrity.get("source_runtime_environment_sha256")
        or row.get("source_git") != source_git
        or row.get("training_authorization")
        != integrity.get("source_training_authorization")
        or row.get("release_authorization")
        != integrity.get("release_authorization")
    ):
        raise ValueError(f"{method} inference artifact differs from release receipt")


def write_generation_release_receipt(
    completion_audit: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    audit = _read_object(completion_audit, name="generation completion audit")
    payload = _receipt_payload(completion_audit, audit)
    for method, row in payload["artifacts"].items():
        integrity = verify_inference_artifact(row["path"])
        _validate_artifact_against_row(method, row, integrity)
    target = Path(output)
    if target.exists():
        existing = _read_object(target, name="generation release receipt")
        if existing != payload:
            raise ValueError("existing generation release receipt differs")
        return existing
    write_json_report(target, payload)
    written = _read_object(target, name="generation release receipt")
    if written != payload:
        raise ValueError("written generation release receipt differs")
    return written


def verify_generation_release_receipt(
    receipt_path: str | Path,
    artifact_path: str | Path,
    *,
    artifact_integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _read_object(receipt_path, name="generation release receipt")
    if (
        receipt.get("schema_version") != GENERATION_RELEASE_RECEIPT_SCHEMA_VERSION
        or receipt.get("receipt_type") != GENERATION_RELEASE_RECEIPT_TYPE
        or receipt.get("status") != "completed"
    ):
        raise ValueError("generation release receipt header is invalid")
    audit_descriptor = receipt.get("completion_audit")
    if not isinstance(audit_descriptor, Mapping):
        raise ValueError("generation release receipt lacks completion audit identity")
    audit_path = Path(str(audit_descriptor.get("path", "")))
    actual_audit_identity = _file_identity(audit_path)
    if dict(audit_descriptor) != actual_audit_identity:
        raise ValueError("generation completion audit changed after release")
    audit = _read_object(audit_path, name="generation completion audit")
    expected = _receipt_payload(audit_path, audit)
    if receipt != expected:
        raise ValueError("generation release receipt differs from completion audit")

    artifact = Path(artifact_path).resolve()
    matches = [
        (method, row)
        for method, row in receipt["artifacts"].items()
        if Path(str(row.get("path", ""))).resolve() == artifact
    ]
    if len(matches) != 1:
        raise ValueError("inference artifact is not authorized by release receipt")
    method, row = matches[0]
    integrity = (
        artifact_integrity
        if artifact_integrity is not None
        else verify_inference_artifact(artifact)
    )
    _validate_artifact_against_row(method, row, integrity)
    receipt_identity = _file_identity(receipt_path)
    return {
        "receipt": receipt_identity,
        "completion_audit": actual_audit_identity,
        "completion_profile": receipt["completion_profile"],
        "completion_expectations": receipt["completion_expectations"],
        "method": method,
    }
