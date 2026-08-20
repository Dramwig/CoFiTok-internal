from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation.artifact import (
    inference_export_manifest_path,
    verify_inference_artifact,
    verify_inference_export_manifest,
)
from cofitok.reporting import file_sha256, write_json_report


GENERATION_RELEASE_RECEIPT_SCHEMA_VERSION = 1
GENERATION_RELEASE_RECEIPT_TYPE = "cofitok_generation_release_receipt"
_LARGE_SCALE_REQUIRED_CHECKS = (
    "ten_percent_matched_training",
    "controlled_revision_transition",
    "generation_storage_capacity",
    "full_training_operational_monitor",
    "scaling_promotion_gate",
    "full_matched_training",
    "full_training_runtime_environment",
    "full_training_checkpoint_code_provenance",
    "reproducible_full_checkpoint_files",
    "full_runtime_selection",
    "full_training_audits",
    "full_milestone_evaluations",
    "formal_50k_generation",
    "formal_sampling_runtime_selection",
    "deterministic_visual_quality_audit",
    "deployable_ema_inference_artifacts",
    "final_generation_gate",
    "final_comparison_report",
)
_STABILITY_REQUIRED_CHECKS = (
    "stability_5k_authorization",
    "stability_50k_monitor",
    "stability_50k_pair_summary",
    "stability_50k_checkpoint_integrity",
    "stability_scaling_gate",
    "stability_full_training_readiness",
    "stability_full_readiness_revision_bridge",
    "stability_full_launch_receipt",
    "stability_full_monitor",
    "stability_full_training_pair",
    "stability_full_runtime_selection",
    "stability_full_storage_capacity",
    "stability_full_milestones",
    "stability_full_formal_generation",
    "stability_full_runtime_and_visual",
    "stability_final_gate",
    "stability_strong_baseline_comparison",
    "stability_release_authorized_inference",
)
_CAPACITY_FULL_REQUIRED_CHECKS = (
    "capacity_full_training_supervisor_deployment",
    "capacity_full_training_completion",
    "capacity_full_training_integrity",
    "capacity_full_milestones",
    "capacity_full_posteval_supervisor_deployment",
    "capacity_full_posteval_result",
    "capacity_full_posteval_supervisor_status",
    "capacity_full_formal_generation",
    "capacity_full_runtime_and_visual",
    "capacity_full_final_gate",
    "capacity_full_strong_comparison",
    "capacity_full_release_authorized_inference",
)
_COMPLETION_PROFILES = {
    "large_scale_generation_v1": {
        "status": "complete",
        "check": "deployable_ema_inference_artifacts",
    },
    "stability_generation_system_v1": {
        "status": "pass",
        "check": "stability_release_authorized_inference",
    },
    "capacity_full_generation_system_v1": {
        "status": "pass",
        "check": "capacity_full_release_authorized_inference",
    },
}
_COMPLETION_REQUIRED_CHECKS = {
    "large_scale_generation_v1": _LARGE_SCALE_REQUIRED_CHECKS,
    "stability_generation_system_v1": _STABILITY_REQUIRED_CHECKS,
    "capacity_full_generation_system_v1": _CAPACITY_FULL_REQUIRED_CHECKS,
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


def _validated_completion_checks(
    audit: Mapping[str, Any],
    *,
    profile: str,
    contract: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    checks = audit.get("checks")
    if not isinstance(checks, list):
        raise ValueError("generation completion audit check list is missing")
    expected = _COMPLETION_REQUIRED_CHECKS[profile]
    rows: dict[str, Mapping[str, Any]] = {}
    names: list[str] = []
    for index, row in enumerate(checks):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"generation completion audit check row {index} is malformed"
            )
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"generation completion audit check row {index} lacks a name"
            )
        if name in rows:
            raise ValueError(
                f"generation completion audit contains duplicate check: {name}"
            )
        names.append(name)
        rows[name] = row

    missing = [name for name in expected if name not in rows]
    if missing:
        raise ValueError(
            "generation completion audit is missing required checks: "
            + ", ".join(missing)
        )
    unknown = [name for name in names if name not in expected]
    if unknown:
        raise ValueError(
            "generation completion audit contains unknown checks: "
            + ", ".join(unknown)
        )
    if tuple(names) != expected:
        raise ValueError(
            f"generation completion audit check order differs from {profile} contract"
        )
    non_passing = [name for name in expected if rows[name].get("status") != "pass"]
    if non_passing:
        raise ValueError(
            "generation completion audit requires every check to pass: "
            + ", ".join(non_passing)
        )
    missing_evidence = [
        name
        for name in expected
        if not isinstance(rows[name].get("evidence"), Mapping)
    ]
    if missing_evidence:
        raise ValueError(
            "generation completion audit lacks structured check evidence: "
            + ", ".join(missing_evidence)
        )
    return rows


def _completion_profile(
    audit: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Mapping[str, Any]]]:
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
    checks = _validated_completion_checks(
        audit,
        profile=profile,
        contract=contract,
    )
    return profile, contract, checks


def _completion_inference_evidence(
    audit: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    profile, contract, checks = _completion_profile(audit)
    evidence = checks[contract["check"]].get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(_METHODS):
        raise ValueError("generation completion inference evidence is incomplete")
    return profile, dict(evidence)


def _artifact_receipt_row(method: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(evidence.get("artifact_path", "")))
    if not path.is_absolute():
        raise ValueError(f"{method} completion artifact path is not absolute")
    sha256 = str(evidence.get("artifact_sha256", ""))
    source_checkpoint_sha256 = str(
        evidence.get("source_checkpoint_sha256", "")
    )
    export_manifest = evidence.get("export_manifest")
    smoke_sha256 = evidence.get("smoke_output_sha256")
    if (
        len(sha256) != 64
        or len(source_checkpoint_sha256) != 64
        or int(evidence.get("artifact_bytes", 0)) < 1
        or int(evidence.get("source_checkpoint_bytes", 0)) < 1
        or not isinstance(evidence.get("source_git"), Mapping)
        or not isinstance(evidence.get("training_authorization"), Mapping)
        or not isinstance(evidence.get("release_authorization"), Mapping)
        or not isinstance(export_manifest, Mapping)
        or set(export_manifest) != {"path", "bytes", "sha256"}
        or not Path(str(export_manifest.get("path", ""))).is_absolute()
        or int(export_manifest.get("bytes", 0)) < 1
        or len(str(export_manifest.get("sha256", ""))) != 64
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
        "source_checkpoint_sha256": source_checkpoint_sha256,
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
        "export_manifest": dict(export_manifest),
        "smoke_output_count": int(evidence["smoke_output_count"]),
        "smoke_output_sha256": list(smoke_sha256),
    }


def _receipt_payload(
    completion_audit_path: str | Path,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    profile, evidence = _completion_inference_evidence(audit)
    expectations = (
        audit.get("expected_revisions")
        if profile == "large_scale_generation_v1"
        else audit.get("expectations")
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
        or row.get("source_checkpoint_sha256")
        != integrity.get("source_checkpoint_sha256")
        or row.get("source_runtime_environment_sha256")
        != integrity.get("source_runtime_environment_sha256")
        or row.get("source_git") != source_git
        or row.get("training_authorization")
        != integrity.get("source_training_authorization")
        or row.get("release_authorization")
        != integrity.get("release_authorization")
    ):
        raise ValueError(f"{method} inference artifact differs from release receipt")


def _validate_manifest_against_row(
    method: str,
    row: Mapping[str, Any],
    *,
    verify_sources: bool,
) -> None:
    artifact = Path(str(row.get("path", ""))).resolve()
    expected_path = inference_export_manifest_path(artifact).resolve()
    descriptor = row.get("export_manifest")
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{method} release receipt lacks export manifest")
    actual_descriptor = _file_identity(expected_path)
    if dict(descriptor) != actual_descriptor:
        raise ValueError(f"{method} inference export manifest differs from release receipt")
    if verify_sources:
        manifest = verify_inference_export_manifest(
            expected_path,
            expected_artifact=artifact,
        )
    else:
        manifest = _read_object(
            expected_path,
            name=f"{method} inference export manifest",
        )
    source = manifest.get("source")
    target = manifest.get("target")
    execution = manifest.get("execution")
    training_authorization = row.get("training_authorization")
    expected_manifest_authorization = (
        None
        if not isinstance(training_authorization, Mapping)
        else {
            "authorization_stage": training_authorization.get("stage"),
            "authorization_decision": training_authorization.get("decision"),
            "authorization_gate_bytes": training_authorization.get("gate_bytes"),
            "authorization_gate_sha256": training_authorization.get("gate_sha256"),
            "authorization_gate_identity_sha256": training_authorization.get(
                "gate_identity_sha256"
            ),
        }
    )
    execution_git = None if not isinstance(execution, Mapping) else execution.get("git")
    execution_environment_sha = (
        None
        if not isinstance(execution, Mapping)
        else execution.get("runtime_environment_sha256")
    )
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != row.get("source_checkpoint_sha256")
        or int(source.get("bytes", -1))
        != int(row.get("source_checkpoint_bytes", -2))
        or int(source.get("step", -1)) != int(row.get("checkpoint_step", -2))
        or source.get("runtime_environment_sha256")
        != row.get("source_runtime_environment_sha256")
        or source.get("git") != row.get("source_git")
        or source.get("training_authorization")
        != expected_manifest_authorization
        or not isinstance(target, Mapping)
        or target.get("artifact") != artifact.as_posix()
        or manifest.get("release_authorization")
        != row.get("release_authorization")
        or execution_git != row.get("execution_git")
        or execution_environment_sha
        != row.get("export_runtime_environment_sha256")
    ):
        raise ValueError(f"{method} inference export manifest provenance differs")


def write_generation_release_receipt(
    completion_audit: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    audit = _read_object(completion_audit, name="generation completion audit")
    payload = _receipt_payload(completion_audit, audit)
    for method, row in payload["artifacts"].items():
        integrity = verify_inference_artifact(row["path"])
        _validate_artifact_against_row(method, row, integrity)
        _validate_manifest_against_row(method, row, verify_sources=True)
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
    _validate_manifest_against_row(method, row, verify_sources=False)
    receipt_identity = _file_identity(receipt_path)
    return {
        "receipt": receipt_identity,
        "completion_audit": actual_audit_identity,
        "completion_profile": receipt["completion_profile"],
        "completion_expectations": receipt["completion_expectations"],
        "method": method,
    }
