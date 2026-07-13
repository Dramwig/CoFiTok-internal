from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation_gate import (
    generation_gate_identity_sha256,
    validate_generation_gate_authorization,
)
from cofitok.reporting import file_sha256


GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION = 1
GENERATION_TRAINING_AUTHORIZATION_STAGE = "scaling"


def build_generation_training_authorization(
    gate: dict[str, Any],
    *,
    gate_path: str,
    gate_bytes: int,
    gate_sha256: str,
) -> dict[str, Any]:
    validation = validate_generation_gate_authorization(
        gate,
        expected_stage=GENERATION_TRAINING_AUTHORIZATION_STAGE,
    )
    if gate_bytes < 1:
        raise ValueError("generation training authorization gate is empty")
    if len(gate_sha256) != 64:
        raise ValueError("generation training authorization gate SHA256 is malformed")
    resolved_path = Path(gate_path).resolve().as_posix()
    return {
        "schema_version": GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
        "status": "pass",
        "stage": validation["stage"],
        "decision": validation["decision"],
        "gate_path": resolved_path,
        "gate_bytes": int(gate_bytes),
        "gate_sha256": gate_sha256,
        "gate_identity_sha256": generation_gate_identity_sha256(gate),
        "validated_thresholds": validation["validated_thresholds"],
    }


def capture_generation_training_authorization(
    gate_path: str | Path,
) -> dict[str, Any]:
    path = Path(gate_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    return build_generation_training_authorization(
        gate,
        gate_path=path.as_posix(),
        gate_bytes=path.stat().st_size,
        gate_sha256=file_sha256(path),
    )


def validate_generation_training_authorization(
    authorization: Mapping[str, Any],
    *,
    expected_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "stage",
        "decision",
        "gate_path",
        "gate_bytes",
        "gate_sha256",
        "gate_identity_sha256",
        "validated_thresholds",
    }
    if set(authorization) != required:
        raise ValueError(
            "generation training authorization must contain exactly: "
            + ", ".join(sorted(required))
        )
    if (
        int(authorization["schema_version"])
        != GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION
        or authorization["status"] != "pass"
        or authorization["stage"] != GENERATION_TRAINING_AUTHORIZATION_STAGE
        or authorization["decision"] != "promote_to_full_imagenet256"
    ):
        raise ValueError("generation training authorization identity is invalid")
    gate_path = str(authorization["gate_path"])
    if not gate_path or not Path(gate_path).is_absolute():
        raise ValueError("generation training authorization gate path is not absolute")
    if int(authorization["gate_bytes"]) < 1:
        raise ValueError("generation training authorization gate byte count is invalid")
    for name in ("gate_sha256", "gate_identity_sha256"):
        if len(str(authorization[name])) != 64:
            raise ValueError(f"generation training authorization {name} is malformed")
    thresholds = authorization["validated_thresholds"]
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("generation training authorization thresholds are missing")

    if expected_gate is not None:
        validation = validate_generation_gate_authorization(
            expected_gate,
            expected_stage=GENERATION_TRAINING_AUTHORIZATION_STAGE,
        )
        expected_identity = generation_gate_identity_sha256(expected_gate)
        if authorization["gate_identity_sha256"] != expected_identity:
            raise ValueError("generation training authorization binds another promotion gate")
        if dict(thresholds) != validation["validated_thresholds"]:
            raise ValueError("generation training authorization thresholds differ from the gate")
    return {
        "schema_version": GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
        "stage": authorization["stage"],
        "decision": authorization["decision"],
        "gate_path": gate_path,
        "gate_identity_sha256": authorization["gate_identity_sha256"],
        "gate_sha256": authorization["gate_sha256"],
        "gate_bytes": int(authorization["gate_bytes"]),
    }


def validate_checkpoint_training_authorization(
    checkpoint: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any] | None:
    config = checkpoint.get("config")
    config_data = config.get("data") if isinstance(config, Mapping) else None
    config_runtime = config.get("runtime") if isinstance(config, Mapping) else None
    formal_full = (
        isinstance(config_data, Mapping)
        and isinstance(config_runtime, Mapping)
        and config_data.get("dataset") == "imagenet_256"
        and int(config_runtime.get("steps", -1)) == 300_000
    )
    extra_state = checkpoint.get("extra_state")
    payload_authorization = (
        extra_state.get("training_authorization")
        if isinstance(extra_state, Mapping)
        else None
    )
    sidecar_keys = {
        "authorization_stage",
        "authorization_decision",
        "authorization_gate_bytes",
        "authorization_gate_sha256",
        "authorization_gate_identity_sha256",
    }
    sidecar_bound = sidecar_keys <= set(integrity)
    if formal_full and (payload_authorization is None or not sidecar_bound):
        raise ValueError(
            "Formal full checkpoint lacks its scaling-gate authorization binding"
        )
    if payload_authorization is None and not sidecar_bound:
        return None
    if not isinstance(payload_authorization, Mapping) or not sidecar_bound:
        raise ValueError(
            "Checkpoint payload and sidecar training authorization are inconsistent"
        )
    evidence = validate_generation_training_authorization(payload_authorization)
    expected_sidecar = {
        "authorization_stage": evidence["stage"],
        "authorization_decision": evidence["decision"],
        "authorization_gate_bytes": evidence["gate_bytes"],
        "authorization_gate_sha256": evidence["gate_sha256"],
        "authorization_gate_identity_sha256": evidence["gate_identity_sha256"],
    }
    if any(integrity.get(key) != value for key, value in expected_sidecar.items()):
        raise ValueError(
            "Checkpoint payload training authorization differs from integrity metadata"
        )
    return evidence
