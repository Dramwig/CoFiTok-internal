from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation_authorization import (
    GENERATION_GATE_BINDING_SCHEMA_VERSION,
    build_generation_gate_binding,
    capture_generation_gate_binding,
    validate_generation_gate_binding,
)


GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION = (
    GENERATION_GATE_BINDING_SCHEMA_VERSION
)
GENERATION_TRAINING_AUTHORIZATION_STAGE = "scaling"


def build_generation_training_authorization(
    gate: dict[str, Any],
    *,
    gate_path: str,
    gate_bytes: int,
    gate_sha256: str,
) -> dict[str, Any]:
    return build_generation_gate_binding(
        gate,
        expected_stage=GENERATION_TRAINING_AUTHORIZATION_STAGE,
        gate_path=gate_path,
        gate_bytes=gate_bytes,
        gate_sha256=gate_sha256,
    )


def capture_generation_training_authorization(
    gate_path: str | Path,
) -> dict[str, Any]:
    return capture_generation_gate_binding(
        gate_path,
        expected_stage=GENERATION_TRAINING_AUTHORIZATION_STAGE,
    )


def validate_generation_training_authorization(
    authorization: Mapping[str, Any],
    *,
    expected_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return validate_generation_gate_binding(
        authorization,
        expected_stage=GENERATION_TRAINING_AUTHORIZATION_STAGE,
        expected_gate=expected_gate,
    )


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
    return dict(payload_authorization)
