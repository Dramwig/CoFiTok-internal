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


GENERATION_GATE_BINDING_SCHEMA_VERSION = 1
_EXPECTED_DECISIONS = {
    "scaling": "promote_to_full_imagenet256",
    "full": "large_scale_generation_ready",
}


def build_generation_gate_binding(
    gate: dict[str, Any],
    *,
    expected_stage: str,
    gate_path: str,
    gate_bytes: int,
    gate_sha256: str,
) -> dict[str, Any]:
    validation = validate_generation_gate_authorization(
        gate,
        expected_stage=expected_stage,
    )
    if gate_bytes < 1:
        raise ValueError("generation authorization gate is empty")
    if len(gate_sha256) != 64:
        raise ValueError("generation authorization gate SHA256 is malformed")
    return {
        "schema_version": GENERATION_GATE_BINDING_SCHEMA_VERSION,
        "status": "pass",
        "stage": validation["stage"],
        "decision": validation["decision"],
        "gate_path": Path(gate_path).resolve().as_posix(),
        "gate_bytes": int(gate_bytes),
        "gate_sha256": gate_sha256,
        "gate_identity_sha256": generation_gate_identity_sha256(gate),
        "validated_thresholds": validation["validated_thresholds"],
    }


def capture_generation_gate_binding(
    gate_path: str | Path,
    *,
    expected_stage: str,
) -> dict[str, Any]:
    path = Path(gate_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    return build_generation_gate_binding(
        gate,
        expected_stage=expected_stage,
        gate_path=path.as_posix(),
        gate_bytes=path.stat().st_size,
        gate_sha256=file_sha256(path),
    )


def validate_generation_gate_binding(
    authorization: Mapping[str, Any],
    *,
    expected_stage: str,
    expected_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if expected_stage not in _EXPECTED_DECISIONS:
        raise ValueError(f"unsupported generation authorization stage: {expected_stage}")
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
            "generation authorization must contain exactly: "
            + ", ".join(sorted(required))
        )
    if (
        int(authorization["schema_version"])
        != GENERATION_GATE_BINDING_SCHEMA_VERSION
        or authorization["status"] != "pass"
        or authorization["stage"] != expected_stage
        or authorization["decision"] != _EXPECTED_DECISIONS[expected_stage]
    ):
        raise ValueError("generation authorization identity is invalid")
    gate_path = str(authorization["gate_path"])
    if not gate_path or not Path(gate_path).is_absolute():
        raise ValueError("generation authorization gate path is not absolute")
    if int(authorization["gate_bytes"]) < 1:
        raise ValueError("generation authorization gate byte count is invalid")
    for name in ("gate_sha256", "gate_identity_sha256"):
        if len(str(authorization[name])) != 64:
            raise ValueError(f"generation authorization {name} is malformed")
    thresholds = authorization["validated_thresholds"]
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("generation authorization thresholds are missing")

    if expected_gate is not None:
        validation = validate_generation_gate_authorization(
            expected_gate,
            expected_stage=expected_stage,
        )
        if authorization["gate_identity_sha256"] != generation_gate_identity_sha256(
            expected_gate
        ):
            raise ValueError("generation authorization binds another gate")
        if dict(thresholds) != validation["validated_thresholds"]:
            raise ValueError("generation authorization thresholds differ from the gate")
    return {
        "schema_version": GENERATION_GATE_BINDING_SCHEMA_VERSION,
        "stage": authorization["stage"],
        "decision": authorization["decision"],
        "gate_path": gate_path,
        "gate_identity_sha256": authorization["gate_identity_sha256"],
        "gate_sha256": authorization["gate_sha256"],
        "gate_bytes": int(authorization["gate_bytes"]),
    }
