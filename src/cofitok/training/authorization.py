from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation_authorization import (
    GENERATION_GATE_BINDING_SCHEMA_VERSION,
    build_generation_gate_binding,
    capture_generation_gate_binding,
    validate_generation_gate_binding,
)
from cofitok.reporting import file_sha256


GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION = (
    GENERATION_GATE_BINDING_SCHEMA_VERSION
)
GENERATION_TRAINING_AUTHORIZATION_STAGE = "scaling"
GENERATION_TRAINING_AUTHORIZATION_DECISION = "promote_to_full_imagenet256"
CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE = "capacity_full_experimental"
CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION = (
    "authorize_fresh_matched_300k_training"
)
CAPACITY_FULL_TRAINING_LAUNCH_ROLE = (
    "stability_capacity_full_300k_training_launch_receipt"
)
SUPPORTED_GENERATION_TRAINING_AUTHORIZATION_IDENTITIES = frozenset(
    {
        (
            GENERATION_TRAINING_AUTHORIZATION_STAGE,
            GENERATION_TRAINING_AUTHORIZATION_DECISION,
        ),
        (
            CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE,
            CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION,
        ),
    }
)


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


def build_capacity_full_training_authorization(
    receipt: dict[str, Any],
    *,
    receipt_path: str,
    receipt_bytes: int,
    receipt_sha256: str,
) -> dict[str, Any]:
    from cofitok.generation.capacity_full_training_launch import (
        capacity_full_training_authorization_thresholds,
        capacity_full_training_launch_identity_sha256,
        validate_capacity_full_training_launch_self_bound,
    )

    validate_capacity_full_training_launch_self_bound(receipt)
    if receipt_bytes < 1:
        raise ValueError("capacity-full training authorization receipt is empty")
    if len(receipt_sha256) != 64:
        raise ValueError("capacity-full training authorization SHA256 is malformed")
    path = Path(receipt_path)
    if not path.is_absolute() and not PurePosixPath(receipt_path).is_absolute():
        raise ValueError("capacity-full training authorization path is not absolute")
    return {
        "schema_version": GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
        "status": "pass",
        "stage": CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE,
        "decision": CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION,
        "gate_path": path.as_posix(),
        "gate_bytes": int(receipt_bytes),
        "gate_sha256": receipt_sha256,
        "gate_identity_sha256": capacity_full_training_launch_identity_sha256(
            receipt
        ),
        "validated_thresholds": capacity_full_training_authorization_thresholds(
            receipt
        ),
    }


def capture_generation_training_authorization(
    gate_path: str | Path,
) -> dict[str, Any]:
    path = Path(gate_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    if gate.get("role") == CAPACITY_FULL_TRAINING_LAUNCH_ROLE:
        from cofitok.generation.capacity_full_training_launch import (
            validate_capacity_full_training_launch_self_bound,
        )

        evidence = validate_capacity_full_training_launch_self_bound(gate)
        for name, identity in evidence["source_reports"].items():
            source = Path(str(identity["path"]))
            if (
                not source.is_file()
                or source.stat().st_size != int(identity["bytes"])
                or file_sha256(source) != identity["sha256"]
            ):
                raise ValueError(
                    "capacity-full training authorization source changed: " + name
                )
        return build_capacity_full_training_authorization(
            gate,
            receipt_path=path.as_posix(),
            receipt_bytes=path.stat().st_size,
            receipt_sha256=file_sha256(path),
        )
    return capture_generation_gate_binding(
        path,
        expected_stage=GENERATION_TRAINING_AUTHORIZATION_STAGE,
    )


def validate_generation_training_authorization(
    authorization: Mapping[str, Any],
    *,
    expected_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if authorization.get("stage") == CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE:
        from cofitok.generation.capacity_full_training_launch import (
            capacity_full_training_authorization_thresholds,
            capacity_full_training_launch_identity_sha256,
            validate_capacity_full_training_launch_self_bound,
        )

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
        gate_path = str(authorization["gate_path"])
        thresholds = authorization["validated_thresholds"]
        if (
            int(authorization["schema_version"])
            != GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION
            or authorization["status"] != "pass"
            or authorization["decision"]
            != CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION
            or not gate_path
            or (
                not Path(gate_path).is_absolute()
                and not PurePosixPath(gate_path).is_absolute()
            )
            or int(authorization["gate_bytes"]) < 1
            or len(str(authorization["gate_sha256"])) != 64
            or len(str(authorization["gate_identity_sha256"])) != 64
            or not isinstance(thresholds, Mapping)
        ):
            raise ValueError("capacity-full training authorization identity is invalid")
        expected_static = {
            "target_start_step": 0,
            "target_steps": 300_000,
            "effective_batch_size": 64,
            "capacity_100k_checkpoint_resume_allowed": False,
            "formal_generation_claim_allowed": False,
            "release_authorization_allowed": False,
        }
        if any(thresholds.get(key) != value for key, value in expected_static.items()):
            raise ValueError("capacity-full training authorization boundary differs")
        micro_batch = thresholds.get("micro_batch_size")
        accumulation = thresholds.get("gradient_accumulation_steps")
        if (
            type(micro_batch) is not int
            or micro_batch < 1
            or type(accumulation) is not int
            or accumulation < 1
            or micro_batch * accumulation != 64
            or set(thresholds)
            != set(expected_static) | {"micro_batch_size", "gradient_accumulation_steps"}
        ):
            raise ValueError("capacity-full training runtime authorization differs")
        if expected_gate is not None:
            if expected_gate.get("role") != CAPACITY_FULL_TRAINING_LAUNCH_ROLE:
                raise ValueError("capacity-full authorization binds another source type")
            validate_capacity_full_training_launch_self_bound(expected_gate)
            if (
                authorization["gate_identity_sha256"]
                != capacity_full_training_launch_identity_sha256(expected_gate)
                or dict(thresholds)
                != capacity_full_training_authorization_thresholds(expected_gate)
            ):
                raise ValueError("capacity-full authorization binds another receipt")
        return {
            "schema_version": GENERATION_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
            "stage": authorization["stage"],
            "decision": authorization["decision"],
            "gate_path": gate_path,
            "gate_identity_sha256": authorization["gate_identity_sha256"],
            "gate_sha256": authorization["gate_sha256"],
            "gate_bytes": int(authorization["gate_bytes"]),
        }
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
            "Formal full checkpoint lacks its scaling-gate authorization or "
            "capacity-full training-authorization binding"
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
