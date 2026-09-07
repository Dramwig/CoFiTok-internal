"""Shared CLI loading for the source-bound capacity-screen hold recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_screen import ARM_NAMES
from cofitok.generation.capacity_screen_hold_recovery import ROLLOUT_NAMES
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)


PRIMARY_SOURCES = {
    "exposure_result": "exposure result",
    "exposure_result_validation": "exposure result validation",
    "exposure_decision": "exposure decision",
    "exposure_decision_validation": "exposure decision validation",
    "capacity_result": "capacity screen result",
    "capacity_result_validation": "capacity screen result validation",
    "sampling_recovery_result": "sampling recovery result",
    "min_snr_result": "Min-SNR result",
    "min_snr_guard": "Min-SNR physical guard",
}


def _option(name: str) -> str:
    return name.replace("_", "-")


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    for argument in PRIMARY_SOURCES:
        option = _option(argument)
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--expected-{option}-sha256", required=True)
    for arm in ARM_NAMES:
        parser.add_argument(
            f"--{_option(arm)}-validation", type=Path, required=True
        )
    for name in ROLLOUT_NAMES:
        parser.add_argument(f"--{_option(name)}-rollout", type=Path, required=True)
    parser.add_argument("--decision-project-root", type=Path, required=True)
    parser.add_argument(
        "--allowed-sampling-output-prefix",
        default="/root/autodl-tmp/CoFiTok/checkpoints/generation/",
    )


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the non-authorizing decision after a capacity-screen hold."
    )
    parser.add_argument("--decision", type=Path, required=True)
    add_source_arguments(parser)
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically replay a source-bound capacity-screen hold decision."
    )
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--validator-project-root", type=Path)
    add_source_arguments(parser)
    return parser.parse_args(argv)


def _load_stable(
    path: Path,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = identity(path)
    if expected_sha256 is not None and before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = read_object(path, name=name)
    after = identity(path)
    if after != before:
        raise ValueError(f"{name} changed while it was being read")
    return payload, before


def decision_kwargs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for argument, name in PRIMARY_SOURCES.items():
        payload, descriptor = _load_stable(
            getattr(args, argument),
            name=name,
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
        )
        payloads[argument] = payload
        identities[argument] = descriptor

    arms: dict[str, dict[str, Any]] = {}
    arm_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        payload, descriptor = _load_stable(
            getattr(args, f"{arm}_validation"),
            name=f"{arm} capacity validation",
        )
        arms[arm] = payload
        arm_ids[arm] = descriptor
    identities["capacity_arm_validations"] = arm_ids

    rollouts: dict[str, dict[str, Any]] = {}
    rollout_ids: dict[str, dict[str, Any]] = {}
    for name in ROLLOUT_NAMES:
        payload, descriptor = _load_stable(
            getattr(args, f"{name}_rollout"),
            name=f"{name} rollout",
        )
        rollouts[name] = payload
        rollout_ids[name] = descriptor
    identities["rollouts"] = rollout_ids

    kwargs = {
        "exposure_result": payloads["exposure_result"],
        "exposure_result_identity": identities["exposure_result"],
        "exposure_result_validation": payloads["exposure_result_validation"],
        "exposure_result_validation_identity": identities[
            "exposure_result_validation"
        ],
        "exposure_decision": payloads["exposure_decision"],
        "exposure_decision_identity": identities["exposure_decision"],
        "exposure_decision_validation": payloads[
            "exposure_decision_validation"
        ],
        "exposure_decision_validation_identity": identities[
            "exposure_decision_validation"
        ],
        "capacity_result": payloads["capacity_result"],
        "capacity_result_identity": identities["capacity_result"],
        "capacity_result_validation": payloads["capacity_result_validation"],
        "capacity_result_validation_identity": identities[
            "capacity_result_validation"
        ],
        "capacity_arm_validations": arms,
        "capacity_arm_validation_identities": arm_ids,
        "sampling_recovery_result": payloads["sampling_recovery_result"],
        "sampling_recovery_identity": identities["sampling_recovery_result"],
        "min_snr_result": payloads["min_snr_result"],
        "min_snr_result_identity": identities["min_snr_result"],
        "min_snr_guard": payloads["min_snr_guard"],
        "min_snr_guard_identity": identities["min_snr_guard"],
        "rollout_reports": rollouts,
        "rollout_identities": rollout_ids,
        "decision_git": checkout_identity(args.decision_project_root),
        "allowed_sampling_output_prefix": args.allowed_sampling_output_prefix,
    }
    return kwargs, identities


__all__ = [
    "PRIMARY_SOURCES",
    "add_source_arguments",
    "decision_kwargs",
    "parse_build_args",
    "parse_validate_args",
]
