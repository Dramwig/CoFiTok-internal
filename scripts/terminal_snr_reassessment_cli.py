"""Shared source loading for terminal-SNR objective reassessment CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)


SOURCE_ARGUMENTS = {
    "hold_decision": "capacity hold decision",
    "hold_validation": "capacity hold validation",
    "sampling_recovery_result": "sampling recovery result",
    "cofitok_control_config": "CoFiTok control config",
    "dense_control_config": "dense control config",
    "cofitok_intervention_config": "CoFiTok intervention config",
    "dense_intervention_config": "dense intervention config",
}


def _option(name: str) -> str:
    return name.replace("_", "-")


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    for argument in SOURCE_ARGUMENTS:
        option = _option(argument)
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--expected-{option}-sha256", required=True)
    parser.add_argument("--decision-project-root", type=Path, required=True)
    parser.add_argument(
        "--allowed-sampling-output-prefix",
        default="/root/autodl-tmp/CoFiTok/checkpoints/generation/",
    )


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-authorizing terminal-SNR objective reassessment."
    )
    parser.add_argument("--decision", type=Path, required=True)
    add_source_arguments(parser)
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically replay a terminal-SNR objective reassessment."
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
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = identity(path)
    if before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = read_object(path, name=name)
    after = identity(path)
    if after != before:
        raise ValueError(f"{name} changed while it was being read")
    return payload, before


def reassessment_kwargs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for argument, name in SOURCE_ARGUMENTS.items():
        payload, descriptor = _load_stable(
            getattr(args, argument),
            name=name,
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
        )
        payloads[argument] = payload
        identities[argument] = descriptor

    kwargs = {
        "hold_decision": payloads["hold_decision"],
        "hold_decision_identity": identities["hold_decision"],
        "hold_validation": payloads["hold_validation"],
        "hold_validation_identity": identities["hold_validation"],
        "sampling_recovery_result": payloads["sampling_recovery_result"],
        "sampling_recovery_identity": identities["sampling_recovery_result"],
        "control_configs": {
            "cofitok": payloads["cofitok_control_config"],
            "dense_identity": payloads["dense_control_config"],
        },
        "control_config_identities": {
            "cofitok": identities["cofitok_control_config"],
            "dense_identity": identities["dense_control_config"],
        },
        "intervention_configs": {
            "cofitok": payloads["cofitok_intervention_config"],
            "dense_identity": payloads["dense_intervention_config"],
        },
        "intervention_config_identities": {
            "cofitok": identities["cofitok_intervention_config"],
            "dense_identity": identities["dense_intervention_config"],
        },
        "decision_git": checkout_identity(args.decision_project_root),
        "allowed_sampling_output_prefix": args.allowed_sampling_output_prefix,
    }
    return kwargs, identities


__all__ = [
    "SOURCE_ARGUMENTS",
    "add_source_arguments",
    "parse_build_args",
    "parse_validate_args",
    "reassessment_kwargs",
]
