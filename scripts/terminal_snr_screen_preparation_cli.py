"""CLI source loading for terminal-SNR screen preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.generation.terminal_snr_screen import ARM_NAMES, CONDITION_NAMES


def _stable_load(
    path: Path, *, expected_sha256: str, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = identity(path)
    if before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = read_object(path, name=name)
    after = identity(path)
    if after != before:
        raise ValueError(f"{name} changed while it was being read")
    return payload, before


def add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reassessment", type=Path, required=True)
    parser.add_argument("--expected-reassessment-sha256", required=True)
    parser.add_argument("--reassessment-validation", type=Path, required=True)
    parser.add_argument("--expected-reassessment-validation-sha256", required=True)
    for arm in ARM_NAMES:
        flag = arm.replace("_", "-")
        parser.add_argument(f"--{flag}-config", type=Path, required=True)
        parser.add_argument(f"--expected-{flag}-config-sha256", required=True)
    for condition in CONDITION_NAMES:
        flag = condition.replace("_", "-")
        parser.add_argument(f"--{flag}-pair-validation", type=Path, required=True)
        parser.add_argument(
            f"--expected-{flag}-pair-validation-sha256", required=True
        )
    parser.add_argument("--preparation-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-authorizing terminal-SNR screen preparation."
    )
    parser.add_argument("--preparation", type=Path, required=True)
    add_sources(parser)
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild and validate a terminal-SNR screen preparation."
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    add_sources(parser)
    return parser.parse_args(argv)


def preparation_kwargs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reassessment, reassessment_id = _stable_load(
        args.reassessment,
        expected_sha256=args.expected_reassessment_sha256,
        name="terminal-SNR reassessment",
    )
    validation, validation_id = _stable_load(
        args.reassessment_validation,
        expected_sha256=args.expected_reassessment_validation_sha256,
        name="terminal-SNR reassessment validation",
    )
    configs: dict[str, dict[str, Any]] = {}
    config_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        argument = f"{arm}_config"
        config, descriptor = _stable_load(
            getattr(args, argument),
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
            name=f"{arm} terminal-SNR config",
        )
        configs[arm] = config
        config_ids[arm] = descriptor
    pairs: dict[str, dict[str, Any]] = {}
    pair_ids: dict[str, dict[str, Any]] = {}
    for condition in CONDITION_NAMES:
        argument = f"{condition}_pair_validation"
        pair, descriptor = _stable_load(
            getattr(args, argument),
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
            name=f"{condition} matched config validation",
        )
        pairs[condition] = pair
        pair_ids[condition] = descriptor
    kwargs = {
        "reassessment": reassessment,
        "reassessment_identity": reassessment_id,
        "reassessment_validation": validation,
        "reassessment_validation_identity": validation_id,
        "configs": configs,
        "config_identities": config_ids,
        "pair_validations": pairs,
        "pair_validation_identities": pair_ids,
        "preparation_git": checkout_identity(args.preparation_project_root),
        "output_root": args.output_root,
    }
    sources = {
        "reassessment": reassessment_id,
        "reassessment_validation": validation_id,
        "configs": config_ids,
        "pair_validations": pair_ids,
    }
    return kwargs, sources


__all__ = [
    "add_sources",
    "parse_build_args",
    "parse_validate_args",
    "preparation_kwargs",
]
