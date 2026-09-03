"""Shared CLI loading for the fresh four-arm capacity-screen preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_screen import ARM_NAMES, CAPACITY_NAMES
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)


def add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--decision-validation", type=Path, required=True)
    parser.add_argument("--expected-decision-validation-sha256", required=True)
    for arm in ARM_NAMES:
        flag = arm.replace("_", "-")
        parser.add_argument(f"--{flag}-config", type=Path, required=True)
        parser.add_argument(f"--expected-{flag}-config-sha256", required=True)
    for capacity in CAPACITY_NAMES:
        parser.add_argument(
            f"--{capacity}-pair-validation", type=Path, required=True
        )
        parser.add_argument(
            f"--expected-{capacity}-pair-validation-sha256", required=True
        )
    parser.add_argument("--preparation-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)


def parse_build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the source-bound fresh four-arm capacity-screen preparation."
    )
    parser.add_argument("--preparation", type=Path, required=True)
    add_sources(parser)
    return parser.parse_args()


def parse_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild and validate the capacity-screen preparation."
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    add_sources(parser)
    return parser.parse_args()


def _stable_load(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = identity(path)
    if before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = read_object(path, name=name)
    after = identity(path)
    if after != before:
        raise ValueError(f"{name} changed while it was being read")
    return payload, before


def preparation_kwargs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision, decision_id = _stable_load(
        args.decision,
        expected_sha256=args.expected_decision_sha256,
        name="exposure/capacity scientific decision",
    )
    decision_validation, decision_validation_id = _stable_load(
        args.decision_validation,
        expected_sha256=args.expected_decision_validation_sha256,
        name="exposure/capacity decision validation",
    )
    configs: dict[str, dict[str, Any]] = {}
    config_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        argument = f"{arm}_config"
        config, descriptor = _stable_load(
            getattr(args, argument),
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
            name=f"{arm} capacity-screen config",
        )
        configs[arm] = config
        config_ids[arm] = descriptor
    pair_validations: dict[str, dict[str, Any]] = {}
    pair_ids: dict[str, dict[str, Any]] = {}
    for capacity in CAPACITY_NAMES:
        argument = f"{capacity}_pair_validation"
        report, descriptor = _stable_load(
            getattr(args, argument),
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
            name=f"{capacity} matched config validation",
        )
        pair_validations[capacity] = report
        pair_ids[capacity] = descriptor
    kwargs = {
        "decision": decision,
        "decision_identity": decision_id,
        "decision_validation": decision_validation,
        "decision_validation_identity": decision_validation_id,
        "configs": configs,
        "config_identities": config_ids,
        "pair_validations": pair_validations,
        "pair_validation_identities": pair_ids,
        "preparation_git": checkout_identity(args.preparation_project_root),
        "output_root": args.output_root,
    }
    sources = {
        "decision": decision_id,
        "decision_validation": decision_validation_id,
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
