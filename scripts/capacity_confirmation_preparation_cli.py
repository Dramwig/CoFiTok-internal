"""Shared CLI loading for capacity-confirmation preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_screen import ARM_NAMES
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)


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


def add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--expected-screen-result-sha256", required=True)
    parser.add_argument("--screen-launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-screen-launch-receipt-sha256", required=True)
    for arm in ARM_NAMES:
        option = arm.replace("_", "-")
        parser.add_argument(f"--{option}-screen-validation", type=Path, required=True)
        parser.add_argument(
            f"--expected-{option}-screen-validation-sha256", required=True
        )
    parser.add_argument("--preparation-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen four-arm 10K capacity confirmation preparation."
    )
    parser.add_argument("--preparation", type=Path, required=True)
    add_sources(parser)
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild and validate capacity confirmation preparation."
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    add_sources(parser)
    return parser.parse_args(argv)


def preparation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    result, result_id = _stable_load(
        args.screen_result,
        expected_sha256=args.expected_screen_result_sha256,
        name="capacity screen result",
    )
    launch, launch_id = _stable_load(
        args.screen_launch_receipt,
        expected_sha256=args.expected_screen_launch_receipt_sha256,
        name="capacity screen launch receipt",
    )
    arms: dict[str, dict[str, Any]] = {}
    arm_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        argument = f"{arm}_screen_validation"
        report, descriptor = _stable_load(
            getattr(args, argument),
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
            name=f"{arm} capacity screen validation",
        )
        arms[arm] = report
        arm_ids[arm] = descriptor
    return {
        "screen_result": result,
        "screen_result_identity": result_id,
        "screen_launch_receipt": launch,
        "screen_launch_receipt_identity": launch_id,
        "screen_arm_validations": arms,
        "screen_arm_validation_identities": arm_ids,
        "preparation_git": checkout_identity(args.preparation_project_root),
        "output_root": args.output_root,
    }


__all__ = [
    "add_sources",
    "parse_build_args",
    "parse_validate_args",
    "preparation_kwargs",
]
