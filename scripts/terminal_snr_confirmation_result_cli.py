from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.generation.terminal_snr_screen import ARM_NAMES


def add_result_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    for arm in ARM_NAMES:
        option = arm.replace("_", "-")
        parser.add_argument(f"--{option}-validation", type=Path, required=True)
        parser.add_argument(f"--expected-{option}-validation-sha256", required=True)


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build terminal-SNR frozen four-arm 10K confirmation result."
    )
    add_result_sources(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay terminal-SNR frozen confirmation result."
    )
    add_result_sources(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--validator-project-root", type=Path)
    return parser.parse_args(argv)


def _stable_load(
    path: Path, expected_sha256: str, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = identity(path)
    if before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = read_object(path, name=name)
    after = identity(path)
    if after != before:
        raise ValueError(f"{name} changed while it was being read")
    return payload, before


def result_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preparation, preparation_id = _stable_load(
        args.preparation,
        args.expected_preparation_sha256,
        "terminal-SNR confirmation preparation",
    )
    launch, launch_id = _stable_load(
        args.launch_receipt,
        args.expected_launch_receipt_sha256,
        "terminal-SNR confirmation launch receipt",
    )
    reports: dict[str, dict[str, Any]] = {}
    report_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        path = getattr(args, f"{arm}_validation")
        expected = getattr(args, f"expected_{arm}_validation_sha256")
        reports[arm], report_ids[arm] = _stable_load(
            path, expected, f"{arm} terminal-SNR confirmation validation"
        )
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "launch_receipt": launch,
        "launch_receipt_identity": launch_id,
        "arm_validations": reports,
        "arm_validation_identities": report_ids,
        "result_git": checkout_identity(args.project_root),
    }


__all__ = [
    "add_result_sources",
    "parse_build_args",
    "parse_validate_args",
    "result_kwargs",
]
