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
from cofitok.reporting import file_sha256


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
    parser = argparse.ArgumentParser(description="Build terminal-SNR screen result.")
    add_result_sources(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay terminal-SNR screen result.")
    add_result_sources(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--validator-project-root", type=Path)
    return parser.parse_args(argv)


def _read_bound(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    return read_object(path, name=name)


def result_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preparation = _read_bound(
        args.preparation,
        args.expected_preparation_sha256,
        "terminal-SNR preparation",
    )
    launch = _read_bound(
        args.launch_receipt,
        args.expected_launch_receipt_sha256,
        "terminal-SNR launch receipt",
    )
    reports: dict[str, dict[str, Any]] = {}
    report_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        path = getattr(args, f"{arm}_validation")
        expected = getattr(args, f"expected_{arm}_validation_sha256")
        reports[arm] = _read_bound(path, expected, f"{arm} validation")
        report_ids[arm] = identity(path)
    return {
        "preparation": preparation,
        "preparation_identity": identity(args.preparation),
        "launch_receipt": launch,
        "launch_receipt_identity": identity(args.launch_receipt),
        "arm_validations": reports,
        "arm_validation_identities": report_ids,
        "result_git": checkout_identity(args.project_root),
    }


__all__ = ["parse_build_args", "parse_validate_args", "result_kwargs"]
