"""Shared CLI loaders for terminal-SNR confirmation authorization."""

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


def parse_build_stage_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build user terminal-SNR confirmation stage authorization."
    )
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--source-instruction", required=True)
    return parser.parse_args(argv)


def parse_validate_stage_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate terminal-SNR confirmation stage authorization."
    )
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--expected-stage-authorization-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def stage_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    _, preparation_id = _stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="terminal-SNR confirmation preparation",
    )
    return {
        "preparation_identity": preparation_id,
        "execution_checkout": checkout_identity(args.execution_project_root),
        "output_root": args.output_root,
    }


def add_authorization_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--expected-stage-authorization-sha256", required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)


def parse_build_authorization_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build terminal-SNR confirmation execution authorization."
    )
    parser.add_argument("--authorization", type=Path, required=True)
    add_authorization_sources(parser)
    return parser.parse_args(argv)


def parse_validate_authorization_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay terminal-SNR confirmation execution authorization."
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    add_authorization_sources(parser)
    return parser.parse_args(argv)


def authorization_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preparation, preparation_id = _stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="terminal-SNR confirmation preparation",
    )
    stage, stage_id = _stable_load(
        args.stage_authorization,
        expected_sha256=args.expected_stage_authorization_sha256,
        name="terminal-SNR confirmation stage authorization",
    )
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "stage_authorization": stage,
        "stage_authorization_identity": stage_id,
        "execution_checkout": checkout_identity(args.execution_project_root),
        "output_root": args.output_root,
    }


def add_launch_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--expected-storage-capacity-sha256", required=True)
    parser.add_argument("--live-snapshot", type=Path, required=True)
    parser.add_argument("--expected-live-snapshot-sha256", required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execution-lock", required=True)
    for arm in ARM_NAMES:
        parser.add_argument(f"--{arm.replace('_', '-')}-output-dir", required=True)


def parse_build_launch_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable terminal-SNR confirmation launch receipt."
    )
    parser.add_argument("--launch-receipt", type=Path, required=True)
    add_launch_sources(parser)
    return parser.parse_args(argv)


def parse_validate_launch_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay immutable terminal-SNR confirmation launch receipt."
    )
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    add_launch_sources(parser)
    return parser.parse_args(argv)


def launch_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preparation, preparation_id = _stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="terminal-SNR confirmation preparation",
    )
    authorization, authorization_id = _stable_load(
        args.authorization,
        expected_sha256=args.expected_authorization_sha256,
        name="terminal-SNR confirmation authorization",
    )
    storage, storage_id = _stable_load(
        args.storage_capacity,
        expected_sha256=args.expected_storage_capacity_sha256,
        name="terminal-SNR confirmation storage",
    )
    live, live_id = _stable_load(
        args.live_snapshot,
        expected_sha256=args.expected_live_snapshot_sha256,
        name="terminal-SNR confirmation live snapshot",
    )
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "execution_authorization": authorization,
        "execution_authorization_identity": authorization_id,
        "storage_capacity": storage,
        "storage_capacity_identity": storage_id,
        "live_snapshot": live,
        "live_snapshot_identity": live_id,
        "execution_checkout": checkout_identity(args.execution_project_root),
        "output_root": args.output_root,
        "output_dirs": {
            arm: getattr(args, f"{arm}_output_dir") for arm in ARM_NAMES
        },
        "execution_lock": args.execution_lock,
        "evaluation_state_absent_at_launch": True,
    }


__all__ = [
    "authorization_kwargs",
    "launch_kwargs",
    "parse_build_authorization_args",
    "parse_build_launch_args",
    "parse_build_stage_args",
    "parse_validate_authorization_args",
    "parse_validate_launch_args",
    "parse_validate_stage_args",
    "stage_kwargs",
]
