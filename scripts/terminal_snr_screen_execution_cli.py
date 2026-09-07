"""Stable CLI loading for terminal-SNR screen authorization evidence."""

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


def stable_load(
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
        description="Build exact terminal-SNR screen stage authorization."
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
        description="Validate terminal-SNR screen stage authorization."
    )
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--expected-stage-authorization-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def stage_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    _, preparation_id = stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="terminal-SNR screen preparation",
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
    for arm in ARM_NAMES:
        flag = arm.replace("_", "-")
        parser.add_argument(f"--{flag}-config", type=Path, required=True)
        parser.add_argument(f"--expected-{flag}-config-sha256", required=True)


def parse_build_authorization_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact terminal-SNR screen execution authorization."
    )
    parser.add_argument("--authorization", type=Path, required=True)
    add_authorization_sources(parser)
    return parser.parse_args(argv)


def parse_validate_authorization_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild terminal-SNR screen execution authorization."
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    add_authorization_sources(parser)
    return parser.parse_args(argv)


def authorization_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preparation, preparation_id = stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="terminal-SNR screen preparation",
    )
    stage, stage_id = stable_load(
        args.stage_authorization,
        expected_sha256=args.expected_stage_authorization_sha256,
        name="terminal-SNR screen stage authorization",
    )
    config_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        argument = f"{arm}_config"
        _, descriptor = stable_load(
            getattr(args, argument),
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
            name=f"{arm} execution config",
        )
        config_ids[arm] = descriptor
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "stage_authorization": stage,
        "stage_authorization_identity": stage_id,
        "execution_checkout": checkout_identity(args.execution_project_root),
        "config_identities": config_ids,
        "output_root": args.output_root,
    }


def add_launch_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--expected-runtime-selection-sha256", required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--expected-storage-capacity-sha256", required=True)
    parser.add_argument("--live-snapshot", type=Path, required=True)
    parser.add_argument("--expected-live-snapshot-sha256", required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--benchmark-root", required=True)
    for arm in ARM_NAMES:
        flag = arm.replace("_", "-")
        parser.add_argument(f"--{flag}-config", type=Path, required=True)
        parser.add_argument(f"--expected-{flag}-config-sha256", required=True)
        parser.add_argument(f"--{flag}-run-dir", required=True)


def parse_build_launch_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable terminal-SNR screen launch receipt."
    )
    parser.add_argument("--launch-receipt", type=Path, required=True)
    add_launch_sources(parser)
    return parser.parse_args(argv)


def parse_validate_launch_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild terminal-SNR screen launch receipt."
    )
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    add_launch_sources(parser)
    return parser.parse_args(argv)


def launch_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preparation, preparation_id = stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="terminal-SNR screen preparation",
    )
    authorization, authorization_id = stable_load(
        args.authorization,
        expected_sha256=args.expected_authorization_sha256,
        name="terminal-SNR screen execution authorization",
    )
    runtime, runtime_id = stable_load(
        args.runtime_selection,
        expected_sha256=args.expected_runtime_selection_sha256,
        name="terminal-SNR runtime selection",
    )
    storage, storage_id = stable_load(
        args.storage_capacity,
        expected_sha256=args.expected_storage_capacity_sha256,
        name="terminal-SNR storage capacity",
    )
    live, live_id = stable_load(
        args.live_snapshot,
        expected_sha256=args.expected_live_snapshot_sha256,
        name="terminal-SNR live snapshot",
    )
    config_ids: dict[str, dict[str, Any]] = {}
    run_dirs: dict[str, str] = {}
    for arm in ARM_NAMES:
        argument = f"{arm}_config"
        _, descriptor = stable_load(
            getattr(args, argument),
            expected_sha256=getattr(args, f"expected_{argument}_sha256"),
            name=f"{arm} launch config",
        )
        config_ids[arm] = descriptor
        run_dirs[arm] = getattr(args, f"{arm}_run_dir")
    output_root = Path(args.output_root)
    training_state_absent = not output_root.exists() and all(
        not Path(path).exists() for path in run_dirs.values()
    )
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "execution_authorization": authorization,
        "execution_authorization_identity": authorization_id,
        "runtime_selection": runtime,
        "runtime_selection_identity": runtime_id,
        "storage_capacity": storage,
        "storage_capacity_identity": storage_id,
        "live_snapshot": live,
        "live_snapshot_identity": live_id,
        "config_identities": config_ids,
        "execution_checkout": checkout_identity(args.execution_project_root),
        "output_root": args.output_root,
        "execution_lock": args.execution_lock,
        "run_dirs": run_dirs,
        "benchmark_root": args.benchmark_root,
        "training_state_absent_at_launch": training_state_absent,
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
    "stable_load",
    "stage_kwargs",
]
