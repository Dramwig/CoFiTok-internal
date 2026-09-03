from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_screen import ARM_NAMES


def add_arm_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--sampling-report", type=Path, required=True)
    parser.add_argument("--metrics-report", type=Path, required=True)
    parser.add_argument("--class-fidelity-report", type=Path, required=True)
    parser.add_argument("--checkpoint-evaluation-report", type=Path, required=True)
    parser.add_argument("--rollout-report", type=Path, required=True)


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one physical capacity-screen arm validation report."
    )
    add_arm_source_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one physical capacity-screen arm validation report."
    )
    add_arm_source_arguments(parser)
    parser.add_argument("--arm-validation", type=Path, required=True)
    parser.add_argument("--expected-arm-validation-sha256", required=True)
    return parser.parse_args(argv)


def arm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arm": args.arm,
        "launch_receipt_path": args.launch_receipt,
        "expected_launch_receipt_sha256": args.expected_launch_receipt_sha256,
        "config_path": args.config,
        "training_report_path": args.training_report,
        "sampling_report_path": args.sampling_report,
        "metrics_report_path": args.metrics_report,
        "class_fidelity_report_path": args.class_fidelity_report,
        "checkpoint_evaluation_report_path": args.checkpoint_evaluation_report,
        "rollout_report_path": args.rollout_report,
    }


__all__ = [
    "add_arm_source_arguments",
    "arm_kwargs",
    "parse_build_args",
    "parse_validate_args",
]
