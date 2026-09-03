"""Shared CLI helpers for the source-bound exposure/capacity decision."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.generation.exposure_capacity_decision import SOURCE_REPORT_NAMES


PRIMARY_SOURCES = {
    "exposure_result": ("exposure_result", "exposure result"),
    "validation_receipt": (
        "exposure_result_validation_receipt",
        "exposure result validation receipt",
    ),
    "preparation": ("preparation", "exposure/capacity preparation"),
    "quality_bridge_result": (
        "quality_bridge_result",
        "100K quality-bridge result",
    ),
}

RAW_SOURCE_ARGUMENTS = {
    "cofitok_training": "cofitok_training_report",
    "dense_training": "dense_training_report",
    "cofitok_checkpoint": "cofitok_checkpoint_report",
    "dense_checkpoint": "dense_checkpoint_report",
    "cofitok_rollout": "cofitok_rollout_report",
    "dense_rollout": "dense_rollout_report",
}


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exposure-result", type=Path, required=True)
    parser.add_argument("--expected-exposure-result-sha256", required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--expected-validation-receipt-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--quality-bridge-result", type=Path, required=True)
    parser.add_argument("--expected-quality-bridge-result-sha256", required=True)
    parser.add_argument("--cofitok-training-report", type=Path, required=True)
    parser.add_argument("--dense-training-report", type=Path, required=True)
    parser.add_argument("--cofitok-checkpoint-report", type=Path, required=True)
    parser.add_argument("--dense-checkpoint-report", type=Path, required=True)
    parser.add_argument("--cofitok-rollout-report", type=Path, required=True)
    parser.add_argument("--dense-rollout-report", type=Path, required=True)
    parser.add_argument("--decision-project-root", type=Path, required=True)


def parse_build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the non-authorizing scientific route after the bounded "
            "110K exposure continuation."
        )
    )
    parser.add_argument("--decision", type=Path, required=True)
    add_source_arguments(parser)
    return parser.parse_args()


def parse_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rehash, rebuild, and validate the source-bound exposure/capacity "
            "scientific decision."
        )
    )
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--output", type=Path)
    add_source_arguments(parser)
    return parser.parse_args()


def _load_stable_source(
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
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, Any]] = {}
    source_identities: dict[str, dict[str, Any]] = {}
    for argument, (keyword, name) in PRIMARY_SOURCES.items():
        expected_name = f"expected_{argument}_sha256"
        payload, descriptor = _load_stable_source(
            getattr(args, argument),
            name=name,
            expected_sha256=getattr(args, expected_name),
        )
        loaded[keyword] = payload
        source_identities[keyword] = descriptor

    raw_reports: dict[str, dict[str, Any]] = {}
    raw_identities: dict[str, dict[str, Any]] = {}
    for source_name, argument in RAW_SOURCE_ARGUMENTS.items():
        report, descriptor = _load_stable_source(
            getattr(args, argument),
            name=f"exposure {source_name.replace('_', ' ')}",
        )
        raw_reports[source_name] = report
        raw_identities[source_name] = descriptor
    if set(raw_reports) != SOURCE_REPORT_NAMES:
        raise ValueError("exposure/capacity decision raw source set differs")
    source_identities["raw_reports"] = raw_identities

    kwargs = {
        "exposure_result": loaded["exposure_result"],
        "exposure_result_identity": source_identities["exposure_result"],
        "validation_receipt": loaded["exposure_result_validation_receipt"],
        "validation_receipt_identity": source_identities[
            "exposure_result_validation_receipt"
        ],
        "preparation": loaded["preparation"],
        "preparation_identity": source_identities["preparation"],
        "quality_bridge_result": loaded["quality_bridge_result"],
        "quality_bridge_result_identity": source_identities[
            "quality_bridge_result"
        ],
        "raw_source_reports": raw_reports,
        "raw_source_identities": raw_identities,
        "decision_git": checkout_identity(args.decision_project_root),
    }
    return kwargs, source_identities


__all__ = [
    "RAW_SOURCE_ARGUMENTS",
    "add_source_arguments",
    "decision_kwargs",
    "parse_build_args",
    "parse_validate_args",
]
