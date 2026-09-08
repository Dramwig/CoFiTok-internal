from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--confirmation-result", type=Path, required=True)
    parser.add_argument("--expected-confirmation-result-sha256", required=True)
    parser.add_argument("--confirmation-validation", type=Path, required=True)
    parser.add_argument(
        "--expected-confirmation-validation-sha256", required=True
    )
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--expected-cofitok-config-sha256", required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--expected-dense-config-sha256", required=True)
    parser.add_argument("--expected-preparation-revision", required=True)
    parser.add_argument("--expected-preparation-tree", required=True)
    parser.add_argument("--expected-preparation-branch", required=True)
    parser.add_argument("--output-root", required=True)


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-authorizing fresh endpoint0975 base256/300K "
            "large-capacity preparation from the terminal-SNR confirmation."
        )
    )
    _add_sources(parser)
    parser.add_argument("--preparation", type=Path, required=True)
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay a terminal-SNR large-capacity preparation."
        )
    )
    _add_sources(parser)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
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


def _expected_git(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "revision": args.expected_preparation_revision,
        "tree": args.expected_preparation_tree,
        "branch": args.expected_preparation_branch,
        "tracked_dirty": False,
    }


def preparation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preparation_git = checkout_identity(args.project_root)
    if preparation_git != _expected_git(args):
        raise ValueError("large-capacity preparation checkout identity differs")
    result, result_id = _stable_load(
        args.confirmation_result,
        args.expected_confirmation_result_sha256,
        "terminal-SNR confirmation result",
    )
    validation, validation_id = _stable_load(
        args.confirmation_validation,
        args.expected_confirmation_validation_sha256,
        "terminal-SNR confirmation validation",
    )
    cofitok_config, cofitok_config_id = _stable_load(
        args.cofitok_config,
        args.expected_cofitok_config_sha256,
        "large-capacity CoFiTok config",
    )
    dense_config, dense_config_id = _stable_load(
        args.dense_config,
        args.expected_dense_config_sha256,
        "large-capacity dense config",
    )
    return {
        "confirmation_result": result,
        "confirmation_result_identity": result_id,
        "confirmation_validation": validation,
        "confirmation_validation_identity": validation_id,
        "cofitok_config": cofitok_config,
        "cofitok_config_identity": cofitok_config_id,
        "dense_config": dense_config,
        "dense_config_identity": dense_config_id,
        "preparation_git": preparation_git,
        "output_root": args.output_root,
    }


__all__ = [
    "parse_build_args",
    "parse_validate_args",
    "preparation_kwargs",
]
