from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.generation.terminal_snr_large_capacity import CONFIG_FILENAMES


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument(
        "--expected-stage-authorization-sha256", required=True
    )
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--expected-cofitok-config-sha256", required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--expected-dense-config-sha256", required=True)
    parser.add_argument("--output-root", required=True)


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the immutable matched-pair validation receipt for fresh "
            "terminal-SNR endpoint0975 base256/300K work."
        )
    )
    _add_sources(parser)
    parser.add_argument("--pair-validation", type=Path, required=True)
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay a terminal-SNR large-capacity matched-pair "
            "validation receipt."
        )
    )
    _add_sources(parser)
    parser.add_argument("--pair-validation", type=Path, required=True)
    parser.add_argument("--expected-pair-validation-sha256", required=True)
    return parser.parse_args(argv)


def stable_load(
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


def _expected_checkout(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "revision": args.expected_execution_revision,
        "tree": args.expected_execution_tree,
        "branch": args.expected_execution_branch,
        "tracked_dirty": False,
    }


def _require_canonical_config_paths(args: argparse.Namespace) -> None:
    project_root = args.execution_project_root.resolve()
    expected = {
        "cofitok": (
            project_root / "configs" / "generation" / CONFIG_FILENAMES["cofitok"]
        ).resolve(),
        "dense_identity": (
            project_root
            / "configs"
            / "generation"
            / CONFIG_FILENAMES["dense_identity"]
        ).resolve(),
    }
    actual = {
        "cofitok": args.cofitok_config.resolve(),
        "dense_identity": args.dense_config.resolve(),
    }
    if actual != expected:
        raise ValueError(
            "large-capacity pair validation must use the canonical execution "
            "checkout configs"
        )


def pair_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    execution = checkout_identity(args.execution_project_root)
    if execution != _expected_checkout(args):
        raise ValueError("large-capacity execution checkout identity differs")
    _require_canonical_config_paths(args)
    preparation, preparation_id = stable_load(
        args.preparation,
        args.expected_preparation_sha256,
        "terminal-SNR large-capacity preparation",
    )
    stage, stage_id = stable_load(
        args.stage_authorization,
        args.expected_stage_authorization_sha256,
        "terminal-SNR large-capacity stage authorization",
    )
    cofitok_config, cofitok_config_id = stable_load(
        args.cofitok_config,
        args.expected_cofitok_config_sha256,
        "terminal-SNR large-capacity CoFiTok config",
    )
    dense_config, dense_config_id = stable_load(
        args.dense_config,
        args.expected_dense_config_sha256,
        "terminal-SNR large-capacity dense config",
    )
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "stage_authorization": stage,
        "stage_authorization_identity": stage_id,
        "cofitok_config": cofitok_config,
        "cofitok_config_identity": cofitok_config_id,
        "dense_config": dense_config,
        "dense_config_identity": dense_config_id,
        "execution_checkout": execution,
        "output_root": args.output_root,
    }


__all__ = [
    "pair_kwargs",
    "parse_build_args",
    "parse_validate_args",
    "stable_load",
]
