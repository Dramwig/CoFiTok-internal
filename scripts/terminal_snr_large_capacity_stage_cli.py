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
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)
    parser.add_argument("--output-root", required=True)


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the non-executing user-goal stage authorization for fresh "
            "terminal-SNR endpoint0975 matched base256/300K work."
        )
    )
    _add_sources(parser)
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--source-instruction", required=True)
    return parser.parse_args(argv)


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically replay a terminal-SNR large-capacity stage authorization."
    )
    _add_sources(parser)
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--expected-stage-authorization-sha256", required=True)
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


def stage_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preparation, preparation_id = stable_load(
        args.preparation,
        args.expected_preparation_sha256,
        "terminal-SNR large-capacity preparation",
    )
    execution = checkout_identity(args.execution_project_root)
    expected = {
        "revision": args.expected_execution_revision,
        "tree": args.expected_execution_tree,
        "branch": args.expected_execution_branch,
        "tracked_dirty": False,
    }
    if execution != expected:
        raise ValueError("large-capacity execution checkout identity differs")
    return {
        "preparation": preparation,
        "preparation_identity": preparation_id,
        "execution_checkout": execution,
        "output_root": args.output_root,
    }


__all__ = ["parse_build_args", "parse_validate_args", "stable_load", "stage_kwargs"]
