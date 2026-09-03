"""Rehash, rebuild, and validate a capacity-screen preparation."""

from __future__ import annotations

import json

from cofitok.generation.capacity_screen import build_capacity_screen_preparation
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256
try:
    from scripts.capacity_screen_preparation_cli import (
        parse_validate_args,
        preparation_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from capacity_screen_preparation_cli import parse_validate_args, preparation_kwargs


def main() -> None:
    args = parse_validate_args()
    path = reject_symlink_chain(
        args.preparation,
        name="capacity screen preparation",
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"capacity screen preparation is missing: {path}")
    if file_sha256(path) != args.expected_preparation_sha256:
        raise ValueError("capacity screen preparation SHA256 differs")
    actual = read_object(path, name="capacity screen preparation")
    kwargs, sources = preparation_kwargs(args)
    expected = build_capacity_screen_preparation(**kwargs)
    if actual != expected:
        raise ValueError("capacity screen preparation does not match source replay")
    print(
        json.dumps(
            {
                "status": "pass",
                "preparation_sha256": file_sha256(path),
                "source_evidence": sources,
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
