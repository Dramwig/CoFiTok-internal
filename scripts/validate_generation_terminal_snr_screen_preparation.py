"""Physically rebuild a terminal-SNR screen preparation."""

from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_screen import (
    build_terminal_snr_screen_preparation,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.terminal_snr_screen_preparation_cli import (
        parse_validate_args,
        preparation_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_screen_preparation_cli import (  # type: ignore[no-redef]
        parse_validate_args,
        preparation_kwargs,
    )


def main() -> None:
    args = parse_validate_args()
    path = reject_symlink_chain(
        args.preparation, name="terminal-SNR screen preparation"
    ).resolve()
    if file_sha256(path) != args.expected_preparation_sha256:
        raise ValueError("terminal-SNR screen preparation SHA256 differs")
    actual = read_object(path, name="terminal-SNR screen preparation")
    kwargs, sources = preparation_kwargs(args)
    expected = build_terminal_snr_screen_preparation(**kwargs)
    if actual != expected:
        raise ValueError("terminal-SNR preparation differs from physical replay")
    print(
        json.dumps(
            {
                "status": "pass",
                "preparation_sha256": file_sha256(path),
                "verified_source_groups": sorted(sources),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
