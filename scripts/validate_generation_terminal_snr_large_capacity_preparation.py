from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_large_capacity import (
    build_terminal_snr_large_capacity_preparation,
    validate_terminal_snr_large_capacity_preparation_contract,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.terminal_snr_large_capacity_preparation_cli import (
        parse_validate_args,
        preparation_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_large_capacity_preparation_cli import (
        parse_validate_args,
        preparation_kwargs,
    )


def main() -> None:
    args = parse_validate_args()
    path = reject_symlink_chain(
        args.preparation, name="terminal-SNR large-capacity preparation"
    ).resolve()
    if file_sha256(path) != args.expected_preparation_sha256:
        raise ValueError("terminal-SNR large-capacity preparation SHA256 differs")
    actual = read_object(path, name="terminal-SNR large-capacity preparation")
    validate_terminal_snr_large_capacity_preparation_contract(
        actual, expected_output_root=args.output_root
    )
    expected = build_terminal_snr_large_capacity_preparation(
        **preparation_kwargs(args)
    )
    if actual != expected:
        raise ValueError(
            "terminal-SNR large-capacity preparation differs from physical replay"
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "preparation_sha256": file_sha256(path),
                "training_launch_allowed": False,
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
