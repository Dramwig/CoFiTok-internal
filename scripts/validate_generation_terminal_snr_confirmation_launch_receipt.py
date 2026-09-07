from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_confirmation_execution import (
    build_terminal_snr_confirmation_launch_receipt,
    validate_terminal_snr_confirmation_launch_receipt_contract,
    validate_terminal_snr_confirmation_launch_receipt_physical,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.terminal_snr_confirmation_execution_cli import (
        launch_kwargs,
        parse_validate_launch_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_confirmation_execution_cli import (
        launch_kwargs,
        parse_validate_launch_args,
    )


def main() -> None:
    args = parse_validate_launch_args()
    path = reject_symlink_chain(
        args.launch_receipt, name="terminal-SNR confirmation launch receipt"
    ).resolve()
    if file_sha256(path) != args.expected_launch_receipt_sha256:
        raise ValueError("terminal-SNR confirmation launch receipt SHA256 differs")
    actual = read_object(path, name="terminal-SNR confirmation launch receipt")
    kwargs = launch_kwargs(args)
    validate_terminal_snr_confirmation_launch_receipt_contract(
        actual, expected_execution_checkout=kwargs["execution_checkout"]
    )
    validate_terminal_snr_confirmation_launch_receipt_physical(
        actual, expected_execution_checkout=kwargs["execution_checkout"]
    )
    expected = build_terminal_snr_confirmation_launch_receipt(**kwargs)
    if actual != expected:
        raise ValueError("terminal-SNR confirmation launch receipt differs from replay")
    print(
        json.dumps(
            {
                "status": "pass",
                "launch_receipt_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
