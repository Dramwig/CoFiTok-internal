from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_confirmation_execution import (
    build_terminal_snr_confirmation_execution_authorization,
    validate_terminal_snr_confirmation_execution_authorization,
    validate_terminal_snr_confirmation_execution_authorization_physical,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.terminal_snr_confirmation_execution_cli import (
        authorization_kwargs,
        parse_validate_authorization_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_confirmation_execution_cli import (
        authorization_kwargs,
        parse_validate_authorization_args,
    )


def main() -> None:
    args = parse_validate_authorization_args()
    path = reject_symlink_chain(
        args.authorization,
        name="terminal-SNR confirmation execution authorization",
    ).resolve()
    if file_sha256(path) != args.expected_authorization_sha256:
        raise ValueError("terminal-SNR confirmation authorization SHA256 differs")
    actual = read_object(path, name="terminal-SNR confirmation authorization")
    kwargs = authorization_kwargs(args)
    validate_terminal_snr_confirmation_execution_authorization(
        actual,
        preparation=kwargs["preparation"],
        preparation_identity=kwargs["preparation_identity"],
        expected_execution_checkout=kwargs["execution_checkout"],
        expected_output_root=kwargs["output_root"],
    )
    validate_terminal_snr_confirmation_execution_authorization_physical(
        actual,
        preparation=kwargs["preparation"],
        preparation_identity=kwargs["preparation_identity"],
        expected_execution_checkout=kwargs["execution_checkout"],
        expected_output_root=kwargs["output_root"],
    )
    expected = build_terminal_snr_confirmation_execution_authorization(**kwargs)
    if actual != expected:
        raise ValueError("terminal-SNR confirmation authorization differs from replay")
    print(
        json.dumps(
            {
                "status": "pass",
                "authorization_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
