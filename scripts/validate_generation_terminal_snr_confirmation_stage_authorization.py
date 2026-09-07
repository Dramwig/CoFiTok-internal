from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_confirmation_execution import (
    validate_terminal_snr_confirmation_stage_authorization,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.terminal_snr_confirmation_execution_cli import (
        parse_validate_stage_args,
        stage_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_confirmation_execution_cli import (
        parse_validate_stage_args,
        stage_kwargs,
    )


def main() -> None:
    args = parse_validate_stage_args()
    path = reject_symlink_chain(
        args.stage_authorization,
        name="terminal-SNR confirmation stage authorization",
    ).resolve()
    if file_sha256(path) != args.expected_stage_authorization_sha256:
        raise ValueError("terminal-SNR confirmation stage authorization SHA256 differs")
    report = read_object(path, name="terminal-SNR confirmation stage authorization")
    kwargs = stage_kwargs(args)
    validate_terminal_snr_confirmation_stage_authorization(
        report,
        preparation_identity=kwargs["preparation_identity"],
        execution_checkout=kwargs["execution_checkout"],
        expected_output_root=kwargs["output_root"],
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "stage_authorization_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
