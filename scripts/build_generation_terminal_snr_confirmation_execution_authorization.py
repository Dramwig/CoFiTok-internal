from __future__ import annotations

import json

from cofitok.generation.terminal_snr_confirmation_execution import (
    build_terminal_snr_confirmation_execution_authorization,
    validate_terminal_snr_confirmation_execution_authorization_physical,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_confirmation_execution_cli import (
        authorization_kwargs,
        parse_build_authorization_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_confirmation_execution_cli import (
        authorization_kwargs,
        parse_build_authorization_args,
    )


def main() -> None:
    args = parse_build_authorization_args()
    output = reject_symlink_chain(
        args.authorization,
        name="terminal-SNR confirmation execution authorization",
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite authorization: {output}")
    kwargs = authorization_kwargs(args)
    report = build_terminal_snr_confirmation_execution_authorization(**kwargs)
    validate_terminal_snr_confirmation_execution_authorization_physical(
        report,
        preparation=kwargs["preparation"],
        preparation_identity=kwargs["preparation_identity"],
        expected_execution_checkout=kwargs["execution_checkout"],
        expected_output_root=kwargs["output_root"],
    )
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "authorization_sha256": file_sha256(output),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
