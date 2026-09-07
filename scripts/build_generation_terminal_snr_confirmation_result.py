from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_confirmation_result import (
    build_terminal_snr_confirmation_result,
    validate_terminal_snr_confirmation_result_contract,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_confirmation_result_cli import (
        parse_build_args,
        result_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_confirmation_result_cli import parse_build_args, result_kwargs


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.output, name="terminal-SNR confirmation result output"
    )
    report = build_terminal_snr_confirmation_result(**result_kwargs(args))
    validate_terminal_snr_confirmation_result_contract(report)
    if output.exists():
        if not args.resume:
            raise FileExistsError(
                "terminal-SNR confirmation result exists; pass --resume"
            )
        if read_object(output, name="terminal-SNR confirmation result") != report:
            raise ValueError("existing terminal-SNR confirmation result differs")
    else:
        write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": report["scientific_status"],
                "support_collapse_resolved": report["support_collapse_resolved"],
                "result_sha256": file_sha256(output),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
