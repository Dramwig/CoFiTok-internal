from __future__ import annotations

import json

from cofitok.generation.terminal_snr_confirmation_arm import (
    build_terminal_snr_confirmation_arm_validation,
    validate_terminal_snr_confirmation_arm_validation,
)
from cofitok.inference_replay import read_json_object, reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_confirmation_arm_cli import arm_kwargs, parse_build_args
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_confirmation_arm_cli import arm_kwargs, parse_build_args


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.output, name="terminal-SNR confirmation arm validation output"
    )
    report = build_terminal_snr_confirmation_arm_validation(**arm_kwargs(args))
    if output.exists():
        if not args.resume:
            raise FileExistsError(
                "terminal-SNR confirmation arm validation exists; pass --resume"
            )
        if read_json_object(output, name="terminal-SNR confirmation arm") != report:
            raise ValueError("existing terminal-SNR confirmation arm differs from replay")
    else:
        write_json_report(output, report)
    written = read_json_object(output, name="terminal-SNR confirmation arm")
    validate_terminal_snr_confirmation_arm_validation(written)
    print(
        json.dumps(
            {
                "status": "pass",
                "arm": written["arm"],
                "report_sha256": file_sha256(output),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
