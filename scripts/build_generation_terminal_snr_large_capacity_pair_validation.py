from __future__ import annotations

import json

from cofitok.generation.terminal_snr_large_capacity_execution import (
    build_terminal_snr_large_capacity_pair_validation,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_large_capacity_pair_cli import (
        pair_kwargs,
        parse_build_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_large_capacity_pair_cli import pair_kwargs, parse_build_args


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.pair_validation,
        name="terminal-SNR large-capacity pair validation",
    ).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite large-capacity pair validation: {output}"
        )
    report = build_terminal_snr_large_capacity_pair_validation(
        **pair_kwargs(args)
    )
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pair_validation_sha256": file_sha256(output),
                "execution_ready": False,
                "training_launch_allowed": False,
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
