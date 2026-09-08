from __future__ import annotations

import json
from pathlib import Path

from cofitok.generation.terminal_snr_large_capacity import (
    build_terminal_snr_large_capacity_preparation,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_large_capacity_preparation_cli import (
        parse_build_args,
        preparation_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_large_capacity_preparation_cli import (
        parse_build_args,
        preparation_kwargs,
    )


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.preparation, name="terminal-SNR large-capacity preparation"
    ).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite large-capacity preparation: {output}"
        )
    if Path(args.output_root).exists():
        raise ValueError(
            "fresh large-capacity output root must be absent at preparation"
        )
    report = build_terminal_snr_large_capacity_preparation(
        **preparation_kwargs(args)
    )
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "scientific_status": report["scientific_status"],
                "preparation_sha256": file_sha256(output),
                "training_launch_allowed": False,
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
