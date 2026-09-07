"""Build an immutable, non-authorizing terminal-SNR screen preparation."""

from __future__ import annotations

import json

from cofitok.generation.terminal_snr_screen import (
    build_terminal_snr_screen_preparation,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_screen_preparation_cli import (
        parse_build_args,
        preparation_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_screen_preparation_cli import (  # type: ignore[no-redef]
        parse_build_args,
        preparation_kwargs,
    )


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.preparation, name="terminal-SNR screen preparation"
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preparation: {output}")
    kwargs, _ = preparation_kwargs(args)
    report = build_terminal_snr_screen_preparation(**kwargs)
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": "pass",
                "preparation_sha256": file_sha256(output),
                "scientific_status": report["scientific_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
