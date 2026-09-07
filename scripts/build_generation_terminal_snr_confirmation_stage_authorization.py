from __future__ import annotations

import json

from cofitok.generation.terminal_snr_confirmation_execution import (
    build_terminal_snr_confirmation_stage_authorization,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_confirmation_execution_cli import (
        parse_build_stage_args,
        stage_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_confirmation_execution_cli import (
        parse_build_stage_args,
        stage_kwargs,
    )


def main() -> None:
    args = parse_build_stage_args()
    output = reject_symlink_chain(
        args.stage_authorization,
        name="terminal-SNR confirmation stage authorization",
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite stage authorization: {output}")
    report = build_terminal_snr_confirmation_stage_authorization(
        **stage_kwargs(args),
        approved_by=args.approved_by,
        approved_at=args.approved_at,
        source_instruction=args.source_instruction,
    )
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "stage_authorization_sha256": file_sha256(output),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
