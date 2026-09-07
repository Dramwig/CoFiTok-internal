"""Build an immutable, non-authorizing terminal-SNR reassessment."""

from __future__ import annotations

import json

from cofitok.generation.terminal_snr_reassessment import (
    build_terminal_snr_reassessment,
    validate_terminal_snr_reassessment,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_reassessment_cli import (
        parse_build_args,
        reassessment_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from terminal_snr_reassessment_cli import parse_build_args, reassessment_kwargs


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.decision, name="terminal-SNR reassessment"
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reassessment: {output}")
    kwargs, _ = reassessment_kwargs(args)
    report = build_terminal_snr_reassessment(**kwargs)
    validate_terminal_snr_reassessment(report)
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": report["scientific_status"],
                "selected_intervention": report["selected_intervention"]["id"],
                "decision_sha256": file_sha256(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
