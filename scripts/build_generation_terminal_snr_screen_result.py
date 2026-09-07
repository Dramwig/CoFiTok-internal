from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_screen_result import (
    build_terminal_snr_screen_result,
    validate_terminal_snr_screen_result_contract,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_screen_result_cli import parse_build_args, result_kwargs
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_screen_result_cli import (  # type: ignore[no-redef]
        parse_build_args,
        result_kwargs,
    )


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(args.output, name="terminal-SNR result output")
    report = build_terminal_snr_screen_result(**result_kwargs(args))
    validate_terminal_snr_screen_result_contract(report)
    if output.exists():
        if not args.resume:
            raise FileExistsError("terminal-SNR result exists; pass --resume to replay")
        if read_object(output, name="terminal-SNR result") != report:
            raise ValueError("existing terminal-SNR result differs from replay")
    else:
        write_json_report(output, report)
    print(json.dumps({
        "status": "pass",
        "scientific_status": report["scientific_status"],
        "screen_pass": report["screen_pass"],
        "result_sha256": file_sha256(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
