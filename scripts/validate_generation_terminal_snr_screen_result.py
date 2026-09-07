from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.generation.terminal_snr_screen_result import (
    build_terminal_snr_screen_result,
    build_terminal_snr_screen_validation_receipt,
    validate_terminal_snr_screen_result_contract,
    validate_terminal_snr_screen_validation_receipt,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_screen_result_cli import (
        parse_validate_args,
        result_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_screen_result_cli import (  # type: ignore[no-redef]
        parse_validate_args,
        result_kwargs,
    )


def main() -> None:
    args = parse_validate_args()
    path = reject_symlink_chain(args.result, name="terminal-SNR result").resolve()
    if file_sha256(path) != args.expected_result_sha256:
        raise ValueError("terminal-SNR result SHA256 differs")
    actual = read_object(path, name="terminal-SNR result")
    validate_terminal_snr_screen_result_contract(actual)
    expected = build_terminal_snr_screen_result(**result_kwargs(args))
    if actual != expected:
        raise ValueError("terminal-SNR result differs from physical replay")
    receipt_id = None
    if (args.validation_receipt is None) is not (
        args.validator_project_root is None
    ):
        raise ValueError("validation receipt and validator root must be paired")
    if args.validation_receipt is not None:
        receipt_path = reject_symlink_chain(
            args.validation_receipt, name="terminal-SNR result validation"
        ).resolve()
        validator_git = checkout_identity(args.validator_project_root)
        receipt = build_terminal_snr_screen_validation_receipt(
            result=actual,
            result_identity=identity(path),
            validator_git=validator_git,
        )
        if receipt_path.exists():
            existing = read_object(receipt_path, name="terminal-SNR result validation")
            validate_terminal_snr_screen_validation_receipt(
                existing,
                result=actual,
                result_identity=identity(path),
                validator_git=validator_git,
            )
            if existing != receipt:
                raise ValueError("existing terminal-SNR result validation differs")
        else:
            write_json_report(receipt_path, receipt)
            validate_terminal_snr_screen_validation_receipt(
                read_object(receipt_path, name="terminal-SNR result validation"),
                result=actual,
                result_identity=identity(path),
                validator_git=validator_git,
            )
        receipt_id = identity(receipt_path)
    print(json.dumps({
        "status": "pass",
        "scientific_status": actual["scientific_status"],
        "screen_pass": actual["screen_pass"],
        "result_sha256": file_sha256(path),
        "validation_receipt": receipt_id,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
