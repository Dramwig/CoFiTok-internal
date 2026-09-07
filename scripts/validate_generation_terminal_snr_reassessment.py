"""Rebuild and validate a source-bound terminal-SNR reassessment."""

from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.generation.terminal_snr_reassessment import (
    build_terminal_snr_reassessment,
    build_terminal_snr_reassessment_validation,
    validate_terminal_snr_reassessment,
    validate_terminal_snr_reassessment_validation,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.terminal_snr_reassessment_cli import (
        parse_validate_args,
        reassessment_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from terminal_snr_reassessment_cli import parse_validate_args, reassessment_kwargs


def main() -> None:
    args = parse_validate_args()
    decision_path = reject_symlink_chain(
        args.decision, name="terminal-SNR reassessment"
    ).resolve()
    if file_sha256(decision_path) != args.expected_decision_sha256:
        raise ValueError("terminal-SNR reassessment SHA256 differs")
    actual = read_object(decision_path, name="terminal-SNR reassessment")
    kwargs, verified_sources = reassessment_kwargs(args)
    expected = build_terminal_snr_reassessment(**kwargs)
    if actual != expected:
        raise ValueError("terminal-SNR reassessment differs from physical replay")
    validate_terminal_snr_reassessment(actual)

    receipt_identity = None
    if (args.validation_receipt is None) is not (
        args.validator_project_root is None
    ):
        raise ValueError(
            "validation receipt and validator project root must be provided together"
        )
    if args.validation_receipt is not None:
        receipt_path = reject_symlink_chain(
            args.validation_receipt, name="terminal-SNR reassessment validation"
        ).resolve()
        receipt = build_terminal_snr_reassessment_validation(
            decision=actual,
            decision_identity=identity(decision_path),
            validator_git=checkout_identity(args.validator_project_root),
        )
        if receipt_path.exists():
            existing = read_object(
                receipt_path, name="terminal-SNR reassessment validation"
            )
            validate_terminal_snr_reassessment_validation(
                existing,
                decision=actual,
                decision_identity=identity(decision_path),
            )
            if existing != receipt:
                raise ValueError("existing reassessment validation differs")
        else:
            write_json_report(receipt_path, receipt)
            validate_terminal_snr_reassessment_validation(
                read_object(
                    receipt_path, name="terminal-SNR reassessment validation"
                ),
                decision=actual,
                decision_identity=identity(decision_path),
            )
        receipt_identity = identity(receipt_path)

    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": actual["scientific_status"],
                "selected_intervention": actual["selected_intervention"]["id"],
                "decision_sha256": file_sha256(decision_path),
                "validation_receipt": receipt_identity,
                "verified_source_groups": sorted(verified_sources),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
