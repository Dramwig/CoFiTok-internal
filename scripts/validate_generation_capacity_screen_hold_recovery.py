"""Rebuild and validate the source-bound capacity-screen hold decision."""

from __future__ import annotations

import json

from cofitok.generation.capacity_screen_hold_recovery import (
    build_hold_recovery_decision,
    build_hold_recovery_validation,
    validate_hold_recovery_decision,
    validate_hold_recovery_validation,
)
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.capacity_screen_hold_recovery_cli import (
        decision_kwargs,
        parse_validate_args,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_screen_hold_recovery_cli import decision_kwargs, parse_validate_args


def main() -> None:
    args = parse_validate_args()
    decision_path = reject_symlink_chain(
        args.decision, name="capacity screen hold recovery decision"
    ).resolve()
    if file_sha256(decision_path) != args.expected_decision_sha256:
        raise ValueError("capacity screen hold recovery decision SHA256 differs")
    actual = read_object(decision_path, name="capacity screen hold recovery decision")
    kwargs, verified_sources = decision_kwargs(args)
    expected = build_hold_recovery_decision(**kwargs)
    if actual != expected:
        raise ValueError("capacity screen hold decision differs from physical replay")
    validate_hold_recovery_decision(actual)

    receipt_identity = None
    if (args.validation_receipt is None) is not (
        args.validator_project_root is None
    ):
        raise ValueError(
            "validation receipt and validator project root must be provided together"
        )
    if args.validation_receipt is not None:
        receipt_path = reject_symlink_chain(
            args.validation_receipt,
            name="capacity screen hold recovery validation",
        ).resolve()
        receipt = build_hold_recovery_validation(
            decision=actual,
            decision_identity=identity(decision_path),
            validator_git=checkout_identity(args.validator_project_root),
        )
        if receipt_path.exists():
            existing = read_object(
                receipt_path, name="capacity screen hold recovery validation"
            )
            validate_hold_recovery_validation(
                existing,
                decision=actual,
                decision_identity=identity(decision_path),
            )
            if existing != receipt:
                raise ValueError("existing hold recovery validation differs")
        else:
            write_json_report(receipt_path, receipt)
            validate_hold_recovery_validation(
                read_object(
                    receipt_path,
                    name="capacity screen hold recovery validation",
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
                "selected_intervention": actual["next_stage"][
                    "selected_intervention"
                ],
                "decision_sha256": file_sha256(decision_path),
                "validation_receipt": receipt_identity,
                "verified_source_groups": sorted(verified_sources),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
