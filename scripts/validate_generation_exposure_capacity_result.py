"""Rebuild and compare a bounded exposure continuation result."""

from __future__ import annotations

import json
from pathlib import Path

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.generation.exposure_capacity_result import (
    build_result,
    build_validation_receipt,
    validate_validation_receipt,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
from exposure_capacity_result_cli import parse_args, result_kwargs


def main() -> None:
    args = parse_args()
    result_path = reject_symlink_chain(args.result, name="continuation result").resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"continuation result is missing: {result_path}")
    for path, expected, name in (
        (args.authorization, args.expected_authorization_sha256, "authorization"),
        (args.gate, args.expected_gate_sha256, "execution gate"),
        (args.preparation, args.expected_preparation_sha256, "preparation"),
        (
            args.standing_authorization,
            args.expected_standing_authorization_sha256,
            "standing authorization",
        ),
        (
            args.stage_authorization,
            args.expected_stage_authorization_sha256,
            "stage authorization",
        ),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{name} SHA256 differs")
    actual = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict):
        raise ValueError("continuation result must be a JSON object")
    expected = build_result(**result_kwargs(args))
    if actual != expected:
        raise ValueError("continuation result does not match revalidated source evidence")
    receipt_identity = None
    if args.validation_receipt is not None:
        receipt_path = reject_symlink_chain(
            args.validation_receipt,
            name="continuation result validation receipt",
        ).resolve()
        receipt = build_validation_receipt(
            result=actual,
            result_identity=identity(result_path),
            validator_git=checkout_identity(
                args.validator_project_root
                if args.validator_project_root is not None
                else Path(__file__).resolve().parents[1]
            ),
        )
        if receipt_path.exists():
            existing = read_object(
                receipt_path,
                name="continuation result validation receipt",
            )
            validate_validation_receipt(
                existing,
                result=actual,
                result_identity=identity(result_path),
            )
            if existing != receipt:
                raise ValueError(
                    "existing continuation result validation receipt differs"
                )
        else:
            write_json_report(receipt_path, receipt)
            written = read_object(
                receipt_path,
                name="continuation result validation receipt",
            )
            validate_validation_receipt(
                written,
                result=actual,
                result_identity=identity(result_path),
            )
        receipt_identity = identity(receipt_path)
    print(
        json.dumps(
            {
                "status": "pass",
                "result_sha256": file_sha256(result_path),
                "scientific_status": actual["scientific_status"],
                "generation_advantage_proven": actual["generation_advantage_proven"],
                "validation_receipt": receipt_identity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
