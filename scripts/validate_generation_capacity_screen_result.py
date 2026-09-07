from __future__ import annotations

import json

from cofitok.generation.capacity_screen_result import (
    build_capacity_screen_validation_receipt,
    build_capacity_screen_result,
    validate_capacity_screen_result_contract,
    validate_capacity_screen_validation_receipt,
)
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.capacity_screen_result_cli import parse_validate_args, result_kwargs
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_screen_result_cli import parse_validate_args, result_kwargs


def main() -> None:
    args = parse_validate_args()
    path = reject_symlink_chain(args.result, name="capacity screen result").resolve()
    if file_sha256(path) != args.expected_result_sha256:
        raise ValueError("capacity screen result SHA256 differs")
    actual = read_object(path, name="capacity screen result")
    validate_capacity_screen_result_contract(actual)
    expected = build_capacity_screen_result(**result_kwargs(args))
    if actual != expected:
        raise ValueError("capacity screen result differs from physical replay")
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
            name="capacity screen result validation receipt",
        ).resolve()
        receipt = build_capacity_screen_validation_receipt(
            result=actual,
            result_identity=identity(path),
            validator_git=checkout_identity(args.validator_project_root),
        )
        if receipt_path.exists():
            existing = read_object(
                receipt_path,
                name="capacity screen result validation receipt",
            )
            validate_capacity_screen_validation_receipt(
                existing,
                result=actual,
                result_identity=identity(path),
            )
            if existing != receipt:
                raise ValueError(
                    "existing capacity screen validation receipt differs"
                )
        else:
            write_json_report(receipt_path, receipt)
            validate_capacity_screen_validation_receipt(
                read_object(
                    receipt_path,
                    name="capacity screen result validation receipt",
                ),
                result=actual,
                result_identity=identity(path),
            )
        receipt_identity = identity(receipt_path)
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": actual["scientific_status"],
                "result_sha256": file_sha256(path),
                "validation_receipt": receipt_identity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
