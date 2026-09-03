from __future__ import annotations

import json

from cofitok.generation.capacity_confirmation_result import (
    build_capacity_confirmation_result,
    validate_capacity_confirmation_result_contract,
)
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.capacity_confirmation_result_cli import parse_build_args, result_kwargs
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_confirmation_result_cli import parse_build_args, result_kwargs


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.output, name="capacity confirmation result output"
    )
    report = build_capacity_confirmation_result(**result_kwargs(args))
    validate_capacity_confirmation_result_contract(report)
    if output.exists():
        if not args.resume:
            raise FileExistsError(
                "capacity confirmation result exists; pass --resume to replay it"
            )
        existing = read_object(output, name="capacity confirmation result")
        if existing != report:
            raise ValueError("existing capacity confirmation result differs from replay")
    else:
        write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": report["scientific_status"],
                "result_sha256": file_sha256(output),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
