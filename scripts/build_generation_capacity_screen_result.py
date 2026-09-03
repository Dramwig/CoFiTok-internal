from __future__ import annotations

import json

from cofitok.generation.capacity_screen_result import (
    build_capacity_screen_result,
    validate_capacity_screen_result_contract,
)
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.capacity_screen_result_cli import parse_build_args, result_kwargs
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_screen_result_cli import parse_build_args, result_kwargs


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(args.output, name="capacity screen result output")
    report = build_capacity_screen_result(**result_kwargs(args))
    validate_capacity_screen_result_contract(report)
    if output.exists():
        if not args.resume:
            raise FileExistsError("capacity screen result exists; pass --resume to replay it")
        existing = read_object(output, name="capacity screen result")
        if existing != report:
            raise ValueError("existing capacity screen result differs from replay")
    else:
        write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": report["scientific_status"],
                "result_sha256": file_sha256(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
