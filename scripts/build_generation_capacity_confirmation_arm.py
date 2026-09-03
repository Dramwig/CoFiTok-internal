from __future__ import annotations

import json

from cofitok.generation.capacity_confirmation_arm import (
    build_capacity_confirmation_arm_validation,
    validate_capacity_confirmation_arm_validation,
)
from cofitok.inference_replay import read_json_object, reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.capacity_confirmation_arm_cli import arm_kwargs, parse_build_args
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_confirmation_arm_cli import arm_kwargs, parse_build_args


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.output, name="capacity confirmation arm validation output"
    )
    report = build_capacity_confirmation_arm_validation(**arm_kwargs(args))
    if output.exists():
        if not args.resume:
            raise FileExistsError(
                "capacity confirmation arm validation exists; pass --resume to replay it"
            )
        existing = read_json_object(
            output, name="capacity confirmation arm validation"
        )
        if existing != report:
            raise ValueError(
                "existing capacity confirmation arm validation differs from replay"
            )
    else:
        write_json_report(output, report)
    written = read_json_object(output, name="capacity confirmation arm validation")
    validate_capacity_confirmation_arm_validation(written)
    print(
        json.dumps(
            {
                "status": "pass",
                "arm": written["arm"],
                "report_sha256": file_sha256(output),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
