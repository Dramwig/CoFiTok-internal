from __future__ import annotations

from cofitok.generation.capacity_screen_arm import (
    build_capacity_screen_arm_validation,
    validate_capacity_screen_arm_validation,
)
from cofitok.inference_replay import read_json_object, reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.capacity_screen_arm_cli import arm_kwargs, parse_build_args
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_screen_arm_cli import arm_kwargs, parse_build_args


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(args.output, name="capacity arm validation output")
    report = build_capacity_screen_arm_validation(**arm_kwargs(args))
    if output.exists():
        if not args.resume:
            raise FileExistsError(
                "capacity arm validation exists; pass --resume to replay it"
            )
        existing = read_json_object(output, name="capacity arm validation")
        if existing != report:
            raise ValueError("existing capacity arm validation differs from replay")
    else:
        write_json_report(output, report)
    written = read_json_object(output, name="capacity arm validation")
    validate_capacity_screen_arm_validation(written)
    print(
        {
            "status": "pass",
            "arm": written["arm"],
            "report_sha256": file_sha256(output),
        }
    )


if __name__ == "__main__":
    main()
