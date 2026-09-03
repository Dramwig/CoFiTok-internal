from __future__ import annotations

import json

from cofitok.generation.capacity_screen_arm import (
    build_capacity_screen_arm_validation,
    validate_capacity_screen_arm_validation,
)
from cofitok.inference_replay import read_json_object, reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.capacity_screen_arm_cli import arm_kwargs, parse_validate_args
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_screen_arm_cli import arm_kwargs, parse_validate_args


def main() -> None:
    args = parse_validate_args()
    path = reject_symlink_chain(
        args.arm_validation, name="capacity arm validation"
    ).resolve()
    if file_sha256(path) != args.expected_arm_validation_sha256:
        raise ValueError("capacity arm validation SHA256 differs")
    actual = read_json_object(path, name="capacity arm validation")
    validate_capacity_screen_arm_validation(actual)
    expected = build_capacity_screen_arm_validation(**arm_kwargs(args))
    if actual != expected:
        raise ValueError("capacity arm validation differs from direct physical replay")
    print(
        json.dumps(
            {
                "status": "pass",
                "arm": actual["arm"],
                "report_sha256": file_sha256(path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
