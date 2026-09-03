from __future__ import annotations

import json

from cofitok.generation.capacity_screen_result import (
    build_capacity_screen_result,
    validate_capacity_screen_result_contract,
)
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

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
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": actual["scientific_status"],
                "result_sha256": file_sha256(path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
