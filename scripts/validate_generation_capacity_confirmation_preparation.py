from __future__ import annotations

import json

from cofitok.generation.capacity_confirmation import (
    build_capacity_confirmation_preparation,
    validate_capacity_confirmation_preparation_contract,
)
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.capacity_confirmation_preparation_cli import (
        parse_validate_args,
        preparation_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from capacity_confirmation_preparation_cli import parse_validate_args, preparation_kwargs


def main() -> None:
    args = parse_validate_args()
    path = reject_symlink_chain(
        args.preparation, name="capacity confirmation preparation"
    ).resolve()
    if file_sha256(path) != args.expected_preparation_sha256:
        raise ValueError("capacity confirmation preparation SHA256 differs")
    actual = read_object(path, name="capacity confirmation preparation")
    validate_capacity_confirmation_preparation_contract(
        actual, expected_output_root=args.output_root
    )
    expected = build_capacity_confirmation_preparation(**preparation_kwargs(args))
    if actual != expected:
        raise ValueError("capacity confirmation preparation differs from replay")
    print(
        json.dumps(
            {
                "status": "pass",
                "preparation_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
