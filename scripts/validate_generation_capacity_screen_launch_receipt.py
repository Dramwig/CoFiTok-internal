from __future__ import annotations

import json

from cofitok.generation.capacity_screen_execution import (
    build_capacity_screen_launch_receipt,
)
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256
try:
    from scripts.capacity_screen_execution_cli import (
        launch_kwargs,
        parse_validate_launch_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from capacity_screen_execution_cli import launch_kwargs, parse_validate_launch_args


def main() -> None:
    args = parse_validate_launch_args()
    path = reject_symlink_chain(
        args.launch_receipt,
        name="capacity screen launch receipt",
    ).resolve()
    if file_sha256(path) != args.expected_launch_receipt_sha256:
        raise ValueError("capacity screen launch receipt SHA256 differs")
    actual = read_object(path, name="capacity screen launch receipt")
    expected = build_capacity_screen_launch_receipt(**launch_kwargs(args))
    if actual != expected:
        raise ValueError("capacity screen launch receipt replay differs")
    print(
        json.dumps(
            {
                "status": "pass",
                "launch_receipt_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
