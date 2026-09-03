from __future__ import annotations

import json

from cofitok.generation.capacity_screen_execution import (
    validate_capacity_screen_stage_authorization,
)
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.capacity_screen_execution_cli import (
        parse_validate_stage_args,
        stage_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_screen_execution_cli import parse_validate_stage_args, stage_kwargs


def main() -> None:
    args = parse_validate_stage_args()
    path = reject_symlink_chain(
        args.stage_authorization, name="capacity screen stage authorization"
    ).resolve()
    if file_sha256(path) != args.expected_stage_authorization_sha256:
        raise ValueError("capacity screen stage authorization SHA256 differs")
    report = read_object(path, name="capacity screen stage authorization")
    kwargs = stage_kwargs(args)
    validate_capacity_screen_stage_authorization(
        report,
        preparation_identity=kwargs["preparation_identity"],
        execution_checkout=kwargs["execution_checkout"],
        expected_output_root=kwargs["output_root"],
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "stage_authorization_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
