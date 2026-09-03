from __future__ import annotations

import json

from cofitok.generation.capacity_confirmation_execution import (
    build_capacity_confirmation_execution_authorization,
    validate_capacity_confirmation_execution_authorization,
)
from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.capacity_confirmation_execution_cli import (
        authorization_kwargs,
        parse_validate_authorization_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from capacity_confirmation_execution_cli import (
        authorization_kwargs,
        parse_validate_authorization_args,
    )


def main() -> None:
    args = parse_validate_authorization_args()
    path = reject_symlink_chain(
        args.authorization, name="capacity confirmation execution authorization"
    ).resolve()
    if file_sha256(path) != args.expected_authorization_sha256:
        raise ValueError("capacity confirmation execution authorization SHA256 differs")
    actual = read_object(path, name="capacity confirmation execution authorization")
    kwargs = authorization_kwargs(args)
    validate_capacity_confirmation_execution_authorization(
        actual,
        preparation_identity=kwargs["preparation_identity"],
        expected_execution_checkout=kwargs["execution_checkout"],
        expected_output_root=kwargs["output_root"],
    )
    expected = build_capacity_confirmation_execution_authorization(**kwargs)
    if actual != expected:
        raise ValueError("capacity confirmation authorization differs from replay")
    print(
        json.dumps(
            {
                "status": "pass",
                "authorization_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
