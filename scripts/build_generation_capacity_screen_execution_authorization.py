from __future__ import annotations

import json

from cofitok.generation.capacity_screen_execution import (
    build_capacity_screen_execution_authorization,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
try:
    from scripts.capacity_screen_execution_cli import (
        authorization_kwargs,
        parse_build_authorization_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from capacity_screen_execution_cli import (
        authorization_kwargs,
        parse_build_authorization_args,
    )


def main() -> None:
    args = parse_build_authorization_args()
    path = reject_symlink_chain(
        args.authorization,
        name="capacity screen execution authorization",
    ).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite authorization: {path}")
    report = build_capacity_screen_execution_authorization(
        **authorization_kwargs(args)
    )
    write_json_report(path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "authorization_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
