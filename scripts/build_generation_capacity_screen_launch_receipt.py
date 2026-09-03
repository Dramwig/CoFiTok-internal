from __future__ import annotations

import json

from cofitok.generation.capacity_screen_execution import (
    build_capacity_screen_launch_receipt,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
try:
    from scripts.capacity_screen_execution_cli import (
        launch_kwargs,
        parse_build_launch_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from capacity_screen_execution_cli import launch_kwargs, parse_build_launch_args


def main() -> None:
    args = parse_build_launch_args()
    path = reject_symlink_chain(
        args.launch_receipt,
        name="capacity screen launch receipt",
    ).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite launch receipt: {path}")
    report = build_capacity_screen_launch_receipt(**launch_kwargs(args))
    write_json_report(path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "launch_receipt_sha256": file_sha256(path),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
