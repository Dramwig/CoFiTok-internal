"""Build the immutable source-bound four-arm capacity-screen preparation."""

from __future__ import annotations

import json

from cofitok.generation.capacity_screen import (
    build_capacity_screen_preparation,
    validate_capacity_screen_preparation_contract,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
try:
    from scripts.capacity_screen_preparation_cli import (
        parse_build_args,
        preparation_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from capacity_screen_preparation_cli import parse_build_args, preparation_kwargs


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.preparation,
        name="capacity screen preparation",
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite capacity preparation: {output}")
    kwargs, _ = preparation_kwargs(args)
    report = build_capacity_screen_preparation(**kwargs)
    validate_capacity_screen_preparation_contract(
        report,
        expected_output_root=args.output_root,
    )
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "scientific_status": report["scientific_status"],
                "preparation_sha256": file_sha256(output),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
