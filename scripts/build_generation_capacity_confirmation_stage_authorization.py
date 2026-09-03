from __future__ import annotations

import json

from cofitok.generation.capacity_confirmation_execution import (
    build_capacity_confirmation_stage_authorization,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.capacity_confirmation_execution_cli import (
        parse_build_stage_args,
        stage_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from capacity_confirmation_execution_cli import parse_build_stage_args, stage_kwargs


def main() -> None:
    args = parse_build_stage_args()
    output = reject_symlink_chain(
        args.stage_authorization, name="capacity confirmation stage authorization"
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite stage authorization: {output}")
    kwargs = stage_kwargs(args)
    report = build_capacity_confirmation_stage_authorization(
        **kwargs,
        approved_by=args.approved_by,
        approved_at=args.approved_at,
        source_instruction=args.source_instruction,
    )
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "stage_authorization_sha256": file_sha256(output),
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
