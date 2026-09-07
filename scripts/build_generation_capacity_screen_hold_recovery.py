"""Build the immutable, non-authorizing capacity-screen hold decision."""

from __future__ import annotations

import json

from cofitok.generation.capacity_screen_hold_recovery import (
    build_hold_recovery_decision,
    validate_hold_recovery_decision,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.capacity_screen_hold_recovery_cli import (
        decision_kwargs,
        parse_build_args,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from capacity_screen_hold_recovery_cli import decision_kwargs, parse_build_args


def main() -> None:
    args = parse_build_args()
    output = reject_symlink_chain(
        args.decision, name="capacity screen hold recovery decision"
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite hold recovery decision: {output}")
    kwargs, _ = decision_kwargs(args)
    decision = build_hold_recovery_decision(**kwargs)
    validate_hold_recovery_decision(decision)
    write_json_report(output, decision)
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": decision["scientific_status"],
                "selected_intervention": decision["next_stage"][
                    "selected_intervention"
                ],
                "decision_sha256": file_sha256(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
