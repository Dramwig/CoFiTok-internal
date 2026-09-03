"""Build the source-bound scientific route after the 110K exposure result."""

from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_decision import (
    build_decision,
    validate_decision_contract,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
try:
    from scripts.exposure_capacity_decision_cli import decision_kwargs, parse_build_args
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from exposure_capacity_decision_cli import decision_kwargs, parse_build_args


def main() -> None:
    args = parse_build_args()
    decision_path = reject_symlink_chain(
        args.decision,
        name="exposure/capacity scientific decision",
    ).resolve()
    if decision_path.exists():
        raise FileExistsError(
            f"refusing to overwrite exposure/capacity decision: {decision_path}"
        )
    kwargs, _ = decision_kwargs(args)
    decision = build_decision(**kwargs)
    validate_decision_contract(decision)
    write_json_report(decision_path, decision)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "decision": decision["decision"],
                "scientific_status": decision["scientific_status"],
                "generation_advantage_proven": decision[
                    "generation_advantage_proven"
                ],
                "decision_sha256": file_sha256(decision_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
