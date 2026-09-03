"""Rebuild and validate the source-bound exposure/capacity decision."""

from __future__ import annotations

import copy
import json

from cofitok.generation.exposure_capacity_authorization import identity, read_object
from cofitok.generation.exposure_capacity_decision import (
    AUTHORIZATION_BOUNDARY,
    build_decision,
    validate_decision_contract,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
try:
    from scripts.exposure_capacity_decision_cli import (
        decision_kwargs,
        parse_validate_args,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from exposure_capacity_decision_cli import decision_kwargs, parse_validate_args


VALIDATION_SCHEMA = "cofitok_generation_exposure_capacity_decision_validation_v1"
VALIDATION_ROLE = "content_addressed_exposure_capacity_scientific_decision_validation"


def main() -> None:
    args = parse_validate_args()
    decision_path = reject_symlink_chain(
        args.decision,
        name="exposure/capacity scientific decision",
    ).resolve()
    if not decision_path.is_file():
        raise FileNotFoundError(
            f"exposure/capacity decision is missing: {decision_path}"
        )
    if file_sha256(decision_path) != args.expected_decision_sha256:
        raise ValueError("exposure/capacity decision SHA256 differs")
    actual = read_object(decision_path, name="exposure/capacity scientific decision")
    kwargs, verified_sources = decision_kwargs(args)
    expected = build_decision(**kwargs)
    if actual != expected:
        raise ValueError(
            "exposure/capacity decision does not match revalidated source evidence"
        )
    validated = validate_decision_contract(actual)
    report = {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "decision": identity(decision_path),
        "scientific_route": validated["decision"],
        "decision_git": copy.deepcopy(validated["decision_git"]),
        "verified_sources": verified_sources,
        "generation_advantage_proven": False,
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }
    if args.output is not None:
        output_path = reject_symlink_chain(
            args.output,
            name="exposure/capacity decision validation",
        ).resolve()
        if output_path.exists():
            existing = read_object(
                output_path,
                name="exposure/capacity decision validation",
            )
            if existing != report:
                raise ValueError(
                    "existing exposure/capacity decision validation differs"
                )
        else:
            write_json_report(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
