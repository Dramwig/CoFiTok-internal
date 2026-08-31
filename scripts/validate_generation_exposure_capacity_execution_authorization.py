"""Validate an exposure continuation authorization and all static bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    validate_authorization_contract,
)
from cofitok.reporting import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exposure continuation authorization.")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--expected-stage-authorization-sha256", required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    args = parser.parse_args()
    for path, expected, label in (
        (args.authorization, args.expected_authorization_sha256, "authorization"),
        (args.gate, args.expected_gate_sha256, "candidate gate"),
        (args.preparation, args.expected_preparation_sha256, "preparation"),
        (
            args.standing_authorization,
            args.expected_standing_authorization_sha256,
            "standing authorization",
        ),
        (
            args.stage_authorization,
            args.expected_stage_authorization_sha256,
            "stage authorization",
        ),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} SHA256 differs")
    preparation = json.loads(args.preparation.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (preparation, gate, authorization)):
        raise ValueError("authorization sources must be JSON objects")
    stage_identity = identity(args.stage_authorization)
    if authorization.get("stage_authorization_identity") != stage_identity:
        raise ValueError("authorization binds another stage authorization")
    validated = validate_authorization_contract(
        authorization,
        gate=gate,
        preparation=preparation,
        preparation_identity=identity(args.preparation),
        gate_identity=identity(args.gate),
        standing_identity=identity(args.standing_authorization),
        source_checkout=checkout_identity(args.source_project_root),
        execution_checkout=checkout_identity(args.execution_project_root),
        config_identities={
            "cofitok": identity(args.cofitok_config),
            "dense_identity": identity(args.dense_config),
        },
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "schema_version": validated["schema_version"],
                "selected_arm": validated["selected_arm"],
                "decision": validated["decision"],
                "authorization_boundary": validated["authorization_boundary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
