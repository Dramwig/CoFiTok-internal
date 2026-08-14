from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.inference_replay import file_identity
from cofitok.training.authorization import capture_generation_training_authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay a generation training authorization and verify its "
            "immutable file identity."
        )
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-stage", required=True)
    parser.add_argument("--expected-decision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.authorization)
    if identity["sha256"] != args.expected_sha256:
        raise ValueError("generation training authorization SHA256 differs")
    evidence = capture_generation_training_authorization(args.authorization)
    if (
        evidence.get("stage") != args.expected_stage
        or evidence.get("decision") != args.expected_decision
        or evidence.get("gate_sha256") != args.expected_sha256
        or int(evidence.get("gate_bytes", -1)) != identity["bytes"]
    ):
        raise ValueError("generation training authorization identity differs")
    print(
        json.dumps(
            {
                "status": "authorized",
                "authorization": identity,
                "evidence": evidence,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
