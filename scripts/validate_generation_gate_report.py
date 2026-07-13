from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation_gate import validate_generation_gate_authorization
from cofitok.generation_gate_sources import verify_generation_gate_source_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a generation gate before authorizing the next formal stage."
    )
    parser.add_argument("--gate", required=True)
    parser.add_argument("--stage", required=True, choices=["scaling", "full"])
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Verify bound source files without requiring a passing scientific gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.gate).open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    if gate.get("stage") != args.stage:
        raise ValueError(f"expected {args.stage} generation gate")
    source_evidence = verify_generation_gate_source_reports(gate)
    evidence = (
        {"stage": args.stage, "sources": source_evidence}
        if args.sources_only
        else {
            **validate_generation_gate_authorization(
                gate, expected_stage=args.stage
            ),
            "sources": source_evidence,
        }
    )
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
