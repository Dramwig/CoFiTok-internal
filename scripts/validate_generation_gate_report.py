from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation_gate import validate_generation_gate_authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a generation gate before authorizing the next formal stage."
    )
    parser.add_argument("--gate", required=True)
    parser.add_argument("--stage", required=True, choices=["scaling", "full"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.gate).open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    evidence = validate_generation_gate_authorization(gate, expected_stage=args.stage)
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
