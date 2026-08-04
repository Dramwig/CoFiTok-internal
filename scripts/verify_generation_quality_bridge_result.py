from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from scripts.build_generation_quality_bridge_result import (
    _common_arguments,
    build_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently replay a source-bound full-data 100K quality bridge "
            "terminal result."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_identity = file_identity(args.result)
    if result_identity["sha256"] != args.expected_result_sha256:
        raise ValueError("quality bridge result SHA256 differs")
    actual = read_json_object(args.result, name="quality bridge terminal result")
    expected = build_from_args(args)
    if actual != expected:
        raise ValueError("quality bridge terminal result is not reproducible")
    print(
        json.dumps(
            {
                "status": "verified",
                "result": result_identity,
                "quality_screen": actual["quality_screen"],
                "authorization_boundary": actual["authorization_boundary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
