from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.reporting import file_sha256
from scripts.build_generation_epsilon_stability_preparation import (
    add_source_args,
    build_from_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay and validate an epsilon-stability preparation."
    )
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    add_source_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.preparation)
    if file_sha256(path) != args.expected_preparation_sha256:
        raise ValueError("epsilon-stability preparation SHA256 differs")
    with path.open("r", encoding="utf-8") as handle:
        actual = json.load(handle)
    expected = build_from_paths(
        **{
            key: value
            for key, value in vars(args).items()
            if key not in {"preparation", "expected_preparation_sha256"}
        },
        require_output_absent=False,
    )
    if actual != expected:
        raise ValueError("epsilon-stability preparation does not replay")
    print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
