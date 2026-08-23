from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation import build_epsilon_stability_sampling_design
from cofitok.reporting import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and validate a matched epsilon-stability sampling design."
        )
    )
    parser.add_argument("--design", required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.design)
    if file_sha256(path) != args.expected_design_sha256:
        raise ValueError("epsilon-stability design SHA256 differs from evidence")
    with path.open("r", encoding="utf-8") as handle:
        actual = json.load(handle)
    expected = build_epsilon_stability_sampling_design()
    if actual != expected:
        raise ValueError("epsilon-stability design does not deterministically replay")
    print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
