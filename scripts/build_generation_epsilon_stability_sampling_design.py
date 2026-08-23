from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation import build_epsilon_stability_sampling_design
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic, permanently non-authorizing matched "
            "epsilon-stability sampling design."
        )
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json_report(
        Path(args.output),
        build_epsilon_stability_sampling_design(),
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
