from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.min_snr_pilot import validate_evaluation_arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one completed matched Min-SNR pilot evaluation arm."
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--expected-prefix-budget", type=int, required=True)
    parser.add_argument("--sampling-preflight", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--class-fidelity", required=True)
    parser.add_argument("--checkpoint-eval", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def main() -> None:
    args = parse_args()
    row = validate_evaluation_arm(
        {
            "sampling_preflight": _read(args.sampling_preflight),
            "generation": _read(args.generation),
            "class_fidelity": _read(args.class_fidelity),
            "checkpoint_eval": _read(args.checkpoint_eval),
        },
        arm=args.arm,
        expected_prefix_budget=args.expected_prefix_budget,
    )
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
