from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.reporting import file_sha256
from scripts.build_generation_quality_bridge_preparation import build_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild and validate a frozen quality-bridge preparation report."
    )
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--promotion-gate", required=True)
    parser.add_argument("--expected-promotion-gate-sha256", required=True)
    parser.add_argument("--cofitok-config", required=True)
    parser.add_argument("--dense-config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparation_path = Path(args.preparation)
    if file_sha256(preparation_path) != args.expected_preparation_sha256:
        raise ValueError("quality bridge preparation SHA256 differs from approval")
    with preparation_path.open("r", encoding="utf-8") as handle:
        actual = json.load(handle)
    expected = build_from_paths(
        promotion_gate_path=args.promotion_gate,
        expected_promotion_gate_sha256=args.expected_promotion_gate_sha256,
        cofitok_config_path=args.cofitok_config,
        dense_config_path=args.dense_config,
    )
    if actual != expected:
        raise ValueError("quality bridge preparation does not deterministically replay")
    print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
