from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.min_snr_pilot import validate_result
from cofitok.reporting import file_sha256
from scripts.build_generation_min_snr_pilot_result import (
    add_source_arguments,
    build_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a matched Min-SNR pilot result.")
    add_source_arguments(parser)
    parser.add_argument("--result", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    args = parser.parse_args()
    path = Path(args.result)
    if file_sha256(path) != args.expected_result_sha256:
        raise ValueError("Min-SNR pilot result SHA256 differs")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict):
        raise ValueError("Min-SNR pilot result is not an object")
    validate_result(actual)
    expected = build_report(args)
    if actual != expected:
        raise ValueError("Min-SNR pilot result does not replay exactly")
    print(args.expected_result_sha256)


if __name__ == "__main__":
    main()
