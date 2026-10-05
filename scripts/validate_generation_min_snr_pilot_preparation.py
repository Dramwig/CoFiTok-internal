from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.min_snr_pilot import validate_preparation
from cofitok.reporting import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Min-SNR pilot preparation.")
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    path = Path(args.preparation)
    if file_sha256(path) != args.expected_sha256:
        raise ValueError("Min-SNR pilot preparation SHA256 differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Min-SNR pilot preparation is not an object")
    validate_preparation(value)
    print(args.expected_sha256)


if __name__ == "__main__":
    main()

