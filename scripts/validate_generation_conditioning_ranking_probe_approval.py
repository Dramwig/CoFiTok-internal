from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.conditioning_ranking_probe import (
    validate_conditioning_ranking_probe_approval,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the exact execution-only class-ranking probe approval."
    )
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--expected-output-root", required=True)
    args = parser.parse_args()
    with args.approval.open(encoding="utf-8") as handle:
        approval = json.load(handle)
    validate_conditioning_ranking_probe_approval(
        approval,
        expected_revision=args.expected_revision,
        expected_preparation_sha256=args.expected_preparation_sha256,
        expected_output_root=args.expected_output_root,
    )
    print(args.approval)


if __name__ == "__main__":
    main()
