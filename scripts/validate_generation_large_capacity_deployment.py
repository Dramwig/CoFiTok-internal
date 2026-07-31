from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_generation_large_capacity_deployment_receipt import (
        verify_deployment_receipt,
    )
except ModuleNotFoundError:
    from build_generation_large_capacity_deployment_receipt import (
        verify_deployment_receipt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay an isolated large-capacity deployment receipt."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    verified = verify_deployment_receipt(
        payload,
        receipt_path=args.receipt,
        expected_receipt_sha256=args.expected_receipt_sha256,
    )
    print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    main()
