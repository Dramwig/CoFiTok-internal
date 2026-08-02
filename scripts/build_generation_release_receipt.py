from __future__ import annotations

import argparse
import json

from cofitok.generation import write_generation_release_receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a consumer-verifiable inference receipt after the terminal "
            "generation completion audit passes."
        )
    )
    parser.add_argument("--completion-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = write_generation_release_receipt(
        args.completion_audit,
        args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
