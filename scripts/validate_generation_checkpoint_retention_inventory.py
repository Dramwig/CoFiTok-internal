from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.checkpoint_retention import verify_checkpoint_retention_inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a read-only checkpoint retention inventory."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.inventory.read_text(encoding="utf-8"))
    verified = verify_checkpoint_retention_inventory(report)
    print(json.dumps(verified["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
