from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.checkpoint_retention import build_checkpoint_retention_inventory
from cofitok.reporting import write_json_report


def _reference(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or parts[2] not in {"authoritative", "operational"}:
        raise argparse.ArgumentTypeError(
            "reference root must be LABEL=PATH=authoritative|operational"
        )
    return parts[0], Path(parts[1]), parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a read-only, fail-closed checkpoint retention inventory."
    )
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=_reference, action="append", default=[])
    parser.add_argument("--physical-hash", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_checkpoint_retention_inventory(
        inventory_root=args.inventory_root,
        reference_roots=args.reference_root,
        physical_hash=args.physical_hash,
    )
    write_json_report(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
