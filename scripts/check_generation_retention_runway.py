from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cofitok.checkpoint_retention import build_retention_runway_report
from cofitok.reporting import write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check generation runway without counting unapproved archive candidates."
    )
    parser.add_argument("--retention-inventory", type=Path, required=True)
    parser.add_argument(
        "--expected-retention-inventory-sha256",
        required=True,
        help="Pinned SHA256 from an independently replayed physical inventory.",
    )
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--required-free-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    usage = shutil.disk_usage(args.path)
    inventory = json.loads(args.retention_inventory.read_text(encoding="utf-8"))
    report = build_retention_runway_report(
        retention_report=inventory,
        retention_report_path=args.retention_inventory,
        expected_retention_report_sha256=args.expected_retention_inventory_sha256,
        filesystem_path=args.path,
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        required_free_bytes=args.required_free_bytes,
    )
    write_json_report(args.output, report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(78)


if __name__ == "__main__":
    main()
