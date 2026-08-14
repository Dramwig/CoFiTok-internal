from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation_control_continuity import restore_continuity_archive
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize verified capacity-generation control assets into bounded, "
            "absent paths. This command never launches or signals a process."
        )
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = restore_continuity_archive(
        archive_root=args.archive_root,
        expected_manifest_sha256=args.expected_manifest_sha256,
        destination_root=args.destination_root,
    )
    write_json_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
