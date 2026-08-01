from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation_sample_archive import audit_generation_sample_archive
from cofitok.reporting import write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify an offline archive of completed generation sample roots."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-root", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_generation_sample_archive(
        archive_path=args.archive,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_roots=args.expected_root,
    )
    write_json_report(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
