from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cofitok.generation_control_process_snapshot import (
    AUTHORIZATION_BOUNDARY,
    EFFECTS,
    VERIFICATION_ROLE,
    VERIFICATION_SCHEMA_VERSION,
    verify_process_relaunch_manifest,
)
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a capacity control-process relaunch snapshot as inert data and "
            "optionally require every original live process identity to still match."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-process-count", type=int, default=17)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"process snapshot verification exists: {args.output}")
    try:
        report = verify_process_relaunch_manifest(
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_process_count=args.expected_process_count,
            require_live=args.require_live,
        )
    except Exception as error:
        report = {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "role": VERIFICATION_ROLE,
            "status": "failed",
            "complete": False,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "manifest": args.manifest.resolve().as_posix(),
            "expected_process_count": args.expected_process_count,
            "live_verification_required": args.require_live,
            "error_type": type(error).__name__,
            "detail": str(error),
            "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
            "effects": dict(EFFECTS),
        }
        write_json_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    write_json_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
