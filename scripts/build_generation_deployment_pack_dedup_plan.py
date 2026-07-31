from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.deployment_pack_dedup import build_deployment_pack_dedup_plan
from cofitok.reporting import write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a read-only, process-aware deployment Git-pack dedup plan."
    )
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--canonical-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_deployment_pack_dedup_plan(
        deployment_root=args.deployment_root,
        canonical_checkout=args.canonical_checkout,
    )
    write_json_report(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
