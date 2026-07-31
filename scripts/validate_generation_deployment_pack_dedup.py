from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.deployment_pack_dedup import verify_deployment_pack_dedup_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a deployment Git-pack hardlink dedup result."
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    args = parser.parse_args()
    report = json.loads(args.result.read_text(encoding="utf-8"))
    verified = verify_deployment_pack_dedup_result(
        report,
        result_path=args.result,
        expected_result_sha256=args.expected_result_sha256,
    )
    print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    main()
