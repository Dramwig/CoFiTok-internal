from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.stability_scaling import build_stability_scaling_decision
from cofitok.reporting import file_sha256, write_json_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed multi-seed rollout-stability scaling decision."
    )
    parser.add_argument("--screening-report", type=Path, required=True)
    parser.add_argument(
        "--robust-report",
        type=Path,
        action="append",
        required=True,
        dest="robust_reports",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-robust-reports", type=int, default=2)
    parser.add_argument("--min-robust-images", type=int, default=64)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main() -> int:
    args = _parse_args()
    report = build_stability_scaling_decision(
        screening_report=_load(args.screening_report),
        robust_reports=[_load(path) for path in args.robust_reports],
        min_robust_reports=args.min_robust_reports,
        min_robust_images=args.min_robust_images,
    )
    report["sources"] = {
        "screening_report": _source(args.screening_report),
        "robust_reports": [_source(path) for path in args.robust_reports],
    }
    output_path = args.output_dir / "scaling_decision.json"
    write_json_report(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "status": report["status"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )
    return 2 if args.require_pass and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
