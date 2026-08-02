from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.frozen_supplemental import (
    build_frozen_posteval_verification,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the immutable stability 50K post-evaluation waiter "
            "completed before a separate supplemental diagnosis starts."
        )
    )
    parser.add_argument("--posteval-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-evaluation-revision", required=True)
    parser.add_argument("--expected-evaluation-branch", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_bound_verification(args: argparse.Namespace) -> dict[str, Any]:
    before = _source(args.posteval_status)
    report = build_frozen_posteval_verification(
        posteval_status=_read(args.posteval_status),
        source=before,
        verifier_git=git_provenance(PROJECT_ROOT),
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
        expected_evaluation_revision=args.expected_evaluation_revision,
        expected_evaluation_branch=args.expected_evaluation_branch,
    )
    if _source(args.posteval_status) != before:
        raise ValueError("post-evaluation status changed while verifying it")
    return report


def main() -> int:
    args = parse_args()
    with exclusive_output_lock(
        args.output,
        role="generation_stability_frozen_posteval_verification",
    ):
        report = build_bound_verification(args)
        if args.output.exists():
            if not args.resume:
                raise FileExistsError(
                    "post-evaluation verification exists; pass --resume to replay it"
                )
            if _read(args.output) != report:
                raise ValueError(
                    "existing post-evaluation verification differs from exact replay"
                )
        else:
            write_json_report(args.output, report)
    print(args.output.resolve().as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
