from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation.factorization_quality_regression import (
    OUTPUT_ROOT,
    QUALITY_BRIDGE_ROOT,
    build_preparation,
)
from cofitok.inference_replay import prepare_manifest
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the non-authorizing exact-100K matched factorization "
            "quality-regression diagnostic."
        )
    )
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--quality-bridge-root", default=QUALITY_BRIDGE_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    git = git_provenance(PROJECT_ROOT)
    expected = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if git != expected:
        raise ValueError("factorization-regression preparation checkout identity differs")
    report = build_preparation(
        execution_git=git,
        quality_bridge_root=args.quality_bridge_root,
        output_root=args.output_root,
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
