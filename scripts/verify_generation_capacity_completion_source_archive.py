from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object

try:
    from scripts.archive_generation_capacity_completion_sources import (
        build_source_archive,
    )
except ModuleNotFoundError:
    from archive_generation_capacity_completion_sources import build_source_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the exact hard-link archive of matched step-50K sources."
    )
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.archive)
    if identity["sha256"] != args.expected_archive_sha256:
        raise ValueError("capacity completion source archive SHA256 differs")
    actual = read_json_object(args.archive, name="capacity completion source archive")
    expected = build_source_archive(
        decision_path=args.decision.resolve(),
        expected_decision_sha256=args.expected_decision_sha256,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        output_root=args.output_root.resolve(),
        create_missing=False,
    )
    if actual != expected:
        raise ValueError("capacity completion source archive is not reproducible")
    print(args.archive)


if __name__ == "__main__":
    main()
