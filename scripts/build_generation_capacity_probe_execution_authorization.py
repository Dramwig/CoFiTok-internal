from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation.capacity_probe_execution import (
    build_capacity_probe_execution_authorization,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output-root", required=True)


def build_from_paths(
    args: argparse.Namespace,
    *,
    require_current_git: bool,
) -> dict:
    preparation_identity = file_identity(args.preparation)
    standing_identity = file_identity(args.standing_authorization)
    if preparation_identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("capacity probe preparation SHA256 differs")
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    expected_git = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if require_current_git and git_provenance(args.project_root) != expected_git:
        raise ValueError("capacity probe execution authorization Git identity differs")
    return build_capacity_probe_execution_authorization(
        preparation=read_json_object(
            args.preparation,
            name="capacity probe preparation",
        ),
        preparation_identity=preparation_identity,
        standing_authorization=read_json_object(
            args.standing_authorization,
            name="standing experiment authorization",
        ),
        standing_authorization_identity=standing_identity,
        execution_git=expected_git,
        output_root=Path(args.output_root).resolve().as_posix(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact capacity-probe-only execution authorization from the "
            "user's standing experiment instruction."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"capacity probe execution authorization exists: {args.output}"
        )
    report = build_from_paths(args, require_current_git=True)
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
