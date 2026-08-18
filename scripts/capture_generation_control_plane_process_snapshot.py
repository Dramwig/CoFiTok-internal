from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation_control_continuity import git_state
from cofitok.generation_control_process_snapshot import (
    build_process_relaunch_manifest,
    verify_process_relaunch_manifest,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture an inert, source-bound relaunch recipe for the active capacity "
            "generation control processes without launching, signaling, or querying GPU."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--lineage-report", type=Path, required=True)
    parser.add_argument("--static-continuity-manifest", type=Path, required=True)
    parser.add_argument("--expected-static-continuity-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-process-count", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    expected_git = {
        "path": project.as_posix(),
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    observed_git = git_state(project)
    if observed_git != expected_git:
        raise ValueError("process snapshot builder Git state differs")
    if args.output.exists() or args.verification_output.exists():
        raise FileExistsError("process snapshot output already exists")
    manifest = build_process_relaunch_manifest(
        lineage_report_path=args.lineage_report,
        static_continuity_manifest_path=args.static_continuity_manifest,
        expected_static_continuity_sha256=args.expected_static_continuity_sha256,
        builder_git=observed_git,
        expected_process_count=args.expected_process_count,
    )
    write_json_report(args.output, manifest)
    identity = file_identity(args.output)
    verification = verify_process_relaunch_manifest(
        manifest_path=args.output,
        expected_manifest_sha256=identity["sha256"],
        expected_process_count=args.expected_process_count,
        require_live=True,
    )
    write_json_report(args.verification_output, verification)
    print(
        json.dumps(
            {
                "manifest": identity,
                "verification": file_identity(args.verification_output),
                "result": verification,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
