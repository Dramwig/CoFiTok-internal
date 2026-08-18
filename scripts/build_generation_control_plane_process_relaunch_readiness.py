from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from cofitok.generation_control_continuity import git_state
from cofitok.generation_control_process_relaunch import (
    assess_process_relaunch_readiness,
    build_process_relaunch_approval,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess whether every selected capacity control process is absent and "
            "optionally issue a source-bound CPU-control relaunch approval."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--expected-formal-revision", required=True)
    parser.add_argument("--expected-formal-tree", required=True)
    parser.add_argument("--expected-formal-branch", required=True)
    parser.add_argument("--expected-formal-porcelain-count", type=int, required=True)
    parser.add_argument("--expected-formal-porcelain-sha256", required=True)
    parser.add_argument("--expected-process-count", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--issue-approval", action="store_true")
    parser.add_argument("--approval-output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def _formal_state(args: argparse.Namespace) -> dict:
    project = args.formal_project.resolve()
    state = git_state(project)
    porcelain = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=project,
        check=True,
        capture_output=True,
    ).stdout
    observed = {
        **state,
        "porcelain_count": len(porcelain.splitlines()),
        "porcelain_sha256": hashlib.sha256(porcelain).hexdigest(),
    }
    expected = {
        "path": project.as_posix(),
        "revision": args.expected_formal_revision,
        "tree": args.expected_formal_tree,
        "branch": args.expected_formal_branch,
        "tracked_dirty": False,
        "porcelain_count": args.expected_formal_porcelain_count,
        "porcelain_sha256": args.expected_formal_porcelain_sha256,
    }
    if observed != expected:
        raise ValueError("formal checkout snapshot differs")
    return observed


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    implementation = git_state(project)
    expected_implementation = {
        "path": project.as_posix(),
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if implementation != expected_implementation:
        raise ValueError("process relaunch implementation Git state differs")
    if args.output.exists():
        raise FileExistsError(f"process relaunch readiness exists: {args.output}")
    if args.issue_approval and args.approval_output is None:
        raise ValueError("--issue-approval requires --approval-output")
    if args.approval_output is not None and args.approval_output.exists():
        raise FileExistsError(
            f"process relaunch approval exists: {args.approval_output}"
        )
    report = assess_process_relaunch_readiness(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        standing_authorization_path=args.standing_authorization,
        expected_standing_authorization_sha256=args.expected_standing_authorization_sha256,
        formal_git=_formal_state(args),
        implementation_git=implementation,
        expected_process_count=args.expected_process_count,
    )
    write_json_report(args.output, report)
    result: dict[str, object] = {
        "readiness": file_identity(args.output),
        "result": report,
    }
    if args.issue_approval:
        approval = build_process_relaunch_approval(
            readiness_path=args.output,
            expected_readiness_sha256=file_identity(args.output)["sha256"],
        )
        write_json_report(args.approval_output, approval)
        result["approval"] = file_identity(args.approval_output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_ready and report["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
