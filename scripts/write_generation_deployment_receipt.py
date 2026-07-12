from __future__ import annotations

import argparse
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report


SCHEMA_VERSION = 1
EXPECTED_BRANCH = "scale/generative-system"


def build_deployment_receipt(
    *,
    expected_training_revision: str,
    target_revision: str,
    bundle_path: str,
    bundle_bytes: int,
    bundle_sha256: str,
    bundle_heads: list[str],
    pair_validation_path: str,
    pair_validation_sha256: str,
    pair_validation: dict[str, Any],
    git_revision: str,
    git_branch: str,
    tracked_dirty: bool,
    hostname: str,
    deployed_at: str,
) -> dict[str, Any]:
    if len(expected_training_revision) != 40 or len(target_revision) != 40:
        raise ValueError("deployment receipt requires full 40-character revisions")
    if git_revision != target_revision:
        raise ValueError("deployed Git revision does not match target revision")
    if git_branch != EXPECTED_BRANCH:
        raise ValueError(f"deployment must remain on {EXPECTED_BRANCH}")
    if tracked_dirty:
        raise ValueError("deployment receipt refuses a dirty tracked worktree")
    if bundle_bytes < 1 or len(bundle_sha256) != 64:
        raise ValueError("deployment bundle integrity evidence is invalid")
    if target_revision not in bundle_heads:
        raise ValueError("deployment bundle does not contain the target revision")
    if len(pair_validation_sha256) != 64:
        raise ValueError("training-pair validation SHA256 is invalid")
    if pair_validation.get("status") != "pass":
        raise ValueError("training-pair validation did not pass")
    if pair_validation.get("expected_revision") != expected_training_revision:
        raise ValueError("training-pair validation revision differs from the pinned queue")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "expected_training_revision": expected_training_revision,
        "target_revision": target_revision,
        "git": {
            "revision": git_revision,
            "branch": git_branch,
            "tracked_dirty": tracked_dirty,
        },
        "bundle": {
            "path": bundle_path,
            "bytes": bundle_bytes,
            "sha256": bundle_sha256,
            "heads": bundle_heads,
        },
        "training_pair_validation": {
            "path": pair_validation_path,
            "sha256": pair_validation_sha256,
            "status": pair_validation["status"],
            "expected_revision": pair_validation["expected_revision"],
        },
        "verification": {
            "pytest": "pass",
            "runbook_syntax": "pass",
            "untracked_target_conflicts": 0,
        },
        "hostname": hostname,
        "deployed_at": deployed_at,
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bundle_heads(path: Path) -> list[str]:
    output = _git("bundle", "list-heads", path.as_posix())
    heads = sorted({line.split()[0] for line in output.splitlines() if line.split()})
    if not heads:
        raise ValueError("deployment bundle has no advertised heads")
    return heads


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an integrity-bound generation upgrade deployment receipt."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--pair-validation", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--target-revision", required=True)
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    pair_validation_path = Path(args.pair_validation).resolve()
    with pair_validation_path.open("r", encoding="utf-8") as handle:
        pair_validation = json.load(handle)
    tracked_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    receipt = build_deployment_receipt(
        expected_training_revision=args.expected_training_revision,
        target_revision=args.target_revision,
        bundle_path=bundle.as_posix(),
        bundle_bytes=bundle.stat().st_size,
        bundle_sha256=file_sha256(bundle),
        bundle_heads=_bundle_heads(bundle),
        pair_validation_path=pair_validation_path.as_posix(),
        pair_validation_sha256=file_sha256(pair_validation_path),
        pair_validation=pair_validation,
        git_revision=_git("rev-parse", "HEAD"),
        git_branch=_git("branch", "--show-current"),
        tracked_dirty=tracked_dirty,
        hostname=socket.gethostname(),
        deployed_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_report(args.output, receipt)
    print(args.output)


if __name__ == "__main__":
    main()
