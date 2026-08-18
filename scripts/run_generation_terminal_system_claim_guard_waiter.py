from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ModuleNotFoundError:  # Windows-only local CPU tests.
    fcntl = None  # type: ignore[assignment]

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report

try:
    import build_generation_terminal_system_claim_guard as terminal_guard
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import build_generation_terminal_system_claim_guard as terminal_guard


SCHEMA_VERSION = 1
ROLE = "generation_terminal_system_claim_guard_waiter"
AUTHORIZATION_BOUNDARY = {
    "cpu_only_evidence_binding_allowed": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "promotion_or_release_allowed": False,
    "process_signals_allowed": False,
    "upstream_decisions_modified": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact terminal quality, statistical, visual, and runtime "
            "artifacts, then build a CPU-only non-authorizing system claim guard."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--statistical-claim-guard", type=Path, required=True)
    parser.add_argument("--statistical-waiter-status", type=Path, required=True)
    parser.add_argument("--visual-audit-waiter-status", type=Path, required=True)
    parser.add_argument("--runtime-claim-guard", type=Path, required=True)
    parser.add_argument("--runtime-waiter-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()


def validate_self_git(
    project: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    root = reject_symlink_chain(project, name="terminal system guard project")
    if not root.is_dir():
        raise FileNotFoundError(f"terminal system guard project is missing: {root}")
    observed = {**git_provenance(root), "tree": _tree(root)}
    expected = {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if observed != expected:
        raise ValueError(f"terminal system guard Git identity differs: {observed}")
    return observed


def _upstream_failure(path: Path, *, label: str) -> str | None:
    if not path.is_file():
        return None
    report = read_json_object(path, name=label)
    status = str(report.get("status", ""))
    phase = str(report.get("phase", ""))
    if status in {"failed", "error"} or phase in {"failed", "error"}:
        return f"{label}:{status or phase}:{report.get('detail', '')}"
    return None


def _base_status(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "poll_seconds": args.poll_seconds,
        "timeout_seconds": args.timeout_seconds,
        "authorization_boundary": AUTHORIZATION_BOUNDARY,
        "expected": {
            "git": {
                "revision": args.expected_revision,
                "tree": args.expected_tree,
                "branch": args.expected_branch,
                "tracked_dirty": False,
            },
            "quality_output_root": args.quality_output_root.resolve().as_posix(),
            "sources": {
                "quality_result": args.quality_result.resolve().as_posix(),
                "statistical_claim_guard": (
                    args.statistical_claim_guard.resolve().as_posix()
                ),
                "visual_audit_waiter_status": (
                    args.visual_audit_waiter_status.resolve().as_posix()
                ),
                "runtime_claim_guard": args.runtime_claim_guard.resolve().as_posix(),
            },
            "output": args.output.resolve().as_posix(),
        },
    }


def _write_status(
    path: Path,
    base: dict[str, Any],
    *,
    status: str,
    detail: str,
    polls: int,
    **extra: Any,
) -> None:
    write_json_report(
        path,
        {
            **base,
            "updated_at": _utc_now(),
            "status": status,
            "detail": detail,
            "polls": polls,
            **extra,
        },
    )


def _source_identities(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "quality_result": file_identity(
            reject_symlink_chain(args.quality_result, name="terminal quality result")
        ),
        "statistical_claim_guard": file_identity(
            reject_symlink_chain(
                args.statistical_claim_guard,
                name="terminal statistical claim guard",
            )
        ),
        "visual_audit_waiter_status": file_identity(
            reject_symlink_chain(
                args.visual_audit_waiter_status,
                name="terminal visual-audit waiter status",
            )
        ),
        "runtime_claim_guard": file_identity(
            reject_symlink_chain(
                args.runtime_claim_guard,
                name="terminal runtime claim guard",
            )
        ),
    }


def run(args: argparse.Namespace) -> int:
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("terminal system guard waiter timings must be positive")
    status_output = args.status_output.resolve()
    lock_path = args.lock.resolve()
    output = args.output.resolve()
    if status_output.is_symlink() or lock_path.is_symlink() or output.is_symlink():
        raise ValueError("terminal system guard waiter outputs must not be symlinks")
    status_output.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = _base_status(args)
    polls = 0
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "another terminal system guard waiter owns the lock"
                ) from error
        try:
            while True:
                git = validate_self_git(
                    args.project,
                    expected_revision=args.expected_revision,
                    expected_tree=args.expected_tree,
                    expected_branch=args.expected_branch,
                )
                failures = [
                    failure
                    for failure in (
                        _upstream_failure(
                            args.statistical_waiter_status,
                            label="statistical claim waiter",
                        ),
                        _upstream_failure(
                            args.visual_audit_waiter_status,
                            label="visual-audit waiter",
                        ),
                        _upstream_failure(
                            args.runtime_waiter_status,
                            label="runtime claim waiter",
                        ),
                    )
                    if failure is not None
                ]
                if failures:
                    _write_status(
                        status_output,
                        base,
                        status="failed",
                        detail="upstream_terminal_evidence_failed",
                        polls=polls,
                        git=git,
                        failures=failures,
                        guard=None,
                    )
                    return 1
                source_paths = {
                    "quality_result": args.quality_result,
                    "statistical_claim_guard": args.statistical_claim_guard,
                    "visual_audit_waiter_status": args.visual_audit_waiter_status,
                    "runtime_claim_guard": args.runtime_claim_guard,
                }
                missing = [
                    name for name, path in source_paths.items() if not path.is_file()
                ]
                if args.visual_audit_waiter_status.is_file():
                    visual_status = read_json_object(
                        args.visual_audit_waiter_status,
                        name="terminal visual-audit waiter status",
                    )
                    if visual_status.get("status") != "completed":
                        missing.append("visual_audit_waiter_terminal_status")
                if missing:
                    if time.monotonic() - started >= args.timeout_seconds:
                        _write_status(
                            status_output,
                            base,
                            status="failed",
                            detail="terminal_system_guard_waiter_timeout",
                            polls=polls,
                            git=git,
                            missing=missing,
                            guard=None,
                        )
                        return 1
                    _write_status(
                        status_output,
                        base,
                        status="waiting",
                        detail="waiting_for_exact_terminal_system_sources",
                        polls=polls,
                        git=git,
                        missing=missing,
                        guard=None,
                    )
                    polls += 1
                    time.sleep(args.poll_seconds)
                    continue

                identities = _source_identities(args)
                report = terminal_guard.build_guard(
                    quality_result_path=args.quality_result,
                    expected_quality_result_sha256=identities["quality_result"][
                        "sha256"
                    ],
                    statistical_claim_guard_path=args.statistical_claim_guard,
                    expected_statistical_claim_guard_sha256=identities[
                        "statistical_claim_guard"
                    ]["sha256"],
                    visual_audit_waiter_status_path=args.visual_audit_waiter_status,
                    expected_visual_audit_waiter_status_sha256=identities[
                        "visual_audit_waiter_status"
                    ]["sha256"],
                    runtime_claim_guard_path=args.runtime_claim_guard,
                    expected_runtime_claim_guard_sha256=identities[
                        "runtime_claim_guard"
                    ]["sha256"],
                    quality_output_root=args.quality_output_root,
                )
                guard_identity = prepare_manifest(
                    output,
                    report,
                    resume=output.is_file(),
                    overwrite=False,
                )
                _write_status(
                    status_output,
                    base,
                    status="completed",
                    detail="terminal_system_claim_guard_source_revalidated",
                    polls=polls,
                    git=git,
                    sources=identities,
                    guard=guard_identity,
                    guard_status=report["status"],
                    guard_decision=report["decision"],
                )
                return 0
        except Exception as error:
            _write_status(
                status_output,
                base,
                status="failed",
                detail=f"{type(error).__name__}:{error}",
                polls=polls,
                guard=None,
            )
            raise


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
