from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import write_json_report

try:
    import build_generation_runtime_claim_guard_comparison as comparison_builder
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import build_generation_runtime_claim_guard_comparison as comparison_builder


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_runtime_claim_guard_strict_comparison_waiter"
DEPLOYMENT_SCHEMA_VERSION = 1
DEPLOYMENT_ROLE = "generation_runtime_claim_guard_strict_comparison_deployment"
SCOPE = {
    "cpu_only": True,
    "non_authorizing": True,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "source_waiter_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "full_300k_launch_allowed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _git_identity(project: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _terminal_state(status_path: Path, guard_path: Path, *, label: str) -> dict[str, Any]:
    if not status_path.is_file():
        return {"status": "waiting", "detail": f"waiting_for_{label}_status"}
    status = read_json_object(status_path, name=f"{label} waiter status")
    state = str(status.get("status", ""))
    if state == "failed":
        return {
            "status": "failed",
            "detail": f"{label}_waiter_failed",
            "source_detail": status.get("detail"),
        }
    if state == "pass" and status.get("phase") == "completed" and guard_path.is_file():
        return {"status": "ready", "detail": f"{label}_guard_ready"}
    if state not in {"waiting", "pass"}:
        raise ValueError(f"{label} waiter state differs")
    return {"status": "waiting", "detail": f"waiting_for_{label}_guard"}


def _write_pid(path: Path, *, expected: Mapping[str, Any]) -> None:
    write_json_report(
        path,
        {
            "schema_version": 1,
            "role": WAITER_ROLE,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "expected_control_revision": expected["control_git"]["revision"],
            "started_at": _utc_now(),
        },
    )


def _remove_pid(path: Path) -> None:
    if not path.is_file():
        return
    try:
        report = read_json_object(path, name="runtime claim comparison waiter PID")
    except ValueError:
        return
    if report.get("role") == WAITER_ROLE and int(report.get("pid", -1)) == os.getpid():
        path.unlink()


def _status(
    *,
    status: str,
    detail: str,
    expected: Mapping[str, Any],
    deployment_receipt: Mapping[str, Any],
    sources: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": WAITER_SCHEMA_VERSION,
        "role": WAITER_ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected": dict(expected),
        "deployment_receipt": dict(deployment_receipt),
        "sources": dict(sources) if sources is not None else None,
        "comparison": dict(comparison) if comparison is not None else None,
        "scope": SCOPE,
        "updated_at": _utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for canonical and strict runtime claim guards, compare their "
            "claim boundaries, and publish one CPU-only non-authorizing report."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--canonical-status", type=Path, required=True)
    parser.add_argument("--canonical-guard", type=Path, required=True)
    parser.add_argument("--strict-status", type=Path, required=True)
    parser.add_argument("--strict-guard", type=Path, required=True)
    parser.add_argument("--wrapper-receipt", type=Path, required=True)
    parser.add_argument("--expected-wrapper-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-canonical-control-revision", required=True)
    parser.add_argument("--expected-strict-control-revision", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--require-recovery-binding", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        parser.error("runtime claim comparison waiter timing is invalid")
    return args


def _run_locked(args: argparse.Namespace) -> int:
    require_recovery_binding = bool(
        getattr(args, "require_recovery_binding", False)
    )
    project = reject_symlink_chain(
        args.project,
        name="runtime claim comparison control checkout",
    ).resolve()
    quality_root = reject_symlink_chain(
        args.quality_output_root,
        name="quality bridge output root",
    ).resolve()
    paths = {
        "canonical_status": reject_symlink_chain(
            args.canonical_status, name="canonical runtime claim status"
        ).resolve(),
        "canonical_guard": reject_symlink_chain(
            args.canonical_guard, name="canonical runtime claim guard"
        ).resolve(),
        "strict_status": reject_symlink_chain(
            args.strict_status, name="strict runtime claim status"
        ).resolve(),
        "strict_guard": reject_symlink_chain(
            args.strict_guard, name="strict runtime claim guard"
        ).resolve(),
        "wrapper_receipt": reject_symlink_chain(
            args.wrapper_receipt, name="strict replay wrapper receipt"
        ).resolve(),
        "output_root": reject_symlink_chain(
            args.output_root, name="runtime claim comparison output root"
        ).resolve(),
        "status_output": reject_symlink_chain(
            args.status_output, name="runtime claim comparison status"
        ).resolve(),
        "pid_file": reject_symlink_chain(
            args.pid_file, name="runtime claim comparison PID"
        ).resolve(),
        "deployment_output": reject_symlink_chain(
            args.deployment_receipt_output,
            name="runtime claim comparison deployment receipt",
        ).resolve(),
        "comparison_output": reject_symlink_chain(
            args.comparison_output, name="runtime claim comparison output"
        ).resolve(),
    }
    output_root = paths["output_root"]
    if (
        output_root == quality_root
        or not all(
            _is_within(paths[name], quality_root)
            for name in (
                "canonical_status",
                "canonical_guard",
                "strict_status",
                "strict_guard",
                "wrapper_receipt",
                "output_root",
            )
        )
        or not all(
            _is_within(paths[name], output_root)
            for name in (
                "status_output",
                "pid_file",
                "deployment_output",
                "comparison_output",
            )
        )
    ):
        raise ValueError("runtime claim comparison path scope differs")
    control_git = {
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    if _git_identity(project) != control_git:
        raise ValueError("runtime claim comparison control checkout differs")
    wrapper_identity = file_identity(paths["wrapper_receipt"])
    if wrapper_identity["sha256"] != args.expected_wrapper_receipt_sha256:
        raise ValueError("strict replay wrapper receipt SHA256 differs")
    expected = {
        "control_git": control_git,
        "training_git": {
            "revision": args.expected_training_revision,
            "branch": args.expected_training_branch,
        },
        "source_controls": {
            "canonical": args.expected_canonical_control_revision,
            "strict": args.expected_strict_control_revision,
        },
        "sources": {
            name: paths[name].as_posix()
            for name in (
                "canonical_status",
                "canonical_guard",
                "strict_status",
                "strict_guard",
                "wrapper_receipt",
            )
        },
        "comparison_output": paths["comparison_output"].as_posix(),
        "require_recovery_binding": require_recovery_binding,
    }
    deployment = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "role": DEPLOYMENT_ROLE,
        "status": "pass",
        "expected": expected,
        "wrapper_receipt": wrapper_identity,
        "scope": SCOPE,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    deployment_identity = prepare_manifest(
        paths["deployment_output"],
        deployment,
        resume=paths["deployment_output"].is_file(),
        overwrite=False,
    )
    _write_pid(paths["pid_file"], expected=expected)

    def publish(
        *,
        status: str,
        detail: str,
        sources: Mapping[str, Any] | None = None,
        comparison: Mapping[str, Any] | None = None,
    ) -> None:
        write_json_report(
            paths["status_output"],
            _status(
                status=status,
                detail=detail,
                expected=expected,
                deployment_receipt=deployment_identity,
                sources=sources,
                comparison=comparison,
            ),
        )

    deadline = time.monotonic() + args.timeout_seconds
    while True:
        if _git_identity(project) != control_git:
            raise ValueError("runtime claim comparison control checkout changed")
        if file_identity(paths["wrapper_receipt"]) != wrapper_identity:
            raise ValueError("strict replay wrapper receipt changed")
        canonical_state = _terminal_state(
            paths["canonical_status"],
            paths["canonical_guard"],
            label="canonical",
        )
        strict_state = _terminal_state(
            paths["strict_status"],
            paths["strict_guard"],
            label="strict",
        )
        if "failed" in {canonical_state["status"], strict_state["status"]}:
            publish(
                status="failed",
                detail=(
                    canonical_state["detail"]
                    if canonical_state["status"] == "failed"
                    else strict_state["detail"]
                ),
                sources={"canonical": canonical_state, "strict": strict_state},
            )
            return 1
        if canonical_state["status"] == strict_state["status"] == "ready":
            break
        publish(
            status="waiting",
            detail=(
                strict_state["detail"]
                if canonical_state["status"] == "ready"
                else canonical_state["detail"]
            ),
            sources={"canonical": canonical_state, "strict": strict_state},
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            publish(status="failed", detail="runtime_claim_comparison_waiter_timeout")
            return 1
        time.sleep(min(args.poll_seconds, remaining))

    source_identities = {
        name: file_identity(paths[name])
        for name in (
            "canonical_status",
            "canonical_guard",
            "strict_status",
            "strict_guard",
            "wrapper_receipt",
        )
    }
    report = comparison_builder.build_comparison(
        canonical_status_path=paths["canonical_status"],
        expected_canonical_status_sha256=source_identities["canonical_status"][
            "sha256"
        ],
        canonical_guard_path=paths["canonical_guard"],
        expected_canonical_guard_sha256=source_identities["canonical_guard"]["sha256"],
        strict_status_path=paths["strict_status"],
        expected_strict_status_sha256=source_identities["strict_status"]["sha256"],
        strict_guard_path=paths["strict_guard"],
        expected_strict_guard_sha256=source_identities["strict_guard"]["sha256"],
        wrapper_receipt_path=paths["wrapper_receipt"],
        expected_wrapper_receipt_sha256=wrapper_identity["sha256"],
        expected_canonical_control_revision=args.expected_canonical_control_revision,
        expected_strict_control_revision=args.expected_strict_control_revision,
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
        require_recovery_binding=require_recovery_binding,
    )
    if any(file_identity(paths[name]) != identity for name, identity in source_identities.items()):
        raise ValueError("runtime claim comparison sources changed during build")
    comparison_identity = prepare_manifest(
        paths["comparison_output"],
        report,
        resume=paths["comparison_output"].is_file(),
        overwrite=False,
    )
    publish(
        status="pass",
        detail=str(report["decision"]),
        sources=source_identities,
        comparison={
            "identity": comparison_identity,
            "status": report["status"],
            "decision": report["decision"],
            "claim_policy": report["claim_policy"],
        },
    )
    return 0


def run_waiter(args: argparse.Namespace) -> int:
    output_root = reject_symlink_chain(
        args.output_root,
        name="runtime claim comparison output root",
    ).resolve()
    with exclusive_output_lock(output_root, role=WAITER_ROLE):
        try:
            return _run_locked(args)
        except Exception as error:
            status_output = reject_symlink_chain(
                args.status_output,
                name="runtime claim comparison status",
            ).resolve()
            if _is_within(status_output, output_root):
                write_json_report(
                    status_output,
                    {
                        "schema_version": WAITER_SCHEMA_VERSION,
                        "role": WAITER_ROLE,
                        "status": "failed",
                        "detail": f"{type(error).__name__}: {error}",
                        "pid": os.getpid(),
                        "scope": SCOPE,
                        "updated_at": _utc_now(),
                    },
                )
            raise
        finally:
            pid_file = reject_symlink_chain(
                args.pid_file,
                name="runtime claim comparison PID",
            ).resolve()
            if _is_within(pid_file, output_root):
                _remove_pid(pid_file)


def main() -> int:
    args = parse_args()
    try:
        return run_waiter(args)
    except OutputLockError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
