from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows unit-test import path
    fcntl = None

from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_value(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_checkout(
    checkout: Path,
    *,
    expected_revision: str,
    expected_branch: str,
    expected_tree: str,
) -> dict[str, Any]:
    revision = git_value(checkout, "rev-parse", "HEAD")
    branch = git_value(checkout, "branch", "--show-current")
    tree = git_value(checkout, "rev-parse", "HEAD^{tree}")
    tracked_status = git_value(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if revision != expected_revision:
        raise ValueError("training checkout revision differs")
    if branch != expected_branch:
        raise ValueError("training checkout branch differs")
    if tree != expected_tree:
        raise ValueError("training checkout tree differs")
    if tracked_status:
        raise ValueError("training checkout has tracked modifications")
    return {
        "path": str(checkout.resolve()),
        "revision": revision,
        "branch": branch,
        "tree": tree,
        "tracked_dirty": False,
    }


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid metrics JSON at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"metrics row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("training metrics are empty")
    return rows


def target_is_ready(run_dir: Path, checkpoint_step: int) -> tuple[bool, str]:
    checkpoint = run_dir / f"checkpoint_step_{checkpoint_step:08d}.pt"
    integrity = checkpoint_integrity_path(checkpoint)
    latest_path = run_dir / "latest.json"
    metrics_path = run_dir / "train_metrics.jsonl"
    if not checkpoint.is_file():
        return False, "checkpoint_missing"
    if not integrity.is_file():
        return False, "integrity_manifest_missing"
    if not latest_path.is_file():
        return False, "latest_pointer_missing"
    if not metrics_path.is_file():
        return False, "metrics_missing"
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "latest_pointer_unreadable"
    latest_step = int(latest.get("step", -1))
    if latest_step > checkpoint_step:
        raise ValueError("latest.json advanced beyond the target before audit")
    if latest_step != checkpoint_step:
        return False, "latest_pointer_not_at_target"
    if latest.get("checkpoint") != checkpoint.name:
        return False, "latest_pointer_checkpoint_mismatch"
    try:
        last_line = next(
            line
            for line in reversed(metrics_path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        )
        last_step = int(json.loads(last_line).get("step", -1))
    except (OSError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
        return False, "metrics_tail_unreadable"
    if last_step < checkpoint_step:
        return False, "metrics_not_at_target"
    return True, "ready"


def audit_checkpoint(
    *,
    run_dir: Path,
    checkpoint_step: int,
    training_checkout: Path,
    expected_revision: str,
    expected_branch: str,
    expected_tree: str,
    expected_dataset_sha256: str,
    expected_runtime_sha256: str,
    effective_batch: int,
) -> dict[str, Any]:
    checkout = verify_checkout(
        training_checkout,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_tree=expected_tree,
    )
    checkpoint = run_dir / f"checkpoint_step_{checkpoint_step:08d}.pt"
    integrity_path = checkpoint_integrity_path(checkpoint)
    integrity = verify_training_checkpoint(checkpoint)
    if int(integrity.get("step", -1)) != checkpoint_step:
        raise ValueError("checkpoint integrity step differs")
    if integrity.get("git_revision") != expected_revision:
        raise ValueError("checkpoint integrity revision differs")
    if integrity.get("git_branch") != expected_branch:
        raise ValueError("checkpoint integrity branch differs")
    if integrity.get("git_dirty") is not False:
        raise ValueError("checkpoint integrity reports a dirty checkout")
    if integrity.get("dataset_identity_sha256") != expected_dataset_sha256:
        raise ValueError("checkpoint dataset identity differs")
    if integrity.get("runtime_environment_sha256") != expected_runtime_sha256:
        raise ValueError("checkpoint runtime environment differs")

    latest_path = run_dir / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    expected_latest = {
        "checkpoint": checkpoint.name,
        "step": checkpoint_step,
        "checkpoint_bytes": int(integrity["checkpoint_bytes"]),
        "checkpoint_sha256": integrity["checkpoint_sha256"],
        "integrity_manifest": integrity_path.name,
        "git_revision": expected_revision,
        "git_branch": expected_branch,
        "git_dirty": False,
        "dataset_identity_sha256": expected_dataset_sha256,
        "runtime_environment_sha256": expected_runtime_sha256,
    }
    for key, expected in expected_latest.items():
        if latest.get(key) != expected:
            raise ValueError(f"latest.json {key} differs")

    metrics_path = run_dir / "train_metrics.jsonl"
    rows = read_metrics(metrics_path)
    steps = [int(row.get("step", -1)) for row in rows]
    if not all(left < right for left, right in zip(steps, steps[1:])):
        raise ValueError("training metrics are not strictly increasing")
    if steps[-1] < checkpoint_step:
        raise ValueError("training metrics have not reached the checkpoint")
    if checkpoint_step not in steps:
        raise ValueError("training metrics omit the checkpoint step")
    for row in rows:
        step = int(row.get("step", -1))
        if int(row.get("samples_seen", -1)) != step * effective_batch:
            raise ValueError("training metrics samples_seen binding differs")
    target_row = rows[steps.index(checkpoint_step)]

    return {
        "schema_version": 1,
        "role": "generation_checkpoint_physical_integrity_milestone_audit",
        "status": "pass",
        "audited_at": utc_now(),
        "scope": {
            "read_only_checkpoint_verification": True,
            "gpu_required": False,
            "training_process_signals_allowed": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
        },
        "training_checkout": checkout,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": {
            "step": checkpoint_step,
            "payload": file_identity(checkpoint),
            "integrity_manifest": file_identity(integrity_path),
            "integrity": integrity,
            "physical_sha256_verified": True,
        },
        "latest_pointer": {
            "identity": file_identity(latest_path),
            "content": latest,
            "exact_target_binding": True,
        },
        "metrics": {
            "identity": file_identity(metrics_path),
            "row_count": len(rows),
            "first_step": steps[0],
            "last_step": steps[-1],
            "strictly_increasing": True,
            "samples_seen_binding": f"samples_seen == step * {effective_batch}",
            "samples_seen_binding_verified": True,
            "target_row": target_row,
        },
    }


def waiter_status(
    *,
    status: str,
    detail: str,
    run_dir: Path,
    checkpoint_step: int,
    audit_output: Path,
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "generation_checkpoint_physical_integrity_milestone_waiter",
        "status": status,
        "detail": detail,
        "updated_at": utc_now(),
        "started_at": started_at,
        "pid": os.getpid(),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_step": checkpoint_step,
        "audit_output": str(audit_output.resolve()),
        "waiter_source": file_identity(Path(__file__)),
        "scope": {
            "read_only_checkpoint_verification": True,
            "gpu_required": False,
            "training_process_signals_allowed": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for one generation checkpoint and physically verify its integrity, "
            "latest.json binding, Git/data/runtime provenance, and metric continuity."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--training-checkout", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_step < 1 or args.effective_batch < 1:
        raise ValueError("checkpoint step and effective batch must be positive")
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("poll and timeout seconds must be positive")
    for label, digest in (
        ("dataset", args.expected_dataset_sha256),
        ("runtime", args.expected_runtime_sha256),
        ("source", args.expected_source_sha256),
    ):
        if len(digest) != 64:
            raise ValueError(f"expected {label} SHA256 is malformed")
    source_identity = file_identity(Path(__file__))
    if source_identity["sha256"] != args.expected_source_sha256:
        raise ValueError("waiter source SHA256 differs")

    lock_path = args.status_output.with_suffix(args.status_output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("checkpoint integrity waiter is already active")
            return

    started_at = utc_now()
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while True:
            ready, detail = target_is_ready(args.run_dir, args.checkpoint_step)
            if ready:
                report = audit_checkpoint(
                    run_dir=args.run_dir,
                    checkpoint_step=args.checkpoint_step,
                    training_checkout=args.training_checkout,
                    expected_revision=args.expected_revision,
                    expected_branch=args.expected_branch,
                    expected_tree=args.expected_tree,
                    expected_dataset_sha256=args.expected_dataset_sha256,
                    expected_runtime_sha256=args.expected_runtime_sha256,
                    effective_batch=args.effective_batch,
                )
                write_json_atomic(args.audit_output, report)
                status = waiter_status(
                    status="pass",
                    detail="checkpoint_physical_integrity_verified",
                    run_dir=args.run_dir,
                    checkpoint_step=args.checkpoint_step,
                    audit_output=args.audit_output,
                    started_at=started_at,
                )
                status["audit_output_identity"] = file_identity(args.audit_output)
                write_json_atomic(args.status_output, status)
                print(args.audit_output)
                return

            write_json_atomic(
                args.status_output,
                waiter_status(
                    status="waiting",
                    detail=detail,
                    run_dir=args.run_dir,
                    checkpoint_step=args.checkpoint_step,
                    audit_output=args.audit_output,
                    started_at=started_at,
                ),
            )
            if args.once:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("checkpoint integrity waiter timed out")
            time.sleep(args.poll_seconds)
    except Exception as error:
        write_json_atomic(
            args.status_output,
            waiter_status(
                status="failed",
                detail=f"{type(error).__name__}: {error}",
                run_dir=args.run_dir,
                checkpoint_step=args.checkpoint_step,
                audit_output=args.audit_output,
                started_at=started_at,
            ),
        )
        raise


if __name__ == "__main__":
    main()
