from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MILESTONES = (1000, 2000, 3000, 4000, 5000)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid metrics JSON at line {line_number}"
                ) from error
    if not rows:
        raise ValueError("training metrics are empty")
    return rows


def git_identity(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(
            run("status", "--short", "--untracked-files=no")
        ),
    }


def validate_git(identity: dict[str, Any], expected_revision: str) -> None:
    if identity["revision"] != expected_revision:
        raise ValueError("remote Git revision changed during formal training")
    if identity["branch"] != "scale/generative-system":
        raise ValueError("formal training branch changed")
    if identity["tracked_dirty"]:
        raise ValueError("formal training tracked worktree became dirty")


def run_progress_audit(
    *,
    project_root: Path,
    run_dir: Path,
    output: Path,
    milestone: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(project_root / "scripts" / "audit_generation_training_progress.py"),
        "--run-dir",
        str(run_dir),
        "--expected-steps",
        "50000",
        "--checkpoint-interval",
        "5000",
        "--evaluation-interval",
        "1000",
        "--integrity-policy",
        "required",
        "--output",
        str(output),
    ]
    if milestone >= 5000:
        command.extend(["--required-checkpoint-steps", "5000"])
    subprocess.run(command, cwd=project_root, check=True)
    with output.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_audit(report: dict[str, Any], milestone: int) -> None:
    if report.get("status") not in {"healthy", "complete"}:
        raise ValueError(f"progress audit is not healthy: {report.get('status')}")
    if report.get("issues") or report.get("warnings"):
        raise ValueError("progress audit contains issues or warnings")
    if int(report.get("last_step", -1)) < milestone:
        raise ValueError("progress audit did not reach the requested milestone")
    validation = report.get("validation", {})
    expected_events = milestone // 1000
    if int(validation.get("event_count", -1)) < expected_events:
        raise ValueError("scheduled validation evidence is incomplete")
    if validation.get("logging_complete") is not True:
        raise ValueError("scheduled validation logging is incomplete")
    if milestone >= 5000:
        checkpoint = report.get("checkpoint", {})
        integrity = checkpoint.get("latest_integrity", {})
        if 5000 not in checkpoint.get("steps", []):
            raise ValueError("step-5000 checkpoint is unavailable")
        if checkpoint.get("missing_required_steps"):
            raise ValueError("required checkpoint is missing")
        if integrity.get("status") != "verified":
            raise ValueError("step-5000 checkpoint integrity is not verified")
        if int(integrity.get("step", -1)) != 5000:
            raise ValueError("verified checkpoint step is not 5000")


def milestone_entry(
    report: dict[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "status": "pass",
        "report": report_path.as_posix(),
        "report_bytes": report_path.stat().st_size,
        "audited_at": utc_now(),
        "last_step": int(report["last_step"]),
        "validation_event_count": int(report["validation"]["event_count"]),
        "checkpoint_integrity": report["checkpoint"]["latest_integrity"][
            "status"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for and audit fixed-basis v3 training milestones."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=18000)
    parser.add_argument("--status-output", type=Path, required=True)
    args = parser.parse_args()

    if args.poll_seconds < 10 or args.timeout_seconds < args.poll_seconds:
        raise ValueError("invalid waiter timing")
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    report_root = args.report_root.resolve()
    status_output = args.status_output.resolve()
    metrics_path = run_dir / "train_metrics.jsonl"
    started_at = utc_now()
    started_monotonic = time.monotonic()
    completed: dict[str, dict[str, Any]] = {}

    for milestone in MILESTONES:
        report_path = report_root / f"cofitok_progress_step_{milestone:08d}.json"
        if not report_path.is_file():
            continue
        try:
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            validate_audit(report, milestone)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        completed[str(milestone)] = milestone_entry(
            report,
            report_path=report_path,
        )

    def publish(status: str, detail: str, last_step: int) -> None:
        write_json_atomic(
            status_output,
            {
                "schema_version": 1,
                "observer": "fixed_basis_v3_milestone_waiter",
                "status": status,
                "detail": detail,
                "pid": os.getpid(),
                "started_at": started_at,
                "updated_at": utc_now(),
                "expected_revision": args.expected_revision,
                "run_dir": run_dir.as_posix(),
                "last_step": last_step,
                "milestones": {
                    str(step): completed.get(str(step))
                    for step in MILESTONES
                },
            },
        )

    last_step = 0
    try:
        publish("waiting", "waiting for training metrics", last_step)
        while len(completed) < len(MILESTONES):
            if time.monotonic() - started_monotonic > args.timeout_seconds:
                raise TimeoutError("milestone waiter exceeded its bounded timeout")
            identity = git_identity(project_root)
            validate_git(identity, args.expected_revision)
            if metrics_path.is_file():
                rows = read_metrics(metrics_path)
                last_step = int(rows[-1]["step"])
                metric_age = time.time() - metrics_path.stat().st_mtime
                if metric_age > 1800:
                    raise TimeoutError("training metrics have been stale for 1800 seconds")
                for milestone in MILESTONES:
                    key = str(milestone)
                    if key in completed or last_step < milestone:
                        continue
                    report_path = (
                        report_root
                        / f"cofitok_progress_step_{milestone:08d}.json"
                    )
                    try:
                        report = run_progress_audit(
                            project_root=project_root,
                            run_dir=run_dir,
                            output=report_path,
                            milestone=milestone,
                        )
                        validate_audit(report, milestone)
                    except (subprocess.CalledProcessError, ValueError):
                        # Checkpoint publication and the corresponding log row are
                        # separate atomic operations. Retry inside the bounded window.
                        continue
                    completed[key] = milestone_entry(
                        report,
                        report_path=report_path,
                    )
                publish(
                    "pass" if len(completed) == len(MILESTONES) else "waiting",
                    (
                        "all requested milestones passed"
                        if len(completed) == len(MILESTONES)
                        else "waiting for the next milestone"
                    ),
                    last_step,
                )
            if len(completed) < len(MILESTONES):
                time.sleep(args.poll_seconds)
    except Exception as error:
        publish("failed", f"{type(error).__name__}: {error}", last_step)
        raise


if __name__ == "__main__":
    main()
