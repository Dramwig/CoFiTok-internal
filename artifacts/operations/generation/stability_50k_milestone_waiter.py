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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def git_identity(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--short", "--untracked-files=no")),
    }


def validate_project_identity(
    identity: dict[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> None:
    if identity.get("revision") != expected_revision:
        raise ValueError("audit project revision changed")
    if identity.get("branch") != expected_branch:
        raise ValueError("audit project branch changed")
    if identity.get("tracked_dirty") is not False:
        raise ValueError("audit project tracked worktree is dirty")


def read_last_metric(path: Path) -> tuple[int, float]:
    if not path.is_file():
        return 0, float("inf")
    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        return 0, time.time() - path.stat().st_mtime
    row = json.loads(last_line)
    return int(row["step"]), time.time() - path.stat().st_mtime


def audit_command(
    *,
    project_root: Path,
    run_dir: Path,
    config: Path,
    output: Path,
    expected_steps: int,
    checkpoint_interval: int,
    evaluation_interval: int,
    required_checkpoint_steps: tuple[int, ...],
) -> list[str]:
    return [
        sys.executable,
        str(project_root / "scripts" / "audit_generation_training_progress.py"),
        "--run-dir",
        str(run_dir),
        "--expected-steps",
        str(expected_steps),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--evaluation-interval",
        str(evaluation_interval),
        "--config",
        str(config),
        "--required-checkpoint-steps",
        ",".join(str(step) for step in required_checkpoint_steps),
        "--integrity-policy",
        "required",
        "--output",
        str(output),
    ]


def run_progress_audit(
    *,
    command: list[str],
    project_root: Path,
) -> None:
    environment = os.environ.copy()
    project_src = str(project_root / "src")
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((project_src, inherited_pythonpath))
        if inherited_pythonpath
        else project_src
    )
    subprocess.run(command, cwd=project_root, env=environment, check=True)


def validate_progress_report(
    report: dict[str, Any],
    *,
    milestone_step: int,
    evaluation_interval: int,
    required_checkpoint_steps: tuple[int, ...],
    expected_config_sha256: str,
    expected_training_revision: str,
    expected_training_branch: str,
) -> None:
    if report.get("status") not in {"healthy", "complete"}:
        raise ValueError("progress audit status is not healthy")
    if report.get("issues") or report.get("warnings"):
        raise ValueError("progress audit contains issues or warnings")
    if int(report.get("last_step", -1)) < milestone_step:
        raise ValueError("progress audit has not reached the milestone")

    validation = report.get("validation", {})
    if int(validation.get("event_count", -1)) < milestone_step // evaluation_interval:
        raise ValueError("scheduled validation evidence is incomplete")
    if validation.get("logging_complete") is not True:
        raise ValueError("scheduled validation logging is incomplete")
    if validation.get("provenance_metadata_status") != "complete":
        raise ValueError("scheduled validation provenance is incomplete")

    checkpoint = report.get("checkpoint", {})
    available_steps = tuple(int(step) for step in checkpoint.get("steps", []))
    if any(step not in available_steps for step in required_checkpoint_steps):
        raise ValueError("a required recovery checkpoint is unavailable")
    if checkpoint.get("missing_required_steps"):
        raise ValueError("the progress auditor reports missing recovery checkpoints")
    integrity = checkpoint.get("latest_integrity", {})
    if integrity.get("status") != "verified":
        raise ValueError("milestone checkpoint integrity is not verified")
    if int(integrity.get("step", -1)) != milestone_step:
        raise ValueError("the verified latest checkpoint is not the milestone")
    latest = checkpoint.get("latest", {})
    if int(latest.get("step", -1)) != milestone_step:
        raise ValueError("latest.json is not bound to the milestone checkpoint")
    if latest.get("git_revision") != expected_training_revision:
        raise ValueError("milestone checkpoint training revision mismatch")
    if latest.get("git_branch") != expected_training_branch:
        raise ValueError("milestone checkpoint training branch mismatch")
    if latest.get("git_dirty") is not False:
        raise ValueError("milestone checkpoint reports dirty training code")

    schedules = report.get("consistency_schedules", {})
    if schedules.get("status") != "verified":
        raise ValueError("consistency schedules are not verified")
    if schedules.get("config_sha256") != expected_config_sha256:
        raise ValueError("audited config SHA256 mismatch")
    metric_row_count = int(report.get("metric_row_count", -1))
    for name in ("rollout_consistency", "ema_teacher_consistency"):
        schedule = schedules.get("schedules", {}).get(name, {})
        if int(schedule.get("verified_rows", -1)) != metric_row_count:
            raise ValueError(f"{name} did not verify every metric row")
        active_rows = int(schedule.get("active_rows", -1))
        if active_rows < 1:
            raise ValueError(f"{name} has no active rows at the milestone")
        if int(schedule.get("nonzero_loss_rows", -1)) != active_rows:
            raise ValueError(f"{name} has inactive loss rows after activation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for and independently audit one stability-training milestone."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--expected-project-revision", required=True)
    parser.add_argument("--expected-project-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--milestone-step", type=int, required=True)
    parser.add_argument("--required-checkpoint-steps", type=int, nargs="+", required=True)
    parser.add_argument("--expected-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--evaluation-interval", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=18000)
    parser.add_argument("--metrics-stale-seconds", type=int, default=1800)
    parser.add_argument("--observer-name", default="stability_50k_milestone_waiter")
    args = parser.parse_args()

    if args.poll_seconds < 10 or args.timeout_seconds < args.poll_seconds:
        raise ValueError("invalid waiter timing")
    if args.metrics_stale_seconds < args.poll_seconds:
        raise ValueError("metrics stale threshold must cover at least one poll")
    required_steps = tuple(args.required_checkpoint_steps)
    if required_steps != tuple(sorted(set(required_steps))):
        raise ValueError("required checkpoint steps must be sorted and unique")
    if not required_steps or required_steps[-1] != args.milestone_step:
        raise ValueError("the milestone must be the newest required checkpoint")
    if any(step < 1 or step > args.milestone_step for step in required_steps):
        raise ValueError("required steps must be within the reached milestone")
    if args.milestone_step % args.checkpoint_interval:
        raise ValueError("the milestone must align with checkpoint publication")
    if args.milestone_step > args.expected_steps:
        raise ValueError("milestone exceeds the expected training horizon")

    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    config = args.config.resolve()
    report_output = args.report_output.resolve()
    status_output = args.status_output.resolve()
    metrics = run_dir / "train_metrics.jsonl"
    checkpoint = run_dir / f"checkpoint_step_{args.milestone_step:08d}.pt"
    sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    latest_pointer = run_dir / "latest.json"
    started_at = utc_now()
    started_monotonic = time.monotonic()
    audit_attempts = 0
    last_step = 0
    metric_age_seconds: float | None = None
    command = audit_command(
        project_root=project_root,
        run_dir=run_dir,
        config=config,
        output=report_output,
        expected_steps=args.expected_steps,
        checkpoint_interval=args.checkpoint_interval,
        evaluation_interval=args.evaluation_interval,
        required_checkpoint_steps=required_steps,
    )

    def publish(status: str, detail: str, **extra: Any) -> None:
        write_json_atomic(
            status_output,
            {
                "schema_version": 1,
                "observer": args.observer_name,
                "status": status,
                "detail": detail,
                "pid": os.getpid(),
                "started_at": started_at,
                "updated_at": utc_now(),
                "last_step": last_step,
                "metric_age_seconds": metric_age_seconds,
                "audit_attempts": audit_attempts,
                "project_root": project_root.as_posix(),
                "run_dir": run_dir.as_posix(),
                "config": config.as_posix(),
                "report_output": report_output.as_posix(),
                "milestone_step": args.milestone_step,
                "required_checkpoint_steps": list(required_steps),
                "expected_project_revision": args.expected_project_revision,
                "expected_project_branch": args.expected_project_branch,
                "expected_training_revision": args.expected_training_revision,
                "expected_training_branch": args.expected_training_branch,
                "expected_config_sha256": args.expected_config_sha256,
                "audit_command": command,
                **extra,
            },
        )

    try:
        if file_sha256(config) != args.expected_config_sha256:
            raise ValueError("config SHA256 does not match the pinned expectation")
        publish("waiting", "waiting for milestone metrics")
        while True:
            if time.monotonic() - started_monotonic > args.timeout_seconds:
                raise TimeoutError("milestone waiter exceeded its bounded timeout")
            identity = git_identity(project_root)
            validate_project_identity(
                identity,
                expected_revision=args.expected_project_revision,
                expected_branch=args.expected_project_branch,
            )
            last_step, metric_age_seconds = read_last_metric(metrics)
            if metric_age_seconds > args.metrics_stale_seconds:
                raise TimeoutError("training metrics exceeded the stale threshold")
            if last_step < args.milestone_step:
                publish("waiting", "waiting for milestone metrics")
                time.sleep(args.poll_seconds)
                continue
            if not all(path.is_file() for path in (checkpoint, sidecar, latest_pointer)):
                publish("waiting", "waiting for atomic checkpoint publication")
                time.sleep(args.poll_seconds)
                continue
            latest = json.loads(latest_pointer.read_text(encoding="utf-8"))
            if int(latest.get("step", -1)) < args.milestone_step:
                publish("waiting", "waiting for latest.json milestone binding")
                time.sleep(args.poll_seconds)
                continue
            if int(latest.get("step", -1)) != args.milestone_step:
                raise ValueError("latest.json advanced past the unaudited milestone")

            audit_attempts += 1
            publish("auditing", "running tracked progress auditor")
            run_progress_audit(command=command, project_root=project_root)
            report = json.loads(report_output.read_text(encoding="utf-8"))
            validate_progress_report(
                report,
                milestone_step=args.milestone_step,
                evaluation_interval=args.evaluation_interval,
                required_checkpoint_steps=required_steps,
                expected_config_sha256=args.expected_config_sha256,
                expected_training_revision=args.expected_training_revision,
                expected_training_branch=args.expected_training_branch,
            )
            publish(
                "pass",
                "milestone progress and checkpoint audit passed",
                report_bytes=report_output.stat().st_size,
                report_sha256=file_sha256(report_output),
                checkpoint_bytes=checkpoint.stat().st_size,
                checkpoint_sha256=report["checkpoint"]["latest_integrity"][
                    "checkpoint_sha256"
                ],
            )
            return
    except Exception as error:
        publish("failed", f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
