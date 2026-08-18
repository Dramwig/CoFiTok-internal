from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows test import path
    fcntl = None

from cofitok.configs import load_config
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.rollout import consistency_weight_scale

REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_consistency_schedule_transition_audit"
WAITER_ROLE = "generation_consistency_schedule_transition_waiter"
SCOPE = {
    "read_only_metrics_verification": True,
    "gpu_required": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON source is not an object: {path}")
    return payload


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
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
                raise TypeError(f"metrics row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("training metrics are empty")
    return rows


def git_value(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_checkout(
    checkout: Path,
    *,
    expected_revision: str,
    expected_branch: str,
    expected_tree: str,
) -> dict[str, Any]:
    revision = git_value(checkout, "rev-parse", "HEAD")
    branch = git_value(checkout, "branch", "--show-current")
    tree = git_value(checkout, "show", "-s", "--format=%T", "HEAD")
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
        "path": checkout.resolve().as_posix(),
        "revision": revision,
        "branch": branch,
        "tree": tree,
        "tracked_dirty": False,
    }


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def validate_run_manifest(
    path: Path,
    *,
    expected_revision: str,
    expected_branch: str,
    expected_start_step: int,
    expected_warmup_steps: int,
    expected_weight: float,
    expected_target_steps: int,
    effective_batch: int,
) -> dict[str, Any]:
    manifest = read_object(path)
    git = manifest.get("git")
    config = manifest.get("config")
    loss = config.get("loss") if isinstance(config, dict) else None
    runtime = config.get("runtime") if isinstance(config, dict) else None
    data = config.get("data") if isinstance(config, dict) else None
    optimization = config.get("optimization") if isinstance(config, dict) else None
    if (
        not isinstance(git, dict)
        or git.get("revision") != expected_revision
        or git.get("branch") != expected_branch
        or git.get("dirty") is not False
        or not isinstance(loss, dict)
        or int(loss.get("ema_teacher_consistency_start_step", -1))
        != expected_start_step
        or int(loss.get("ema_teacher_consistency_warmup_steps", -1))
        != expected_warmup_steps
        or not math.isclose(
            float(loss.get("ema_teacher_consistency_weight", float("nan"))),
            expected_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isinstance(runtime, dict)
        or int(runtime.get("steps", -1)) != expected_target_steps
        or not isinstance(data, dict)
        or not isinstance(optimization, dict)
        or int(data.get("batch_size", -1))
        * int(optimization.get("gradient_accumulation_steps", -1))
        != effective_batch
    ):
        raise ValueError("run manifest consistency schedule contract differs")
    return {
        "source": file_identity(path),
        "git": git,
        "schedule": {
            "start_step": expected_start_step,
            "warmup_steps": expected_warmup_steps,
            "weight": expected_weight,
        },
        "target_steps": expected_target_steps,
        "micro_batch_size": int(data["batch_size"]),
        "gradient_accumulation_steps": int(optimization["gradient_accumulation_steps"]),
        "effective_batch": effective_batch,
    }


def validate_config_source(
    path: Path,
    *,
    expected_start_step: int,
    expected_warmup_steps: int,
    expected_weight: float,
    expected_target_steps: int,
    effective_batch: int,
    expected_log_interval: int,
) -> dict[str, Any]:
    config = load_config(path)
    if (
        config.loss.ema_teacher_consistency_start_step != expected_start_step
        or config.loss.ema_teacher_consistency_warmup_steps != expected_warmup_steps
        or not math.isclose(
            config.loss.ema_teacher_consistency_weight,
            expected_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or config.runtime.steps != expected_target_steps
        or config.data.batch_size * config.optimization.gradient_accumulation_steps
        != effective_batch
        or config.optimization.log_interval != expected_log_interval
    ):
        raise ValueError("config consistency schedule contract differs")
    return {
        "source": file_identity(path),
        "name": config.name,
        "schedule": {
            "start_step": expected_start_step,
            "warmup_steps": expected_warmup_steps,
            "weight": expected_weight,
        },
        "target_steps": expected_target_steps,
        "micro_batch_size": config.data.batch_size,
        "gradient_accumulation_steps": (
            config.optimization.gradient_accumulation_steps
        ),
        "effective_batch": effective_batch,
        "log_interval": expected_log_interval,
    }


def audit_transition_rows(
    rows: list[dict[str, Any]],
    *,
    start_step: int,
    warmup_steps: int,
    weight: float,
    target_step: int,
    first_active_step: int,
    minimum_active_rows: int,
    effective_batch: int,
) -> dict[str, Any]:
    if (
        start_step < 0
        or warmup_steps < 1
        or weight <= 0.0
        or target_step <= start_step
        or first_active_step <= start_step
        or first_active_step > target_step
        or minimum_active_rows < 1
        or effective_batch < 1
    ):
        raise ValueError("invalid transition audit parameters")
    steps = [int(row.get("step", -1)) for row in rows]
    if not all(left < right for left, right in pairwise(steps)):
        raise ValueError("training metrics are not strictly increasing")
    if steps[-1] < target_step:
        raise ValueError("training metrics have not reached the transition target")
    by_step = {step: row for step, row in zip(steps, rows)}
    for required_step in (start_step, first_active_step, target_step):
        if required_step not in by_step:
            raise ValueError(f"training metrics omit required step {required_step}")

    audited_rows = [row for row in rows if int(row["step"]) <= target_step]
    active_rows: list[dict[str, Any]] = []
    for row in audited_rows:
        step = int(row["step"])
        if int(row.get("samples_seen", -1)) != step * effective_batch:
            raise ValueError(f"samples_seen binding differs at step {step}")
        expected_scale = consistency_weight_scale(
            step,
            start_step=start_step,
            warmup_steps=warmup_steps,
        )
        actual_scale = _finite(
            row.get("ema_teacher_consistency_scale"),
            label=f"step {step} EMA-teacher scale",
        )
        loss = _finite(
            row.get("ema_teacher_consistency"),
            label=f"step {step} EMA-teacher loss",
        )
        for field in ("total", "epsilon", "grad_norm"):
            _finite(row.get(field), label=f"step {step} {field}")
        if not math.isclose(actual_scale, expected_scale, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"EMA-teacher scale differs at step {step}")
        if expected_scale == 0.0:
            if not math.isclose(loss, 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"EMA-teacher loss is nonzero before activation at step {step}"
                )
        else:
            if loss <= 0.0:
                raise ValueError(
                    f"EMA-teacher loss is not positive at active step {step}"
                )
            active_rows.append(row)

    selected_active = [
        row
        for row in active_rows
        if first_active_step <= int(row["step"]) <= target_step
    ]
    if len(selected_active) < minimum_active_rows:
        raise ValueError("too few active EMA-teacher rows at transition target")
    if int(selected_active[0]["step"]) != first_active_step:
        raise ValueError("first active EMA-teacher row differs")
    scales = [float(row["ema_teacher_consistency_scale"]) for row in selected_active]
    if not all(left < right for left, right in pairwise(scales)):
        raise ValueError("EMA-teacher warmup scales are not strictly increasing")

    start_row = by_step[start_step]
    target_row = by_step[target_step]
    expected_target_scale = consistency_weight_scale(
        target_step,
        start_step=start_step,
        warmup_steps=warmup_steps,
    )
    return {
        "status": "verified",
        "start_step": start_step,
        "first_active_step": first_active_step,
        "target_step": target_step,
        "warmup_steps": warmup_steps,
        "weight": weight,
        "expected_target_scale": expected_target_scale,
        "audited_row_count": len(audited_rows),
        "active_row_count": len(selected_active),
        "start_row": start_row,
        "first_active_row": selected_active[0],
        "target_row": target_row,
        "active_rows": selected_active,
        "active_scale_min": min(scales),
        "active_scale_max": max(scales),
        "active_loss_min": min(
            float(row["ema_teacher_consistency"]) for row in selected_active
        ),
        "active_loss_max": max(
            float(row["ema_teacher_consistency"]) for row in selected_active
        ),
        "samples_seen_binding": f"samples_seen == step * {effective_batch}",
        "samples_seen_binding_verified": True,
        "strictly_increasing_metrics": True,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    if file_sha256(source_path) != args.expected_source_sha256:
        raise ValueError("waiter source SHA256 differs")
    checkout = verify_checkout(
        args.training_checkout,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_tree=args.expected_tree,
    )
    config = validate_config_source(
        args.config,
        expected_start_step=args.expected_start_step,
        expected_warmup_steps=args.expected_warmup_steps,
        expected_weight=args.expected_weight,
        expected_target_steps=args.expected_training_steps,
        effective_batch=args.effective_batch,
        expected_log_interval=args.expected_log_interval,
    )
    run_manifest_path = args.run_dir / "run_manifest.json"
    run_manifest = validate_run_manifest(
        run_manifest_path,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_start_step=args.expected_start_step,
        expected_warmup_steps=args.expected_warmup_steps,
        expected_weight=args.expected_weight,
        expected_target_steps=args.expected_training_steps,
        effective_batch=args.effective_batch,
    )
    metrics_path = args.run_dir / "train_metrics.jsonl"
    rows = read_metrics(metrics_path)
    transition = audit_transition_rows(
        rows,
        start_step=args.expected_start_step,
        warmup_steps=args.expected_warmup_steps,
        weight=args.expected_weight,
        target_step=args.target_step,
        first_active_step=args.first_active_step,
        minimum_active_rows=args.minimum_active_rows,
        effective_batch=args.effective_batch,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass",
        "audited_at": utc_now(),
        "scope": SCOPE,
        "waiter_source": file_identity(source_path),
        "training_checkout": checkout,
        "run_dir": args.run_dir.resolve().as_posix(),
        "config": config,
        "run_manifest": run_manifest,
        "metrics": {
            "source": file_identity(metrics_path),
            "row_count_at_read": len(rows),
            "last_step_at_read": int(rows[-1]["step"]),
        },
        "transition": transition,
    }


def target_is_ready(metrics_path: Path, target_step: int) -> tuple[bool, str]:
    if not metrics_path.is_file():
        return False, "metrics_missing"
    try:
        lines = [
            line
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not lines:
            return False, "metrics_empty"
        last_step = int(json.loads(lines[-1]).get("step", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, "metrics_tail_unreadable"
    if last_step < target_step:
        return False, "transition_target_not_reached"
    return True, "ready"


def status_payload(
    args: argparse.Namespace,
    *,
    status: str,
    detail: str,
    started_at: str,
    report_identity: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": WAITER_ROLE,
        "status": status,
        "detail": detail,
        "started_at": started_at,
        "updated_at": utc_now(),
        "pid": os.getpid(),
        "scope": SCOPE,
        "run_dir": args.run_dir.resolve().as_posix(),
        "target_step": args.target_step,
        "first_active_step": args.first_active_step,
        "minimum_active_rows": args.minimum_active_rows,
        "report_output": args.report_output.resolve().as_posix(),
    }
    if report_identity is not None:
        payload["report_identity"] = report_identity
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return payload


def run_waiter(args: argparse.Namespace) -> int:
    if (
        args.poll_seconds < 1
        or args.timeout_seconds < 1
        or args.target_step < 1
        or args.minimum_active_rows < 1
    ):
        raise ValueError("invalid waiter timing or target parameters")
    started_at = utc_now()
    deadline = time.monotonic() + args.timeout_seconds
    metrics_path = args.run_dir / "train_metrics.jsonl"
    while True:
        ready, detail = target_is_ready(metrics_path, args.target_step)
        if ready:
            try:
                report = build_audit(args)
                write_json_report(args.report_output, report)
                identity = file_identity(args.report_output)
                write_json_report(
                    args.status_output,
                    status_payload(
                        args,
                        status="pass",
                        detail="consistency_schedule_transition_verified",
                        started_at=started_at,
                        report_identity=identity,
                    ),
                )
                return 0
            except BaseException as error:
                write_json_report(
                    args.status_output,
                    status_payload(
                        args,
                        status="failed",
                        detail="consistency_schedule_transition_invalid",
                        started_at=started_at,
                        error=error,
                    ),
                )
                raise
        write_json_report(
            args.status_output,
            status_payload(
                args,
                status="waiting",
                detail=detail,
                started_at=started_at,
            ),
        )
        if args.once:
            return 0
        if time.monotonic() >= deadline:
            error = TimeoutError("consistency schedule transition waiter timed out")
            write_json_report(
                args.status_output,
                status_payload(
                    args,
                    status="failed",
                    detail="timeout",
                    started_at=started_at,
                    error=error,
                ),
            )
            raise error
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a generation consistency schedule transition and verify "
            "the source-bound warmup rows without using GPU or signaling training."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-checkout", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-start-step", type=int, required=True)
    parser.add_argument("--expected-warmup-steps", type=int, required=True)
    parser.add_argument("--expected-weight", type=float, required=True)
    parser.add_argument("--expected-training-steps", type=int, required=True)
    parser.add_argument("--expected-log-interval", type=int, required=True)
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--first-active-step", type=int, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--minimum-active-rows", type=int, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock_path = args.status_output.with_suffix(args.status_output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "consistency schedule waiter lock is held"
                ) from error
        return run_waiter(args)


if __name__ == "__main__":
    raise SystemExit(main())
