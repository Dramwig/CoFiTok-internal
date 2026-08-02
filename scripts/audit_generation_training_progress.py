from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from cofitok.configs import load_config
from cofitok.training import consistency_weight_scale
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


CHECKPOINT_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt$")
REQUIRED_FINITE_FIELDS = (
    "total",
    "epsilon",
    "grad_norm",
    "learning_rate",
    "elapsed_seconds",
)
INTEGRITY_POLICIES = ("legacy_compute", "required")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training metrics do not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid metrics JSON at line {line_number}") from error
            rows.append(row)
    if not rows:
        raise ValueError("Training metrics are empty")
    return rows


def _checkpoint_steps(run_dir: Path) -> list[int]:
    steps = []
    for path in run_dir.glob("checkpoint_step_*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def _latest_pointer(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "latest.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _audit_latest_checkpoint_integrity(
    run_dir: Path,
    checkpoint_steps: list[int],
    latest: dict[str, Any] | None,
    policy: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    if policy not in INTEGRITY_POLICIES:
        raise ValueError(f"integrity_policy must be one of {INTEGRITY_POLICIES}")
    if not checkpoint_steps:
        return ({"policy": policy, "status": "not_available"}, [], [])

    step = checkpoint_steps[-1]
    checkpoint = run_dir / f"checkpoint_step_{step:08d}.pt"
    integrity_path = checkpoint_integrity_path(checkpoint)
    base = {
        "policy": policy,
        "checkpoint": checkpoint.name,
        "step": step,
        "integrity_manifest": integrity_path.name if integrity_path.is_file() else None,
    }
    if not integrity_path.is_file():
        if policy == "required":
            issue = f"latest checkpoint integrity manifest is missing: {integrity_path.name}"
            return ({**base, "status": "missing_manifest"}, [issue], [])
        report = {
            **base,
            "status": "legacy_computed",
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint),
        }
        warning = (
            "latest legacy checkpoint has no integrity manifest; size and SHA256 were "
            "computed read-only and must be bound by post-training migration"
        )
        return report, [], [warning]

    try:
        integrity = verify_training_checkpoint(checkpoint)
        if int(integrity.get("step", -1)) != step:
            raise ValueError("integrity manifest step does not match checkpoint filename")
        if latest is None:
            raise ValueError("latest.json is unavailable for integrity binding")
        if latest.get("integrity_manifest") != integrity_path.name:
            raise ValueError("latest.json does not bind the newest integrity manifest")
        if latest.get("checkpoint_sha256") != integrity.get("checkpoint_sha256"):
            raise ValueError("latest.json SHA256 does not match the integrity manifest")
        if int(latest.get("checkpoint_bytes", -1)) != int(integrity["checkpoint_bytes"]):
            raise ValueError("latest.json byte count does not match the integrity manifest")
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return (
            {**base, "status": "invalid", "error": str(error)},
            [f"latest checkpoint integrity verification failed: {error}"],
            [],
        )
    return (
        {
            **base,
            "status": "verified",
            "checkpoint_bytes": int(integrity["checkpoint_bytes"]),
            "checkpoint_sha256": integrity["checkpoint_sha256"],
            "checkpoint_format_version": int(integrity["checkpoint_format_version"]),
        },
        [],
        [],
    )


def _audit_consistency_schedules(
    rows: list[dict[str, Any]],
    *,
    config_path: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    if config_path is None:
        return {"status": "not_requested"}, []
    config = load_config(config_path)
    schedules = {
        "rollout_consistency": {
            "scale_field": "rollout_consistency_scale",
            "loss_field": "rollout_consistency",
            "weight": config.loss.rollout_consistency_weight,
            "start_step": config.loss.rollout_consistency_start_step,
            "warmup_steps": config.loss.rollout_consistency_warmup_steps,
        },
        "ema_teacher_consistency": {
            "scale_field": "ema_teacher_consistency_scale",
            "loss_field": "ema_teacher_consistency",
            "weight": config.loss.ema_teacher_consistency_weight,
            "start_step": config.loss.ema_teacher_consistency_start_step,
            "warmup_steps": config.loss.ema_teacher_consistency_warmup_steps,
        },
    }
    issues: list[str] = []
    evidence: dict[str, Any] = {}
    for name, schedule in schedules.items():
        scale_field = str(schedule["scale_field"])
        loss_field = str(schedule["loss_field"])
        weight = float(schedule["weight"])
        verified_rows = 0
        active_rows = 0
        nonzero_loss_rows = 0
        for index, row in enumerate(rows):
            step = int(row.get("step", -1))
            scheduled_scale = consistency_weight_scale(
                step,
                start_step=int(schedule["start_step"]),
                warmup_steps=int(schedule["warmup_steps"]),
            )
            expected_scale = scheduled_scale if weight > 0.0 else 0.0
            actual_scale = row.get(scale_field)
            try:
                actual_scale_value = float(actual_scale)
            except (TypeError, ValueError):
                issues.append(f"row {index} has invalid {scale_field}")
                continue
            if not math.isfinite(actual_scale_value):
                issues.append(f"row {index} has invalid {scale_field}")
                continue
            if not math.isclose(
                actual_scale_value, expected_scale, rel_tol=0.0, abs_tol=1e-6
            ):
                issues.append(f"row {index} {scale_field} differs from config schedule")
            loss = row.get(loss_field)
            try:
                loss_value = float(loss)
            except (TypeError, ValueError):
                issues.append(f"row {index} has invalid {loss_field}")
                continue
            if not math.isfinite(loss_value):
                issues.append(f"row {index} has invalid {loss_field}")
                continue
            if expected_scale == 0.0 and not math.isclose(
                loss_value, 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                issues.append(f"row {index} {loss_field} is nonzero while disabled")
            if expected_scale > 0.0:
                active_rows += 1
                if not math.isclose(loss_value, 0.0, rel_tol=0.0, abs_tol=1e-12):
                    nonzero_loss_rows += 1
            verified_rows += 1
        if active_rows > 0 and nonzero_loss_rows == 0:
            issues.append(f"{loss_field} is zero for all active schedule rows")
        last_step = int(rows[-1]["step"])
        scheduled_scale_at_last_step = consistency_weight_scale(
            last_step,
            start_step=int(schedule["start_step"]),
            warmup_steps=int(schedule["warmup_steps"]),
        )
        evidence[name] = {
            **schedule,
            "scheduled_scale_at_last_step": scheduled_scale_at_last_step,
            "expected_scale_at_last_step": (
                scheduled_scale_at_last_step if weight > 0.0 else 0.0
            ),
            "verified_rows": verified_rows,
            "active_rows": active_rows,
            "nonzero_loss_rows": nonzero_loss_rows,
        }
    return {
        "status": "invalid" if issues else "verified",
        "config_path": config_path.resolve().as_posix(),
        "config_sha256": file_sha256(config_path),
        "schedules": evidence,
    }, issues


def audit_progress(
    run_dir: str | Path,
    *,
    expected_steps: int,
    checkpoint_interval: int,
    grad_clip_norm: float = 1.0,
    evaluation_interval: int | None = None,
    required_checkpoint_steps: list[int] | tuple[int, ...] = (),
    integrity_policy: str = "legacy_compute",
    config_path: str | Path | None = None,
    allow_stale_incomplete_training_report: bool = False,
) -> dict[str, Any]:
    if expected_steps < 1 or checkpoint_interval < 1 or grad_clip_norm <= 0.0:
        raise ValueError("expected_steps, checkpoint_interval, and grad_clip_norm must be positive")
    if evaluation_interval is not None and evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive when provided")
    if list(required_checkpoint_steps) != sorted(set(required_checkpoint_steps)):
        raise ValueError("required_checkpoint_steps must be sorted and unique")
    if any(step < 1 or step > expected_steps for step in required_checkpoint_steps):
        raise ValueError("required checkpoint step is outside the expected training range")
    if integrity_policy not in INTEGRITY_POLICIES:
        raise ValueError(f"integrity_policy must be one of {INTEGRITY_POLICIES}")
    root = Path(run_dir)
    rows = _read_metrics(root / "train_metrics.jsonl")
    issues = []
    warnings = []
    steps = [int(row.get("step", -1)) for row in rows]
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        issues.append("metrics steps are not strictly increasing")
    for index, row in enumerate(rows):
        for field in REQUIRED_FINITE_FIELDS:
            value = row.get(field)
            if value is None or not math.isfinite(float(value)):
                issues.append(f"row {index} has non-finite {field}")
        if float(row.get("grad_norm", math.inf)) < 0.0:
            issues.append(f"row {index} has negative grad_norm")
    schedule_audit, schedule_issues = _audit_consistency_schedules(
        rows, config_path=Path(config_path) if config_path is not None else None
    )
    issues.extend(schedule_issues)
    last = rows[-1]
    last_step = int(last["step"])
    if last_step > expected_steps:
        issues.append("last step exceeds expected steps")

    segment_start = 0
    for index in range(1, len(rows)):
        if float(rows[index]["elapsed_seconds"]) <= float(rows[index - 1]["elapsed_seconds"]):
            segment_start = index
    segment = rows[segment_start:]
    seconds_per_step = None
    if len(segment) >= 2:
        delta_steps = int(segment[-1]["step"]) - int(segment[0]["step"])
        delta_seconds = float(segment[-1]["elapsed_seconds"]) - float(
            segment[0]["elapsed_seconds"]
        )
        if delta_steps > 0 and delta_seconds > 0.0:
            seconds_per_step = delta_seconds / delta_steps
    eta_seconds = (
        max(expected_steps - last_step, 0) * seconds_per_step
        if seconds_per_step is not None
        else None
    )

    checkpoint_steps = _checkpoint_steps(root)
    missing_required_checkpoints = [
        step
        for step in required_checkpoint_steps
        if step <= last_step and step not in checkpoint_steps
    ]
    if missing_required_checkpoints:
        issues.append(
            "required checkpoints are missing: "
            + ", ".join(str(step) for step in missing_required_checkpoints)
        )
    latest = _latest_pointer(root)
    due_step = (last_step // checkpoint_interval) * checkpoint_interval
    checkpoint_status = "not_due"
    if due_step > 0:
        if checkpoint_steps and checkpoint_steps[-1] >= due_step:
            checkpoint_status = "available"
        elif last_step == due_step:
            checkpoint_status = "due_grace"
        else:
            checkpoint_status = "overdue"
            issues.append(f"checkpoint for step {due_step} is overdue")
    if checkpoint_steps and checkpoint_steps[-1] > last_step:
        issues.append("checkpoint step exceeds the latest logged step")
    if checkpoint_steps:
        if latest is None:
            issues.append("checkpoints exist without latest.json")
        else:
            expected_name = f"checkpoint_step_{checkpoint_steps[-1]:08d}.pt"
            if latest.get("checkpoint") != expected_name:
                issues.append("latest.json does not point to the newest checkpoint")
            if int(latest.get("step", -1)) != checkpoint_steps[-1]:
                issues.append("latest.json step does not match the newest checkpoint")
    elif latest is not None:
        issues.append("latest.json exists without a checkpoint file")
    latest_integrity, integrity_issues, integrity_warnings = (
        _audit_latest_checkpoint_integrity(
            root,
            checkpoint_steps,
            latest,
            integrity_policy,
        )
    )
    issues.extend(integrity_issues)
    warnings.extend(integrity_warnings)

    training_report = None
    training_report_status = "absent"
    training_report_path = root / "training_report.json"
    if training_report_path.is_file():
        with training_report_path.open("r", encoding="utf-8") as handle:
            training_report = json.load(handle)
        report_step = int(training_report.get("completed_steps", -1))
        stale_incomplete_report = (
            allow_stale_incomplete_training_report
            and report_step < last_step
            and training_report.get("training_complete") is False
            and training_report.get("stop_requested") is True
            and int(training_report.get("target_steps", -1)) == expected_steps
            and int(
                (training_report.get("latest_checkpoint") or {}).get("step", -1)
            )
            == report_step
        )
        if report_step == last_step:
            training_report_status = "current"
        elif stale_incomplete_report:
            training_report_status = "stale_incomplete_resume_report"
        else:
            training_report_status = "mismatched"
            issues.append("training report completed_steps does not match metrics")

    recent = rows[-min(len(rows), 10) :]
    validation_rows = [row for row in rows if "validation_epsilon_mse" in row]
    validation_event_count = len(validation_rows)
    validation_metadata_fields = (
        "validation_event_index",
        "validation_batch_index",
        "validation_num_images",
        "validation_noise_seed",
    )
    validation_metadata_counts = [
        sum(field in row for field in validation_metadata_fields)
        for row in validation_rows
    ]
    if not validation_rows:
        validation_metadata_status = "not_applicable"
    elif all(count == 0 for count in validation_metadata_counts):
        validation_metadata_status = "legacy_absent"
    elif all(count == len(validation_metadata_fields) for count in validation_metadata_counts):
        validation_metadata_status = "complete"
    else:
        validation_metadata_status = "partial"
        warnings.append("validation provenance metadata is only partially logged")
    if validation_metadata_status == "complete":
        event_indices = [
            int(row["validation_event_index"])
            for row in validation_rows
        ]
        if event_indices != list(range(validation_event_count)):
            issues.append("validation event indices are not contiguous from zero")
        if any(int(row["validation_batch_index"]) < 0 for row in validation_rows):
            issues.append("validation batch indices must be non-negative")
        if any(int(row["validation_num_images"]) <= 0 for row in validation_rows):
            issues.append("validation image counts must be positive")
        if any(int(row["validation_noise_seed"]) < 0 for row in validation_rows):
            issues.append("validation noise seeds must be non-negative")
    expected_validation_events = (
        last_step // evaluation_interval if evaluation_interval is not None else None
    )
    if (
        expected_validation_events is not None
        and validation_event_count < expected_validation_events
    ):
        warnings.append(
            "validation metrics are missing from train_metrics.jsonl; legacy revisions "
            "may evaluate after emitting the log row"
        )
    complete = (
        training_report is not None
        and training_report.get("training_complete") is True
        and last_step == expected_steps
    )
    return {
        "schema_version": 2,
        "status": "invalid" if issues else ("complete" if complete else "healthy"),
        "run_dir": root.resolve().as_posix(),
        "expected_steps": expected_steps,
        "last_step": last_step,
        "progress_fraction": last_step / expected_steps,
        "metric_row_count": len(rows),
        "current_segment_start_step": int(segment[0]["step"]),
        "seconds_per_step": seconds_per_step,
        "eta_seconds": eta_seconds,
        "latest_metrics": last,
        "recent_mean": {
            "total": mean(float(row["total"]) for row in recent),
            "epsilon": mean(float(row["epsilon"]) for row in recent),
            "grad_norm": mean(float(row["grad_norm"]) for row in recent),
        },
        "gradient_clipping": {
            "configured_max_norm": grad_clip_norm,
            "logged_norm_semantics": "pre_clip_total_norm",
            "observed_max_pre_clip_norm": max(float(row["grad_norm"]) for row in rows),
            "pre_clip_exceedance_count": sum(
                float(row["grad_norm"]) > grad_clip_norm for row in rows
            ),
        },
        "validation_event_count": validation_event_count,
        "validation": {
            "configured_interval": evaluation_interval,
            "event_count": validation_event_count,
            "expected_event_count": expected_validation_events,
            "logging_complete": (
                expected_validation_events is None
                or validation_event_count >= expected_validation_events
            ),
            "provenance_metadata_status": validation_metadata_status,
            "provenance_fields": list(validation_metadata_fields),
        },
        "consistency_schedules": schedule_audit,
        "training_report": {
            "path": (
                training_report_path.resolve().as_posix()
                if training_report is not None
                else None
            ),
            "status": training_report_status,
            "completed_steps": (
                int(training_report.get("completed_steps", -1))
                if training_report is not None
                else None
            ),
            "training_complete": (
                training_report.get("training_complete")
                if training_report is not None
                else None
            ),
        },
        "checkpoint": {
            "interval": checkpoint_interval,
            "status": checkpoint_status,
            "steps": checkpoint_steps,
            "required_steps": list(required_checkpoint_steps),
            "missing_required_steps": missing_required_checkpoints,
            "latest": latest,
            "latest_integrity": latest_integrity,
        },
        "issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a live generation training run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--config")
    parser.add_argument(
        "--required-checkpoint-steps",
        default="",
        help="Comma-separated checkpoint steps that must remain available once reached.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--integrity-policy",
        choices=INTEGRITY_POLICIES,
        default="legacy_compute",
        help=(
            "legacy_compute hashes a sidecar-free legacy checkpoint read-only; required "
            "fails unless the newest checkpoint, sidecar, and latest.json binding verify"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument(
        "--allow-stale-incomplete-training-report",
        action="store_true",
        help=(
            "accept a prior stop-requested incomplete report that is behind resumed "
            "canonical metrics; all other report/metrics mismatches remain invalid"
        ),
    )
    args = parser.parse_args()
    required_checkpoint_steps = [
        int(value.strip())
        for value in args.required_checkpoint_steps.split(",")
        if value.strip()
    ]

    report = audit_progress(
        args.run_dir,
        expected_steps=args.expected_steps,
        checkpoint_interval=args.checkpoint_interval,
        grad_clip_norm=args.grad_clip_norm,
        evaluation_interval=args.evaluation_interval,
        required_checkpoint_steps=required_checkpoint_steps,
        integrity_policy=args.integrity_policy,
        config_path=args.config,
        allow_stale_incomplete_training_report=(
            args.allow_stale_incomplete_training_report
        ),
    )
    write_json_report(args.output, report)
    print(args.output)
    if report["status"] == "invalid" and not args.allow_invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
