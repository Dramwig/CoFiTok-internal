from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from cofitok.reporting import write_json_report


CHECKPOINT_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt$")
REQUIRED_FINITE_FIELDS = (
    "total",
    "epsilon",
    "grad_norm",
    "learning_rate",
    "elapsed_seconds",
)


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


def audit_progress(
    run_dir: str | Path,
    *,
    expected_steps: int,
    checkpoint_interval: int,
    grad_clip_norm: float = 1.0,
    evaluation_interval: int | None = None,
    required_checkpoint_steps: list[int] | tuple[int, ...] = (),
) -> dict[str, Any]:
    if expected_steps < 1 or checkpoint_interval < 1 or grad_clip_norm <= 0.0:
        raise ValueError("expected_steps, checkpoint_interval, and grad_clip_norm must be positive")
    if evaluation_interval is not None and evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive when provided")
    if list(required_checkpoint_steps) != sorted(set(required_checkpoint_steps)):
        raise ValueError("required_checkpoint_steps must be sorted and unique")
    if any(step < 1 or step > expected_steps for step in required_checkpoint_steps):
        raise ValueError("required checkpoint step is outside the expected training range")
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

    training_report = None
    training_report_path = root / "training_report.json"
    if training_report_path.is_file():
        with training_report_path.open("r", encoding="utf-8") as handle:
            training_report = json.load(handle)
        if int(training_report.get("completed_steps", -1)) != last_step:
            issues.append("training report completed_steps does not match metrics")

    recent = rows[-min(len(rows), 10) :]
    validation_event_count = sum("validation_epsilon_mse" in row for row in rows)
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
        "schema_version": 1,
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
        },
        "checkpoint": {
            "interval": checkpoint_interval,
            "status": checkpoint_status,
            "steps": checkpoint_steps,
            "required_steps": list(required_checkpoint_steps),
            "missing_required_steps": missing_required_checkpoints,
            "latest": latest,
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
    parser.add_argument(
        "--required-checkpoint-steps",
        default="",
        help="Comma-separated checkpoint steps that must remain available once reached.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-invalid", action="store_true")
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
    )
    write_json_report(args.output, report)
    print(args.output)
    if report["status"] == "invalid" and not args.allow_invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
