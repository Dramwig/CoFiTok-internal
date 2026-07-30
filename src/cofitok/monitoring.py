from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt$")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CHECKPOINT_INTEGRITY_POLICIES = ("optional", "required")
FINITE_METRICS = {
    "step",
    "total",
    "epsilon",
    "grad_norm",
    "learning_rate",
    "elapsed_seconds",
    "samples_seen",
    "ema_decay",
}
NONNEGATIVE_METRICS = FINITE_METRICS - {"ema_decay"}
METHOD_PROCESS_PATTERNS = {
    "cofitok": re.compile(r"imagenet256\S*cofitok", re.IGNORECASE),
    "dense_identity": re.compile(r"imagenet256\S*dense", re.IGNORECASE),
}


def _read_metrics(path: Path) -> tuple[dict[str, Any] | None, int, list[str]]:
    if not path.is_file():
        return None, 0, []
    last = None
    row_count = 0
    previous_step = 0
    issues = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append(f"metrics JSON is invalid at line {line_number}: {error.msg}")
                continue
            step = row.get("step")
            if not isinstance(step, int) or isinstance(step, bool) or step <= previous_step:
                issues.append(f"metrics step is not strictly increasing at line {line_number}")
            else:
                previous_step = step
            for key in FINITE_METRICS & row.keys():
                value = row[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    issues.append(f"metric {key} is non-numeric at line {line_number}")
                elif not math.isfinite(float(value)):
                    issues.append(f"metric {key} is non-finite at line {line_number}")
                elif key in NONNEGATIVE_METRICS and float(value) < 0.0:
                    issues.append(f"metric {key} is negative at line {line_number}")
            last = row
    return last, row_count, issues


def _read_report(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, []
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), []
    except json.JSONDecodeError as error:
        return None, [f"training report JSON is invalid: {error.msg}"]


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"{label} JSON is invalid: {error}"]
    if not isinstance(payload, dict):
        return None, [f"{label} must contain a JSON object"]
    return payload, []


def _inspect_checkpoint_integrity(
    checkpoint: Path,
    *,
    step: int,
    checkpoint_bytes: int,
    required: bool,
    expected_git_revision: str | None,
) -> tuple[dict[str, Any], list[str]]:
    integrity_path = checkpoint.with_name(checkpoint.name + ".integrity.json")
    if not integrity_path.is_file():
        return (
            {
                "checkpoint": checkpoint.name,
                "integrity_manifest": integrity_path.name,
                "status": "missing" if required else "optional_missing",
                "checkpoint_bytes": checkpoint_bytes,
                "step": step,
            },
            (
                [f"checkpoint integrity manifest is missing: {integrity_path.name}"]
                if required
                else []
            ),
        )

    integrity, issues = _read_json(
        integrity_path,
        label=f"checkpoint integrity manifest {integrity_path.name}",
    )
    if integrity is None:
        return (
            {
                "checkpoint": checkpoint.name,
                "integrity_manifest": integrity_path.name,
                "status": "invalid",
                "checkpoint_bytes": checkpoint_bytes,
                "step": step,
            },
            issues,
        )

    expected = {
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint_bytes,
        "step": step,
    }
    for field, value in expected.items():
        if integrity.get(field) != value:
            issues.append(
                f"checkpoint integrity manifest {integrity_path.name} "
                f"has mismatched {field}"
            )
    checkpoint_sha256 = integrity.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or SHA256_PATTERN.fullmatch(checkpoint_sha256) is None
    ):
        issues.append(
            f"checkpoint integrity manifest {integrity_path.name} "
            "has an invalid checkpoint_sha256"
        )
    if expected_git_revision is not None:
        if integrity.get("git_revision") != expected_git_revision:
            issues.append(
                f"checkpoint integrity manifest {integrity_path.name} "
                "has mismatched git_revision"
            )
        if integrity.get("git_dirty") is not False:
            issues.append(
                f"checkpoint integrity manifest {integrity_path.name} "
                "does not declare a clean Git state"
            )
    return (
        {
            "checkpoint": checkpoint.name,
            "integrity_manifest": integrity_path.name,
            "status": "invalid" if issues else "metadata_verified",
            "checkpoint_bytes": integrity.get("checkpoint_bytes"),
            "checkpoint_sha256": checkpoint_sha256,
            "step": integrity.get("step"),
            "git_revision": integrity.get("git_revision"),
            "git_dirty": integrity.get("git_dirty"),
            "verification": "metadata_only_no_payload_hash",
        },
        issues,
    )


def _inspect_latest_binding(
    run_dir: Path,
    *,
    checkpoint: dict[str, Any] | None,
    integrity: dict[str, Any] | None,
    required: bool,
    enforce: bool,
) -> tuple[dict[str, Any], list[str]]:
    latest_path = run_dir / "latest.json"
    latest, issues = _read_json(latest_path, label="latest.json")
    if latest is None:
        if required and checkpoint is not None:
            issues.append("latest.json is missing for the newest stable checkpoint")
        return (
            {
                "path": latest_path.name,
                "status": (
                    "missing"
                    if required and checkpoint is not None
                    else (
                        "pending_checkpoint_grace"
                        if checkpoint is not None
                        else "not_available"
                    )
                ),
            },
            issues,
        )
    if checkpoint is None:
        return {
            "path": latest_path.name,
            "status": "pending_checkpoint_grace",
        }, issues

    expected = {
        "checkpoint": checkpoint["name"],
        "checkpoint_bytes": checkpoint["bytes"],
        "step": checkpoint["step"],
    }
    if integrity is not None and integrity.get("status") == "metadata_verified":
        expected.update(
            {
                "integrity_manifest": integrity["integrity_manifest"],
                "checkpoint_sha256": integrity["checkpoint_sha256"],
            }
        )
    binding_issues = []
    for field, value in expected.items():
        if latest.get(field) != value:
            binding_issues.append(f"latest.json has mismatched {field}")
    if enforce:
        issues.extend(binding_issues)
    return (
        {
            "path": latest_path.name,
            "status": (
                "invalid"
                if issues
                else (
                    "pending_checkpoint_grace"
                    if binding_issues
                    else "metadata_verified"
                )
            ),
            "checkpoint": latest.get("checkpoint"),
            "checkpoint_bytes": latest.get("checkpoint_bytes"),
            "checkpoint_sha256": latest.get("checkpoint_sha256"),
            "integrity_manifest": latest.get("integrity_manifest"),
            "step": latest.get("step"),
            "verification": "metadata_only_no_payload_hash",
            "pending_issues": binding_issues if not enforce else [],
        },
        issues,
    )


def inspect_run(
    run_dir: str | Path,
    *,
    expected_steps: int,
    now: float,
    checkpoint_interval: int = 0,
    checkpoint_grace_steps: int = 0,
    checkpoint_integrity_policy: str = "optional",
    expected_checkpoint_revision: str | None = None,
) -> dict[str, Any]:
    if expected_steps < 1 or checkpoint_interval < 0 or checkpoint_grace_steps < 0:
        raise ValueError("monitor run thresholds are invalid")
    if checkpoint_integrity_policy not in CHECKPOINT_INTEGRITY_POLICIES:
        raise ValueError(
            "checkpoint_integrity_policy must be one of "
            + ", ".join(CHECKPOINT_INTEGRITY_POLICIES)
        )
    if expected_checkpoint_revision is not None and (
        not isinstance(expected_checkpoint_revision, str)
        or not expected_checkpoint_revision
    ):
        raise ValueError("expected_checkpoint_revision must be a non-empty string")
    root = Path(run_dir)
    metrics_path = root / "train_metrics.jsonl"
    last, metric_rows, health_issues = _read_metrics(metrics_path)
    report, report_issues = _read_report(root / "training_report.json")
    health_issues.extend(report_issues)
    last_step = int(last.get("step", 0)) if last is not None else 0
    reported_complete = bool(
        report is not None
        and report.get("training_complete") is True
        and int(report.get("completed_steps", -1)) == expected_steps
        and int(report.get("target_steps", -1)) == expected_steps
    )
    checkpoints = []
    checkpoint_integrity = []
    for path in root.glob("checkpoint_step_*.pt") if root.is_dir() else ():
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            step = int(match.group(1))
            size = path.stat().st_size
            checkpoint = {"step": step, "bytes": size, "name": path.name}
            checkpoints.append(checkpoint)
            if size < 1:
                health_issues.append(f"checkpoint is empty: {path.name}")
            required_now = (
                checkpoint_integrity_policy == "required"
                and (reported_complete or last_step >= step + checkpoint_grace_steps)
            )
            integrity, integrity_issues = _inspect_checkpoint_integrity(
                path,
                step=step,
                checkpoint_bytes=size,
                required=required_now,
                expected_git_revision=expected_checkpoint_revision,
            )
            checkpoint_integrity.append(integrity)
            health_issues.extend(integrity_issues)
    checkpoints.sort(key=lambda row: row["step"])
    checkpoint_integrity.sort(key=lambda row: row.get("step") or -1)
    if last_step > expected_steps:
        health_issues.append(
            f"metric step {last_step} exceeds expected target {expected_steps}"
        )
    if checkpoint_interval > 0 and last_step > checkpoint_grace_steps:
        required_checkpoint = (
            (last_step - checkpoint_grace_steps) // checkpoint_interval
        ) * checkpoint_interval
        latest_checkpoint = max((row["step"] for row in checkpoints), default=0)
        if required_checkpoint > latest_checkpoint:
            health_issues.append(
                "checkpoint cadence is late: "
                f"required>={required_checkpoint}, latest={latest_checkpoint}"
            )
    complete = bool(
        report is not None
        and report.get("training_complete") is True
        and int(report.get("completed_steps", -1)) == expected_steps
        and int(report.get("target_steps", -1)) == expected_steps
    )
    if report is not None and report.get("training_complete") is True and not complete:
        health_issues.append("training report completion fields differ from expected steps")
    integrity_by_checkpoint = {
        row["checkpoint"]: row for row in checkpoint_integrity
    }
    newest_checkpoint = checkpoints[-1] if checkpoints else None
    newest_integrity = (
        integrity_by_checkpoint.get(newest_checkpoint["name"])
        if newest_checkpoint is not None
        else None
    )
    newest_ready = bool(
        newest_checkpoint is not None
        and (
            reported_complete
            or last_step >= newest_checkpoint["step"] + checkpoint_grace_steps
        )
    )
    binding_checkpoint = (
        newest_checkpoint
        if newest_ready
        or (
            newest_integrity is not None
            and newest_integrity.get("status") == "metadata_verified"
        )
        else None
    )
    latest_binding, latest_issues = _inspect_latest_binding(
        root,
        checkpoint=binding_checkpoint,
        integrity=newest_integrity,
        required=(
            checkpoint_integrity_policy == "required"
            and newest_ready
        ),
        enforce=newest_ready,
    )
    health_issues.extend(latest_issues)
    activity_mtime = max(
        [
            path.stat().st_mtime
            for path in (metrics_path, root / "training_report.json")
            if path.is_file()
        ]
        or [0.0]
    )
    return {
        "run_dir": root.resolve().as_posix(),
        "exists": root.is_dir(),
        "complete": complete,
        "expected_steps": expected_steps,
        "last_step": last_step,
        "progress_fraction": last_step / expected_steps,
        "last_metric": last,
        "metric_rows": metric_rows,
        "activity_age_seconds": now - activity_mtime if activity_mtime > 0.0 else None,
        "checkpoints": checkpoints,
        "checkpoint_integrity": {
            "policy": checkpoint_integrity_policy,
            "expected_checkpoint_revision": expected_checkpoint_revision,
            "verification": "metadata_only_no_payload_hash",
            "manifests": checkpoint_integrity,
            "latest_binding": latest_binding,
        },
        "training_report": report,
        "health_issues": health_issues,
    }


def build_monitor_report(
    *,
    runs: dict[str, dict[str, Any]],
    training_processes: list[str],
    runbook_processes: list[str],
    stall_seconds: float,
    idle_failure_grace_seconds: float,
    disk: dict[str, int],
    gpu: list[dict[str, Any]],
    updated_at: str,
    hostname: str,
    monitor_name: str = "generation_10pct_matched_pair",
    git: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cofitok = runs["cofitok"]
    dense = runs["dense_identity"]
    health_issues = [
        f"{name}: {issue}"
        for name, run in runs.items()
        for issue in run.get("health_issues", [])
    ]
    if health_issues:
        status, stage, issues = "failed", "training_health", health_issues
    elif cofitok["complete"] and dense["complete"]:
        status, stage, issues = "pass", "complete", []
    else:
        incomplete = [name for name, run in runs.items() if not run["complete"]]
        process_text = "\n".join(training_processes)
        active_methods = [
            name
            for name, pattern in METHOD_PROCESS_PATTERNS.items()
            if name in incomplete and pattern.search(process_text) is not None
        ]
        if len(active_methods) == 1:
            current_name = active_methods[0]
        else:
            current_name = min(
                incomplete,
                key=lambda name: (
                    float(runs[name]["activity_age_seconds"])
                    if runs[name].get("activity_age_seconds") is not None
                    else float("inf")
                ),
            )
        current = runs[current_name]
        age = current.get("activity_age_seconds")
        if training_processes and age is not None and age > stall_seconds:
            status, stage = "stalled", f"{current_name}_training"
            issues = [f"{current_name} metrics have not advanced for {age:.1f} seconds"]
        elif training_processes:
            status, stage, issues = "running", f"{current_name}_training", []
        elif runbook_processes:
            status, stage, issues = "waiting", f"{current_name}_transition", []
        else:
            observed_ages = [
                float(run["activity_age_seconds"])
                for run in runs.values()
                if run.get("activity_age_seconds") is not None
            ]
            freshest_age = min(observed_ages, default=float("inf"))
            if freshest_age <= idle_failure_grace_seconds:
                status, stage, issues = "waiting", f"{current_name}_transition", []
            else:
                status, stage = "failed", f"{current_name}_training"
                issues = [
                    "matched queue is incomplete without an active training or runbook process"
                ]
    return {
        "schema_version": 2,
        "monitor": monitor_name,
        "status": status,
        "stage": stage,
        "updated_at": updated_at,
        "hostname": hostname,
        "git": git or {},
        "thresholds": {
            "stall_seconds": stall_seconds,
            "idle_failure_grace_seconds": idle_failure_grace_seconds,
        },
        "processes": {
            "training": training_processes,
            "runbook": runbook_processes,
        },
        "runs": runs,
        "disk": disk,
        "gpu": gpu,
        "issues": issues,
    }
