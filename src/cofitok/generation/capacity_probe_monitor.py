from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_CONFIGURED_STEPS,
    CAPACITY_PROBE_STOP_STEP,
)
from cofitok.generation.capacity_probe_training import PARTIAL_TRAINING_ROLE
from cofitok.monitoring import (
    CHECKPOINT_PATTERN,
    _inspect_checkpoint_integrity,
    _inspect_latest_binding,
    _inspect_run_manifest,
    _read_json,
    _read_metrics,
    _read_report,
)


CAPACITY_PROBE_MONITOR_SCHEMA_VERSION = 1
CAPACITY_PROBE_MONITOR_NAME = "generation_stability_capacity_probe_250m_10k"
CAPACITY_PROBE_METHOD_PATTERNS = {
    "cofitok": re.compile(
        r"capacity_probe.*(?:rgbtail|base256_cofitok)",
        re.IGNORECASE,
    ),
    "dense_identity": re.compile(
        r"capacity_probe.*(?:dense|base256_dense_identity)",
        re.IGNORECASE,
    ),
}


def _validated_partial_report(
    path: Path,
    *,
    expected_run_dir: Path,
    expected_revision: str,
    expected_branch: str,
    expected_stop_step: int,
    expected_configured_steps: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    payload, issues = _read_json(path, label="capacity probe partial validation")
    if payload is None:
        return None, issues
    checkpoint = payload.get("checkpoint")
    authorization = payload.get("authorization_boundary")
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if (
        int(payload.get("schema_version", -1)) != 1
        or payload.get("status") != "pass"
        or payload.get("role") != PARTIAL_TRAINING_ROLE
        or Path(str(payload.get("run_dir", ""))).resolve() != expected_run_dir
        or payload.get("git") != expected_git
        or int(payload.get("configured_steps", -1)) != expected_configured_steps
        or int(payload.get("completed_steps", -1)) != expected_stop_step
        or payload.get("training_complete") is not False
        or payload.get("intentional_partial_stop") is not True
        or not isinstance(checkpoint, Mapping)
        or int(payload.get("effective_batch_size", -1)) != 64
        or int(payload.get("images_seen", -1)) != expected_stop_step * 64
        or not isinstance(authorization, Mapping)
        or authorization.get("capacity_probe_training_complete") is not True
        or authorization.get("full_training_complete") is not False
        or authorization.get("full_300k_launch_allowed") is not False
        or authorization.get("formal_generation_claim_allowed") is not False
        or authorization.get("release_allowed") is not False
    ):
        issues.append("capacity probe partial validation contract differs")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    expected_checkpoint = expected_run_dir / (
        f"checkpoint_step_{expected_stop_step:08d}.pt"
    )
    integrity_identity = checkpoint.get("integrity_manifest")
    integrity_path = (
        Path(str(integrity_identity.get("path", ""))).resolve()
        if isinstance(integrity_identity, Mapping)
        else Path("").resolve()
    )
    if (
        checkpoint_path != expected_checkpoint
        or type(checkpoint.get("bytes")) is not int
        or int(checkpoint.get("bytes", 0)) < 1
        or not isinstance(checkpoint.get("sha256"), str)
        or len(str(checkpoint.get("sha256"))) != 64
        or not isinstance(integrity_identity, Mapping)
        or type(integrity_identity.get("bytes")) is not int
        or int(integrity_identity.get("bytes", 0)) < 1
        or not isinstance(integrity_identity.get("sha256"), str)
        or len(str(integrity_identity.get("sha256"))) != 64
        or integrity_path
        != expected_checkpoint.with_name(expected_checkpoint.name + ".integrity.json")
    ):
        issues.append("capacity probe partial validation checkpoint differs")
    return payload, issues


def inspect_capacity_probe_run(
    run_dir: str | Path,
    *,
    validation_report: str | Path,
    now: float,
    expected_revision: str,
    expected_branch: str,
    expected_stop_step: int = CAPACITY_PROBE_STOP_STEP,
    expected_configured_steps: int = CAPACITY_PROBE_CONFIGURED_STEPS,
    checkpoint_interval: int = 5_000,
    checkpoint_grace_steps: int = 250,
) -> dict[str, Any]:
    if (
        expected_stop_step < 1
        or expected_configured_steps <= expected_stop_step
        or checkpoint_interval < 1
        or checkpoint_grace_steps < 0
        or not expected_revision
        or not expected_branch
    ):
        raise ValueError("capacity probe monitor expectations are invalid")
    root = Path(run_dir).resolve()
    validation_path = Path(validation_report).resolve()
    metrics_path = root / "train_metrics.jsonl"
    last, metric_rows, health_issues, rows = _read_metrics(metrics_path)
    last_step = int(last.get("step", 0)) if last is not None else 0
    report, report_issues = _read_report(root / "training_report.json")
    health_issues.extend(report_issues)
    manifest, manifest_issues = _inspect_run_manifest(
        root,
        metrics_rows=rows,
        expected_steps=expected_configured_steps,
        checkpoint_interval=checkpoint_interval,
        expected_git_revision=expected_revision,
        required=True,
    )
    health_issues.extend(manifest_issues)
    if last_step > expected_stop_step:
        health_issues.append(
            f"capacity probe metric step {last_step} exceeds intentional stop "
            f"{expected_stop_step}"
        )

    checkpoints: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    for path in root.glob("checkpoint_step_*.pt") if root.is_dir() else ():
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        size = path.stat().st_size
        checkpoints.append({"step": step, "bytes": size, "name": path.name})
        if size < 1:
            health_issues.append(f"checkpoint is empty: {path.name}")
        if step > expected_stop_step:
            health_issues.append(
                f"capacity probe checkpoint exceeds intentional stop: {path.name}"
            )
        stable = last_step >= step + checkpoint_grace_steps or (
            last_step == expected_stop_step and step == expected_stop_step
        )
        integrity, integrity_issues = _inspect_checkpoint_integrity(
            path,
            step=step,
            checkpoint_bytes=size,
            required=stable,
            expected_git_revision=expected_revision,
        )
        integrity_rows.append(integrity)
        health_issues.extend(integrity_issues)
    checkpoints.sort(key=lambda value: value["step"])
    integrity_rows.sort(key=lambda value: int(value.get("step", -1)))
    if last_step > checkpoint_grace_steps:
        required_checkpoint = (
            (last_step - checkpoint_grace_steps) // checkpoint_interval
        ) * checkpoint_interval
        if last_step == expected_stop_step:
            required_checkpoint = expected_stop_step
        latest_checkpoint = max((row["step"] for row in checkpoints), default=0)
        if required_checkpoint > latest_checkpoint:
            health_issues.append(
                "checkpoint cadence is late: "
                f"required>={required_checkpoint}, latest={latest_checkpoint}"
            )

    newest_checkpoint = checkpoints[-1] if checkpoints else None
    integrity_by_name = {row["checkpoint"]: row for row in integrity_rows}
    newest_integrity = (
        integrity_by_name.get(newest_checkpoint["name"])
        if newest_checkpoint is not None
        else None
    )
    newest_stable = bool(
        newest_checkpoint is not None
        and (
            last_step >= newest_checkpoint["step"] + checkpoint_grace_steps
            or (
                last_step == expected_stop_step
                and newest_checkpoint["step"] == expected_stop_step
            )
        )
    )
    latest_binding, latest_issues = _inspect_latest_binding(
        root,
        checkpoint=newest_checkpoint if newest_stable else None,
        integrity=newest_integrity,
        required=newest_stable,
        enforce=newest_stable,
    )
    health_issues.extend(latest_issues)

    clean_stop_report = False
    if report is not None:
        reported_steps = int(report.get("completed_steps", -1))
        reported_target = int(report.get("target_steps", -1))
        clean_stop_report = bool(
            report.get("training_complete") is False
            and reported_steps == expected_stop_step
            and reported_target == expected_configured_steps
            and report.get("stop_requested") is False
            and report.get("stop_signal") is None
        )
        if (
            reported_target != expected_configured_steps
            or reported_steps < 0
            or reported_steps > expected_stop_step
            or report.get("training_complete") is True
            or (reported_steps == expected_stop_step and not clean_stop_report)
        ):
            health_issues.append(
                "capacity probe training report is not the exact intentional stop"
            )

    partial_validation = None
    if validation_path.is_file():
        partial_validation, validation_issues = _validated_partial_report(
            validation_path,
            expected_run_dir=root,
            expected_revision=expected_revision,
            expected_branch=expected_branch,
            expected_stop_step=expected_stop_step,
            expected_configured_steps=expected_configured_steps,
        )
        health_issues.extend(validation_issues)
    complete = bool(
        partial_validation is not None
        and clean_stop_report
        and last_step == expected_stop_step
        and newest_checkpoint is not None
        and newest_checkpoint["step"] == expected_stop_step
        and newest_integrity is not None
        and newest_integrity.get("status") == "metadata_verified"
        and latest_binding.get("status") == "metadata_verified"
        and not health_issues
    )
    activity_paths = [
        metrics_path,
        root / "training_report.json",
        validation_path,
    ]
    activity_mtime = max(
        (path.stat().st_mtime for path in activity_paths if path.is_file()),
        default=0.0,
    )
    return {
        "run_dir": root.as_posix(),
        "exists": root.is_dir(),
        "complete": complete,
        "intentional_partial_stop": True,
        "expected_stop_step": expected_stop_step,
        "configured_steps": expected_configured_steps,
        "last_step": last_step,
        "progress_fraction": last_step / expected_stop_step,
        "last_metric": last,
        "metric_rows": metric_rows,
        "activity_age_seconds": (
            now - activity_mtime if activity_mtime > 0.0 else None
        ),
        "checkpoints": checkpoints,
        "checkpoint_integrity": {
            "policy": "required_after_grace",
            "expected_checkpoint_revision": expected_revision,
            "verification": "metadata_only_no_payload_hash",
            "manifests": integrity_rows,
            "latest_binding": latest_binding,
        },
        "run_manifest": manifest,
        "training_report": report,
        "partial_validation_path": validation_path.as_posix(),
        "partial_validation": partial_validation,
        "health_issues": health_issues,
    }


def build_capacity_probe_monitor_report(
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
    git: dict[str, Any],
    monitor_name: str = CAPACITY_PROBE_MONITOR_NAME,
) -> dict[str, Any]:
    if set(runs) != {"cofitok", "dense_identity"}:
        raise ValueError("capacity probe monitor run set differs")
    health_issues = [
        f"{method}: {issue}"
        for method, run in runs.items()
        for issue in run.get("health_issues", [])
    ]
    if health_issues:
        status, stage, issues = "failed", "training_health", health_issues
    elif all(run.get("complete") is True for run in runs.values()):
        status, stage, issues = "pass", "complete", []
    else:
        incomplete = [name for name, run in runs.items() if not run.get("complete")]
        process_text = "\n".join(training_processes)
        active = [
            name
            for name, pattern in CAPACITY_PROBE_METHOD_PATTERNS.items()
            if name in incomplete and pattern.search(process_text) is not None
        ]
        if len(active) == 1:
            current_name = active[0]
        else:
            current_name = min(
                incomplete,
                key=lambda name: (
                    float(runs[name]["activity_age_seconds"])
                    if runs[name].get("activity_age_seconds") is not None
                    else float("inf")
                ),
            )
        age = runs[current_name].get("activity_age_seconds")
        if training_processes and age is not None and age > stall_seconds:
            status, stage = "stalled", f"{current_name}_training"
            issues = [
                f"{current_name} metrics have not advanced for {age:.1f} seconds"
            ]
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
            freshest = min(observed_ages, default=float("inf"))
            if freshest <= idle_failure_grace_seconds:
                status, stage, issues = (
                    "waiting",
                    f"{current_name}_transition",
                    [],
                )
            else:
                status, stage = "failed", f"{current_name}_training"
                issues = [
                    "capacity probe is incomplete without an active training or "
                    "runbook process"
                ]
    return {
        "schema_version": CAPACITY_PROBE_MONITOR_SCHEMA_VERSION,
        "monitor": monitor_name,
        "status": status,
        "stage": stage,
        "updated_at": updated_at,
        "hostname": hostname,
        "git": git,
        "scope": {
            "intentional_partial_stop_step": CAPACITY_PROBE_STOP_STEP,
            "configured_100k_completion_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
        },
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
