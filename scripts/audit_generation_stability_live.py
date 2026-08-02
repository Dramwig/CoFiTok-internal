from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.monitoring import inspect_run
from cofitok.reporting import write_json_report


ROLE_NAMES = (
    "controller",
    "pair_monitor",
    "watchdog",
    "trainer",
    "posteval_waiter",
    "readiness_waiter",
)
ACTIVE_PAIR_STATES = {"running", "pass"}
ACTIVE_WATCHDOG_STATES = {"running", "passed"}
ACTIVE_RECOVERY_STATES = {"running", "pass"}
ACTIVE_WAITER_STATES = {"waiting", "running", "pass"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    return raw, {
        "path": path.resolve().as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _json_object(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, identity = _source_bytes(path)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload, identity


def git_identity(project: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    return {
        "path": project.resolve().as_posix(),
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--short", "--untracked-files=no")),
    }


def read_metrics(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw, identity = _source_bytes(path)
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid metrics JSON at line {line_number}: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"metrics line {line_number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError("dense metrics are empty")
    return rows, identity


def metrics_evidence(
    rows: list[dict[str, Any]],
    *,
    effective_batch: int,
    metric_age_seconds: float,
    metric_stale_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    previous_step = 0
    for index, row in enumerate(rows):
        step = row.get("step")
        valid_step = not (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step <= previous_step
        )
        if not valid_step:
            issues.append(f"metrics step is not strictly increasing at row {index}")
        else:
            previous_step = step
        samples_seen = row.get("samples_seen")
        if (
            not valid_step
            or not isinstance(step, int)
            or not isinstance(samples_seen, int)
            or isinstance(samples_seen, bool)
            or samples_seen != step * effective_batch
        ):
            issues.append(
                f"samples_seen differs from step * {effective_batch} at row {index}"
            )
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                issues.append(f"metric {key} is non-finite at row {index}")
    if metric_age_seconds > metric_stale_seconds:
        issues.append("dense metrics exceed the configured stale threshold")

    recent_intervals = []
    for previous, current in zip(rows[-7:-1], rows[-6:]):
        previous_step = previous.get("step")
        current_step = current.get("step")
        previous_elapsed = previous.get("elapsed_seconds")
        current_elapsed = current.get("elapsed_seconds")
        if (
            not isinstance(previous_step, int)
            or isinstance(previous_step, bool)
            or not isinstance(current_step, int)
            or isinstance(current_step, bool)
            or not isinstance(previous_elapsed, (int, float))
            or isinstance(previous_elapsed, bool)
            or not math.isfinite(previous_elapsed)
            or not isinstance(current_elapsed, (int, float))
            or isinstance(current_elapsed, bool)
            or not math.isfinite(current_elapsed)
        ):
            continue
        recent_intervals.append(
            {
                "from_step": previous_step,
                "to_step": current_step,
                "elapsed_seconds": float(current_elapsed) - float(previous_elapsed),
            }
        )
    validation_rows = [row for row in rows if "validation_epsilon_mse" in row]
    return (
        {
            "row_count": len(rows),
            "last": rows[-1],
            "metric_age_seconds": metric_age_seconds,
            "metric_stale_seconds": metric_stale_seconds,
            "strictly_increasing": not any(
                "strictly increasing" in issue for issue in issues
            ),
            "samples_seen_exact": not any("samples_seen" in issue for issue in issues),
            "all_finite": not any("non-finite" in issue for issue in issues),
            "validation_event_count": len(validation_rows),
            "latest_validation": validation_rows[-1] if validation_rows else None,
            "recent_logged_intervals": recent_intervals,
        },
        issues,
    )


def collect_processes(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or not directory.is_dir():
            continue
        try:
            argv = (
                (directory / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            if not argv:
                continue
            stat_tail = (directory / "stat").read_text(encoding="utf-8").rsplit(
                ")", 1
            )[1].split()
            process = {
                "pid": int(directory.name),
                "ppid": int(stat_tail[1]),
                "start_ticks": int(stat_tail[19]),
                "argv": argv,
                "cwd": (directory / "cwd").resolve().as_posix(),
            }
            fd6 = directory / "fd" / "6"
            if fd6.exists():
                process["fd6"] = fd6.resolve().as_posix()
            processes.append(process)
        except (FileNotFoundError, IndexError, OSError, PermissionError, ValueError):
            continue
    return sorted(processes, key=lambda item: int(item["pid"]))


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value not in {"", "-"} else None


def collect_gpu() -> dict[str, Any]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    compute_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    pmon_query = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
        capture_output=True,
        text=True,
        check=False,
    )
    devices = []
    if gpu_query.returncode == 0:
        for line in gpu_query.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) != 6:
                continue
            devices.append(
                {
                    "index": int(values[0]),
                    "memory_total_mib": int(values[1]),
                    "memory_used_mib": int(values[2]),
                    "memory_free_mib": int(values[3]),
                    "utilization_percent": int(values[4]),
                    "temperature_c": int(values[5]),
                }
            )
    compute: list[dict[str, Any]] = []
    if compute_query.returncode == 0:
        for line in compute_query.stdout.splitlines():
            values = [value.strip() for value in line.split(",", 2)]
            if len(values) != 3:
                continue
            compute.append(
                {
                    "pid": int(values[0]),
                    "used_memory_mib": int(values[1]),
                    "process_name": values[2],
                }
            )
    pmon_by_pid: dict[int, dict[str, Any]] = {}
    if pmon_query.returncode == 0:
        for line in pmon_query.stdout.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            if len(values) < 11:
                continue
            try:
                pid = int(values[1])
            except ValueError:
                continue
            pmon_by_pid[pid] = {
                "sm_percent": _parse_optional_int(values[3]),
                "memory_percent": _parse_optional_int(values[4]),
                "framebuffer_memory_mib": _parse_optional_int(values[9]),
            }
    for process in compute:
        process.update(pmon_by_pid.get(int(process["pid"]), {}))
    return {
        "devices": devices,
        "compute_processes": compute,
        "query_status": {
            "gpu": gpu_query.returncode,
            "compute": compute_query.returncode,
            "pmon": pmon_query.returncode,
        },
    }


def _contains(process: dict[str, Any], *needles: str) -> bool:
    argv = str(process.get("argv", "")).replace("\\", "/")
    return all(needle.replace("\\", "/") in argv for needle in needles)


def classify_roles(
    processes: list[dict[str, Any]],
    *,
    gpu_compute_pids: set[int],
    output_root: Path,
    dense_run: Path,
    control_project: Path,
    training_project: Path,
    posteval_project: Path,
    readiness_project: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    output_text = output_root.resolve().as_posix()
    dense_text = dense_run.resolve().as_posix()
    training_text = training_project.resolve().as_posix()
    posteval_text = posteval_project.resolve().as_posix()
    readiness_text = readiness_project.resolve().as_posix()
    control_text = control_project.resolve().as_posix()
    roles = {
        "controller": [
            process
            for process in processes
            if int(process.get("ppid", -1)) == 1
            and _contains(
                process,
                control_text,
                "generation_stability_ema_teacher_dense_recovery_after_transition_failure.sh",
            )
        ],
        "pair_monitor": [
            process
            for process in processes
            if _contains(process, "monitor_generation_pair.py", output_text)
        ],
        "watchdog": [
            process
            for process in processes
            if _contains(
                process,
                "run_generation_training_watchdog.py",
                dense_text,
            )
            and process.get("cwd") == training_text
        ],
        "trainer": [
            process
            for process in processes
            if int(process.get("pid", -1)) in gpu_compute_pids
            and _contains(process, "scripts/train_generation.py", dense_text)
            and process.get("cwd") == training_text
        ],
        "posteval_waiter": [
            process
            for process in processes
            if _contains(
                process,
                "run_generation_stability_50k_posteval_waiter.py",
                output_text,
            )
            and process.get("cwd") == posteval_text
        ],
        "readiness_waiter": [
            process
            for process in processes
            if _contains(
                process,
                "run_generation_stability_full_readiness_waiter.py",
                output_text,
            )
            and process.get("cwd") == readiness_text
        ],
    }
    trainer_pids = {int(item["pid"]) for item in roles["trainer"]}
    unrelated = [
        process
        for process in processes
        if int(process.get("pid", -1)) in gpu_compute_pids
        and int(process.get("pid", -1)) not in trainer_pids
    ]
    return roles, unrelated


def _status_issue(
    label: str,
    payload: dict[str, Any],
    allowed: set[str],
) -> list[str]:
    status = payload.get("status")
    return [] if status in allowed else [f"{label} status is {status!r}"]


def _validate_git(
    label: str,
    identity: dict[str, Any],
    *,
    revision: str,
    branch: str,
) -> list[str]:
    issues = []
    if identity.get("revision") != revision:
        issues.append(f"{label} revision changed")
    if identity.get("branch") != branch:
        issues.append(f"{label} branch changed")
    if identity.get("tracked_dirty") is not False:
        issues.append(f"{label} tracked worktree is dirty")
    return issues


def evaluate_snapshot(
    *,
    pair_monitor: dict[str, Any],
    dense_inspection: dict[str, Any],
    dense_metrics: dict[str, Any],
    dense_manifest: dict[str, Any],
    watchdog: dict[str, Any],
    recovery: dict[str, Any],
    posteval: dict[str, Any],
    readiness: dict[str, Any],
    git_identities: dict[str, dict[str, Any]],
    roles: dict[str, list[dict[str, Any]]],
    expected_role_pids: dict[str, int],
    unrelated_gpu_processes: list[dict[str, Any]],
    gpu: dict[str, Any],
    lock_path: Path,
    flock_probe_returncode: int,
    free_bytes: int,
    total_bytes: int,
    matched_required_bytes: int,
    aggregate_required_bytes: int,
    expected_control_revision: str,
    expected_control_branch: str,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_evaluation_revision: str,
    expected_evaluation_branch: str,
    expected_readiness_revision: str,
    expected_readiness_branch: str,
    expected_runtime_sha256: str,
    expected_parameter_count: int,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    observations: list[str] = []
    issues.extend(_status_issue("pair monitor", pair_monitor, ACTIVE_PAIR_STATES))
    issues.extend(_status_issue("watchdog", watchdog, ACTIVE_WATCHDOG_STATES))
    issues.extend(_status_issue("recovery", recovery, ACTIVE_RECOVERY_STATES))
    issues.extend(_status_issue("post-eval waiter", posteval, ACTIVE_WAITER_STATES))
    issues.extend(_status_issue("readiness waiter", readiness, ACTIVE_WAITER_STATES))
    if pair_monitor.get("issues"):
        issues.append("pair monitor reports issues")
    if pair_monitor.get("status") == "running" and pair_monitor.get("stage") != (
        "dense_identity_training"
    ):
        issues.append("pair monitor is not in dense_identity_training")
    if dense_inspection.get("health_issues"):
        issues.extend(
            f"dense inspection: {issue}"
            for issue in dense_inspection["health_issues"]
        )
    if dense_inspection.get("run_manifest", {}).get("status") != "verified":
        issues.append("dense metadata-only run manifest inspection is not verified")
    checkpoint_integrity = dense_inspection.get("checkpoint_integrity", {})
    if checkpoint_integrity.get("policy") != "required":
        issues.append("dense checkpoint integrity policy is not required")
    if checkpoint_integrity.get("verification") != "metadata_only_no_payload_hash":
        issues.append("dense live audit attempted an unsupported checkpoint policy")
    if dense_metrics.get("strictly_increasing") is not True:
        issues.append("dense metrics are not strictly increasing")
    if dense_metrics.get("samples_seen_exact") is not True:
        issues.append("dense samples_seen arithmetic is invalid")
    if dense_metrics.get("all_finite") is not True:
        issues.append("dense metrics contain non-finite values")
    manifest_git = dense_manifest.get("git", {})
    if (
        manifest_git.get("revision") != expected_training_revision
        or manifest_git.get("branch") != expected_training_branch
        or manifest_git.get("dirty") is not False
    ):
        issues.append("dense run manifest training identity changed")
    if dense_manifest.get("runtime_environment_sha256") != expected_runtime_sha256:
        issues.append("dense run manifest runtime environment changed")
    if int(dense_manifest.get("parameter_count", -1)) != expected_parameter_count:
        issues.append("dense run manifest parameter count changed")
    if recovery.get("full_training_launch_allowed") is not False:
        issues.append("dense recovery unexpectedly allows full training")
    if readiness.get("full_training_launch_allowed") is not False:
        issues.append("readiness waiter unexpectedly allows full training")

    expected_git = {
        "control": (expected_control_revision, expected_control_branch),
        "training": (expected_training_revision, expected_training_branch),
        "evaluation": (expected_evaluation_revision, expected_evaluation_branch),
        "readiness": (expected_readiness_revision, expected_readiness_branch),
    }
    for label, (revision, branch) in expected_git.items():
        issues.extend(
            _validate_git(
                label,
                git_identities.get(label, {}),
                revision=revision,
                branch=branch,
            )
        )

    role_evidence = {}
    for role in ROLE_NAMES:
        candidates = roles.get(role, [])
        role_evidence[role] = {
            "count": len(candidates),
            "processes": candidates,
            "expected_launch_pid": expected_role_pids.get(role),
        }
        if len(candidates) != 1:
            issues.append(f"expected exactly one {role}, found {len(candidates)}")
            continue
        expected_pid = expected_role_pids.get(role)
        if expected_pid is not None and int(candidates[0]["pid"]) != expected_pid:
            warnings.append(
                f"{role} PID changed from launch identity {expected_pid} to "
                f"{candidates[0]['pid']}"
            )
    controllers = roles.get("controller", [])
    lock_text = lock_path.resolve().as_posix()
    if len(controllers) == 1 and controllers[0].get("fd6") != lock_text:
        issues.append("recovery controller fd 6 does not point to dense_recovery.lock")
    if flock_probe_returncode != 1:
        issues.append("dense_recovery.lock is not held by the active controller")

    matched_headroom = free_bytes - matched_required_bytes
    aggregate_headroom = free_bytes - aggregate_required_bytes
    if matched_headroom < 0:
        issues.append("matched 50K storage runway is negative")
    if aggregate_headroom < 0:
        warnings.append("aggregate completion storage runway is negative")
    if unrelated_gpu_processes:
        observations.append(
            f"observed {len(unrelated_gpu_processes)} unrelated GPU compute process(es)"
        )
    if gpu.get("query_status", {}).get("compute") != 0:
        issues.append("GPU compute process query failed")

    status = "failed" if issues else ("warning" if warnings else "healthy")
    if pair_monitor.get("status") == "pass" and not issues and not warnings:
        status = "complete"
    return {
        "schema_version": 1,
        "role": "generation_stability_dense_live_audit",
        "status": status,
        "read_only": True,
        "process_control_performed": False,
        "checkpoint_payload_hashes_performed": False,
        "full_training_launch_allowed": False,
        "pair_monitor": {
            "status": pair_monitor.get("status"),
            "stage": pair_monitor.get("stage"),
            "updated_at": pair_monitor.get("updated_at"),
            "issues": pair_monitor.get("issues"),
        },
        "dense": {
            "inspection": dense_inspection,
            "metrics": dense_metrics,
            "manifest": {
                "git": dense_manifest.get("git"),
                "runtime_environment_sha256": dense_manifest.get(
                    "runtime_environment_sha256"
                ),
                "parameter_count": dense_manifest.get("parameter_count"),
            },
        },
        "controllers": {
            "watchdog": watchdog,
            "recovery": recovery,
            "posteval_waiter": posteval,
            "readiness_waiter": readiness,
        },
        "git_identities": git_identities,
        "process_roles": role_evidence,
        "lock": {
            "path": lock_text,
            "controller_fd": 6,
            "nonblocking_probe_returncode": flock_probe_returncode,
            "held": flock_probe_returncode == 1,
        },
        "gpu": {
            **gpu,
            "unrelated_compute_processes": unrelated_gpu_processes,
        },
        "storage": {
            "total_bytes": total_bytes,
            "free_bytes": free_bytes,
            "matched_50k": {
                "required_bytes": matched_required_bytes,
                "headroom_bytes": matched_headroom,
                "status": "pass" if matched_headroom >= 0 else "fail",
            },
            "aggregate_completion": {
                "required_bytes": aggregate_required_bytes,
                "headroom_bytes": aggregate_headroom,
                "status": "pass" if aggregate_headroom >= 0 else "fail",
            },
        },
        "issues": sorted(set(issues)),
        "warnings": sorted(set(warnings)),
        "observations": sorted(set(observations)),
    }


def _expected_role_pids(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        role, separator, raw_pid = value.partition("=")
        if separator != "=" or role not in ROLE_NAMES:
            raise ValueError(f"invalid expected role PID: {value}")
        pid = int(raw_pid)
        if pid < 1 or role in result:
            raise ValueError(f"invalid expected role PID: {value}")
        result[role] = pid
    return result


def _validate_sha256(value: str, label: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one read-only operational audit of active stability dense training."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--full-output-root", type=Path, required=True)
    parser.add_argument("--dense-run", default="dense_rollout_x0_u2_ema_teacher")
    parser.add_argument("--control-project", type=Path, required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--posteval-project", type=Path, required=True)
    parser.add_argument("--readiness-project", type=Path, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-evaluation-revision", required=True)
    parser.add_argument("--expected-evaluation-branch", required=True)
    parser.add_argument("--expected-readiness-revision", required=True)
    parser.add_argument("--expected-readiness-branch", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--expected-parameter-count", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, default=50_000)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--checkpoint-grace-steps", type=int, default=250)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--metric-stale-seconds", type=float, default=1_800.0)
    parser.add_argument("--matched-required-bytes", type=int, required=True)
    parser.add_argument("--aggregate-required-bytes", type=int, required=True)
    parser.add_argument(
        "--expected-role-pid",
        action="append",
        default=[],
        metavar="ROLE=PID",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.expected_parameter_count,
        args.expected_steps,
        args.checkpoint_interval,
        args.effective_batch,
        args.metric_stale_seconds,
        args.matched_required_bytes,
        args.aggregate_required_bytes,
    ) <= 0:
        raise ValueError("audit thresholds must be positive")
    if args.checkpoint_grace_steps < 0:
        raise ValueError("checkpoint grace steps must be non-negative")
    for value, label in (
        (args.expected_runtime_sha256, "expected runtime SHA256"),
    ):
        _validate_sha256(value, label)

    output_root = args.output_root.resolve()
    full_output_root = args.full_output_root.resolve()
    dense_run = output_root / args.dense_run
    source_paths = {
        "pair_monitor": output_root / "pair_monitor.json",
        "dense_metrics": dense_run / "train_metrics.jsonl",
        "dense_manifest": dense_run / "run_manifest.json",
        "watchdog": dense_run / "training_watchdog.json",
        "recovery": output_root / "reports" / "dense_recovery_status.json",
        "posteval": output_root / "reports" / "posteval_waiter.json",
        "readiness": full_output_root / "reports" / "readiness_waiter.json",
    }
    sources: dict[str, dict[str, Any]] = {}
    source_identities: dict[str, dict[str, Any]] = {}
    for name, path in source_paths.items():
        if name == "dense_metrics":
            continue
        sources[name], source_identities[name] = _json_object(path)
    metric_rows, source_identities["dense_metrics"] = read_metrics(
        source_paths["dense_metrics"]
    )
    metric_age = time.time() - source_paths["dense_metrics"].stat().st_mtime
    metric_summary, metric_issues = metrics_evidence(
        metric_rows,
        effective_batch=args.effective_batch,
        metric_age_seconds=metric_age,
        metric_stale_seconds=args.metric_stale_seconds,
    )
    dense_inspection = inspect_run(
        dense_run,
        expected_steps=args.expected_steps,
        now=time.time(),
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_grace_steps=args.checkpoint_grace_steps,
        checkpoint_integrity_policy="required",
        expected_checkpoint_revision=args.expected_training_revision,
    )
    dense_inspection["health_issues"].extend(metric_issues)

    processes = collect_processes()
    gpu = collect_gpu()
    gpu_pids = {
        int(process["pid"]) for process in gpu.get("compute_processes", [])
    }
    roles, unrelated = classify_roles(
        processes,
        gpu_compute_pids=gpu_pids,
        output_root=output_root,
        dense_run=dense_run,
        control_project=args.control_project,
        training_project=args.training_project,
        posteval_project=args.posteval_project,
        readiness_project=args.readiness_project,
    )
    process_by_pid = {int(process["pid"]): process for process in processes}
    unrelated_evidence = [
        {
            **process,
            **next(
                (
                    gpu_process
                    for gpu_process in gpu.get("compute_processes", [])
                    if int(gpu_process["pid"]) == int(process["pid"])
                ),
                {},
            ),
        }
        for process in unrelated
    ]
    lock_path = output_root / "dense_recovery.lock"
    lock_probe = subprocess.run(
        ["flock", "-n", str(lock_path), "true"],
        capture_output=True,
        text=True,
        check=False,
    )
    usage = shutil.disk_usage(output_root)
    report = evaluate_snapshot(
        pair_monitor=sources["pair_monitor"],
        dense_inspection=dense_inspection,
        dense_metrics=metric_summary,
        dense_manifest=sources["dense_manifest"],
        watchdog=sources["watchdog"],
        recovery=sources["recovery"],
        posteval=sources["posteval"],
        readiness=sources["readiness"],
        git_identities={
            "control": git_identity(args.control_project),
            "training": git_identity(args.training_project),
            "evaluation": git_identity(args.posteval_project),
            "readiness": git_identity(args.readiness_project),
        },
        roles=roles,
        expected_role_pids=_expected_role_pids(args.expected_role_pid),
        unrelated_gpu_processes=unrelated_evidence,
        gpu=gpu,
        lock_path=lock_path,
        flock_probe_returncode=lock_probe.returncode,
        free_bytes=usage.free,
        total_bytes=usage.total,
        matched_required_bytes=args.matched_required_bytes,
        aggregate_required_bytes=args.aggregate_required_bytes,
        expected_control_revision=args.expected_control_revision,
        expected_control_branch=args.expected_control_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
        expected_evaluation_revision=args.expected_evaluation_revision,
        expected_evaluation_branch=args.expected_evaluation_branch,
        expected_readiness_revision=args.expected_readiness_revision,
        expected_readiness_branch=args.expected_readiness_branch,
        expected_runtime_sha256=args.expected_runtime_sha256,
        expected_parameter_count=args.expected_parameter_count,
    )
    report.update(
        {
            "generated_at": utc_now(),
            "hostname": socket.gethostname(),
            "sources": source_identities,
            "process_table_size": len(process_by_pid),
        }
    )
    if args.output is not None:
        write_json_report(args.output, report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
