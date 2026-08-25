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
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PILOT_STEP = 50_000
SCHEDULER_HORIZON = 100_000
EFFECTIVE_BATCH_SIZE = 64
CHECKPOINT_INTERVAL = 5_000
CHECKPOINT_GRACE_STEPS = 250
DATASET_IDENTITY_SHA256 = (
    "6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659"
)
RUNTIME_ENVIRONMENT_SHA256 = (
    "d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e"
)
METHODS = ("cofitok", "dense_identity")
RUN_NAMES = {"cofitok": "cofitok_gamma5", "dense_identity": "dense_gamma5"}
CONFIG_NAMES = {
    "cofitok": (
        "imagenet256_min_snr_gamma5_quality_repair_rgbtail3_rollout_x0_u2_"
        "ema_teacher_k8_100k_horizon_50k_pilot.json"
    ),
    "dense_identity": (
        "imagenet256_min_snr_gamma5_quality_repair_rollout_x0_u2_"
        "ema_teacher_dense_100k_horizon_50k_pilot.json"
    ),
}
EVALUATION_ARMS = (
    "legacy_gamma0_cofitok",
    "legacy_gamma0_dense_identity",
    "pilot_gamma5_cofitok",
    "pilot_gamma5_dense_identity",
)
ALLOWED_CONTROLLER_PHASES = {
    "training_pair",
    "training_cofitok",
    "training_dense_identity",
    *(f"evaluating_{arm}" for arm in EVALUATION_ARMS),
    "result_replay",
    "completed",
    "failed",
}
NON_AUTHORIZING_BOUNDARY = {
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "gpu_execution_allowed": False,
    "continuation_beyond_50000_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
EXPECTED_RESULT_BOUNDARY = {
    **NON_AUTHORIZING_BOUNDARY,
    "new_gate_required_for_any_followup": True,
}
REQUIRED_METRIC_FIELDS = (
    "total",
    "epsilon",
    "epsilon_unweighted",
    "min_snr_weight_mean",
    "learning_rate",
    "grad_norm",
)
CHECKPOINT_PATTERN = re.compile(r"checkpoint_step_(\d{8})\.pt$")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _write_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _run(command: list[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def git_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path)

    def git(*args: str) -> str:
        result = _run(["git", *args], cwd=root)
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(
            git("status", "--porcelain=v1", "--untracked-files=all")
        ),
    }


def inspect_metrics(path: str | Path, *, now: float) -> dict[str, Any]:
    metrics_path = Path(path)
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    issues.append(
                        f"metrics JSON is invalid at line {line_number}: {error.msg}"
                    )
                    continue
                if not isinstance(value, dict):
                    issues.append(f"metrics row is not an object at line {line_number}")
                    continue
                rows.append(value)
    previous_step = 0
    downweight_observed = False
    for index, row in enumerate(rows, start=1):
        step = row.get("step")
        samples_seen = row.get("samples_seen")
        if not isinstance(step, int) or isinstance(step, bool) or step <= previous_step:
            issues.append(f"metrics step is not strictly increasing at row {index}")
            continue
        previous_step = step
        if step > PILOT_STEP:
            issues.append(f"metrics step {step} exceeds the authorized 50K stop")
        if samples_seen != step * EFFECTIVE_BATCH_SIZE:
            issues.append(f"metrics samples_seen differs at step {step}")
        for field in REQUIRED_METRIC_FIELDS:
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                issues.append(f"metrics {field} is not finite at step {step}")
        weight = row.get("min_snr_weight_mean")
        if isinstance(weight, (int, float)) and not isinstance(weight, bool):
            if not 0.0 < float(weight) <= 1.0:
                issues.append(f"metrics Min-SNR weight is out of range at step {step}")
            downweight_observed = downweight_observed or float(weight) < 0.999999
        for field, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                issues.append(f"metrics {field} is non-finite at step {step}")
    if rows and int(rows[0].get("step", -1)) != 1:
        issues.append("metrics do not begin at fresh-initialization step 1")
    if rows and not downweight_observed:
        issues.append("metrics have not applied Min-SNR downweighting")
    mtime = metrics_path.stat().st_mtime if metrics_path.is_file() else None
    return {
        "path": metrics_path.resolve().as_posix(),
        "exists": metrics_path.is_file(),
        "row_count": len(rows),
        "first_step": rows[0].get("step") if rows else 0,
        "last_step": rows[-1].get("step") if rows else 0,
        "last_metric": rows[-1] if rows else None,
        "activity_age_seconds": now - mtime if mtime is not None else None,
        "strictly_increasing": not any("strictly increasing" in item for item in issues),
        "samples_seen_binding_verified": not any("samples_seen" in item for item in issues),
        "finite_min_snr_fields_verified": not any(
            "not finite" in item or "out of range" in item or "non-finite" in item
            for item in issues
        ),
        "issues": issues,
    }


def inspect_run_manifest(
    run_dir: str | Path,
    *,
    method: str,
    expected_revision: str,
    expected_branch: str,
    metrics_exist: bool,
) -> dict[str, Any]:
    path = Path(run_dir) / "run_manifest.json"
    issues: list[str] = []
    if not path.is_file():
        if metrics_exist:
            issues.append("run manifest is missing after metrics were emitted")
        return {"path": path.resolve().as_posix(), "status": "missing", "issues": issues}
    try:
        manifest = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "path": path.resolve().as_posix(),
            "status": "invalid",
            "issues": [f"run manifest is unreadable: {error}"],
        }
    config = manifest.get("config")
    git = manifest.get("git")
    if not isinstance(config, dict):
        config = {}
        issues.append("run manifest config is malformed")
    if not isinstance(git, dict):
        git = {}
        issues.append("run manifest Git identity is malformed")
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    loss = config.get("loss") if isinstance(config.get("loss"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    diffusion = (
        config.get("diffusion") if isinstance(config.get("diffusion"), dict) else {}
    )
    optimization = (
        config.get("optimization")
        if isinstance(config.get("optimization"), dict)
        else {}
    )
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    expected = {
        "scheduler_horizon": runtime.get("steps") == SCHEDULER_HORIZON,
        "checkpoint_interval": runtime.get("checkpoint_interval") == CHECKPOINT_INTERVAL,
        "seed": runtime.get("seed") == 2027,
        "gamma": loss.get("min_snr_gamma") == 5.0,
        "prediction_target": diffusion.get("prediction_target") == "epsilon",
        "dataset": data.get("dataset") == "imagenet_256",
        "effective_batch": data.get("batch_size") == EFFECTIVE_BATCH_SIZE,
        "accumulation": optimization.get("gradient_accumulation_steps") == 1,
        "git_revision": git.get("revision") == expected_revision,
        "git_branch": git.get("branch") == expected_branch,
        "git_clean": git.get("dirty") is False,
        "runtime_environment": manifest.get("runtime_environment_sha256")
        == RUNTIME_ENVIRONMENT_SHA256,
        "dataset_identity": (
            manifest.get("dataset_provenance", {}).get("identity_sha256")
            == DATASET_IDENTITY_SHA256
            if isinstance(manifest.get("dataset_provenance"), dict)
            else False
        ),
    }
    if method == "cofitok":
        expected.update(
            {
                "method_synthesis": model.get("synthesis_mode") == "fixed_basis",
                "method_tokens": model.get("token_count") == 8,
                "method_feedback": model.get("predictor_use_feedback") is True,
            }
        )
    else:
        expected.update(
            {
                "method_synthesis": model.get("synthesis_mode") == "dense_identity",
                "method_tokens": model.get("token_count") == 1,
                "method_feedback": model.get("predictor_use_feedback") is False,
            }
        )
    issues.extend(f"run manifest {name} differs" for name, passed in expected.items() if not passed)
    return {
        "path": path.resolve().as_posix(),
        "status": "verified" if not issues else "invalid",
        "scheduler_horizon_steps": runtime.get("steps"),
        "physical_stop_step": PILOT_STEP,
        "checks": expected,
        "issues": issues,
    }


def inspect_checkpoint_metadata(
    run_dir: str | Path,
    *,
    last_step: int,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    root = Path(run_dir)
    issues: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    for path in root.glob("checkpoint_step_*.pt") if root.is_dir() else ():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step > PILOT_STEP:
            issues.append(f"checkpoint {path.name} exceeds the authorized 50K stop")
        row: dict[str, Any] = {
            "step": step,
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "verification": "metadata_only_no_payload_hash",
        }
        sidecar_path = path.with_name(path.name + ".integrity.json")
        if sidecar_path.is_file():
            try:
                sidecar = _read_json(sidecar_path)
                declared_sha = sidecar.get("checkpoint_sha256")
                checks = {
                    "checkpoint": sidecar.get("checkpoint") == path.name,
                    "bytes": sidecar.get("checkpoint_bytes") == path.stat().st_size,
                    "step": sidecar.get("step") == step,
                    "sha256": isinstance(declared_sha, str)
                    and SHA256_PATTERN.fullmatch(declared_sha) is not None,
                    "git_revision": sidecar.get("git_revision") == expected_revision,
                    "git_branch": sidecar.get("git_branch") == expected_branch,
                    "git_clean": sidecar.get("git_dirty") is False,
                    "dataset_identity": sidecar.get("dataset_identity_sha256")
                    == DATASET_IDENTITY_SHA256,
                    "runtime_environment": sidecar.get("runtime_environment_sha256")
                    == RUNTIME_ENVIRONMENT_SHA256,
                }
                row.update(
                    {
                        "integrity_status": (
                            "metadata_verified" if all(checks.values()) else "invalid"
                        ),
                        "declared_sha256": declared_sha,
                        "integrity_checks": checks,
                    }
                )
                issues.extend(
                    f"checkpoint {path.name} {name} metadata differs"
                    for name, passed in checks.items()
                    if not passed
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                row["integrity_status"] = "invalid"
                issues.append(f"checkpoint {path.name} sidecar is unreadable: {error}")
        else:
            row["integrity_status"] = "pending"
            if last_step >= step + CHECKPOINT_GRACE_STEPS:
                issues.append(f"checkpoint {path.name} sidecar is missing after grace")
        checkpoints.append(row)
    checkpoints.sort(key=lambda value: int(value["step"]))
    required_checkpoint = (
        ((last_step - CHECKPOINT_GRACE_STEPS) // CHECKPOINT_INTERVAL)
        * CHECKPOINT_INTERVAL
        if last_step > CHECKPOINT_GRACE_STEPS
        else 0
    )
    latest_step = int(checkpoints[-1]["step"]) if checkpoints else 0
    if required_checkpoint > latest_step:
        issues.append(
            f"checkpoint cadence is late: required>={required_checkpoint}, latest={latest_step}"
        )
    latest_path = root / "latest.json"
    latest: dict[str, Any] | None = None
    if latest_path.is_file():
        try:
            latest = _read_json(latest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"latest.json is unreadable: {error}")
    if latest is not None:
        pointer_step = latest.get("step")
        if not isinstance(pointer_step, int) or pointer_step > PILOT_STEP:
            issues.append("latest.json step is outside the pilot boundary")
        matching = next(
            (row for row in checkpoints if row["step"] == pointer_step), None
        )
        if matching is None:
            issues.append("latest.json does not name an observed checkpoint")
        elif (
            latest.get("checkpoint") != Path(str(matching["path"])).name
            or latest.get("checkpoint_bytes") != matching["bytes"]
            or latest.get("checkpoint_sha256") != matching.get("declared_sha256")
            or latest.get("git_revision") != expected_revision
            or latest.get("git_branch") != expected_branch
            or latest.get("git_dirty") is not False
        ):
            issues.append("latest.json metadata differs from its checkpoint sidecar")
    elif required_checkpoint > 0:
        issues.append("latest.json is missing after checkpoint grace")
    return {
        "checkpoints": checkpoints,
        "latest": latest,
        "required_checkpoint_step": required_checkpoint,
        "issues": issues,
    }


def inspect_training_report(
    run_dir: str | Path,
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    path = Path(run_dir) / "training_report.json"
    if not path.is_file():
        return {"exists": False, "verified": False, "issues": []}
    issues: list[str] = []
    try:
        report = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"exists": True, "verified": False, "issues": [str(error)]}
    git = report.get("git") if isinstance(report.get("git"), dict) else {}
    checks = {
        "completed_steps": report.get("completed_steps") == PILOT_STEP,
        "scheduler_horizon": report.get("target_steps") == SCHEDULER_HORIZON,
        "not_full_training_complete": report.get("training_complete") is False,
        "not_signal_stopped": report.get("stop_requested") is False
        and report.get("stop_signal") is None,
        "git_revision": git.get("revision") == expected_revision,
        "git_branch": git.get("branch") == expected_branch,
        "git_clean": git.get("dirty") is False,
        "runtime_environment": report.get("runtime_environment_sha256")
        == RUNTIME_ENVIRONMENT_SHA256,
        "dataset_identity": (
            report.get("dataset_provenance", {}).get("identity_sha256")
            == DATASET_IDENTITY_SHA256
            if isinstance(report.get("dataset_provenance"), dict)
            else False
        ),
    }
    issues.extend(f"training report {name} differs" for name, passed in checks.items() if not passed)
    return {
        "exists": True,
        "verified": all(checks.values()),
        "checks": checks,
        "issues": issues,
    }


def inspect_training_audit(output_root: str | Path, *, method: str) -> dict[str, Any]:
    path = (
        Path(output_root)
        / "reports/training_audits"
        / f"{method}_50k_physical_audit.json"
    )
    if not path.is_file():
        return {"exists": False, "verified": False, "issues": []}
    try:
        audit = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"exists": True, "verified": False, "issues": [str(error)]}
    training = audit.get("training") if isinstance(audit.get("training"), dict) else {}
    checkpoint = (
        audit.get("checkpoint") if isinstance(audit.get("checkpoint"), dict) else {}
    )
    checks = {
        "status": audit.get("status") == "pass",
        "method": audit.get("method") == method,
        "completed_steps": training.get("completed_steps") == PILOT_STEP,
        "scheduler_horizon": training.get("target_steps") == SCHEDULER_HORIZON,
        "samples_seen": training.get("samples_seen")
        == PILOT_STEP * EFFECTIVE_BATCH_SIZE,
        "metrics": training.get("strictly_increasing_metrics") is True
        and training.get("samples_seen_binding_verified") is True
        and training.get("min_snr_metrics_verified") is True,
        "physical_sha256": checkpoint.get("physical_sha256_verified") is True,
        "latest_binding": checkpoint.get("latest_exact_binding") is True,
    }
    issues = [f"training audit {name} differs" for name, passed in checks.items() if not passed]
    return {
        "exists": True,
        "verified": all(checks.values()),
        "sha256": _sha256(path),
        "checks": checks,
        "issues": issues,
    }


def inspect_arm(
    output_root: str | Path,
    *,
    method: str,
    expected_revision: str,
    expected_branch: str,
    now: float,
) -> dict[str, Any]:
    run_dir = Path(output_root) / RUN_NAMES[method]
    metrics = inspect_metrics(run_dir / "train_metrics.jsonl", now=now)
    manifest = inspect_run_manifest(
        run_dir,
        method=method,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        metrics_exist=bool(metrics["exists"]),
    )
    checkpoints = inspect_checkpoint_metadata(
        run_dir,
        last_step=int(metrics["last_step"]),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    report = inspect_training_report(
        run_dir,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    audit = inspect_training_audit(output_root, method=method)
    issues = [
        *metrics["issues"],
        *manifest["issues"],
        *checkpoints["issues"],
        *report["issues"],
        *audit["issues"],
    ]
    return {
        "method": method,
        "run_dir": run_dir.resolve().as_posix(),
        "metrics": metrics,
        "run_manifest": manifest,
        "checkpoint_metadata": checkpoints,
        "training_report": report,
        "physical_audit": audit,
        "at_physical_stop": int(metrics["last_step"]) == PILOT_STEP,
        "audited": bool(audit["verified"]),
        "issues": issues,
    }


def scan_processes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [
                item.decode("utf-8", errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            if not argv:
                continue
            raw_stat = (entry / "stat").read_text(encoding="utf-8")
            stat = raw_stat.rsplit(")", 1)[1].split()
            environment = {}
            for item in (entry / "environ").read_bytes().split(b"\0"):
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                decoded_key = key.decode("utf-8", errors="replace")
                if decoded_key in {
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "PYTHONPATH",
                }:
                    environment[decoded_key] = value.decode(
                        "utf-8", errors="replace"
                    )
            records.append(
                {
                    "pid": int(entry.name),
                    "ppid": int(stat[1]),
                    "state": stat[0],
                    "nice": int(stat[16]),
                    "start_ticks": int(stat[19]),
                    "cwd": os.readlink(entry / "cwd"),
                    "argv": argv,
                    "cmdline": " ".join(argv),
                    "environment": environment,
                }
            )
        except (FileNotFoundError, IndexError, OSError, PermissionError, ValueError):
            continue
    return records


def descendant_pids(processes: list[dict[str, Any]], root_pid: int) -> set[int]:
    descendants: set[int] = set()
    while True:
        added = {
            int(record["pid"])
            for record in processes
            if int(record["ppid"]) == root_pid or int(record["ppid"]) in descendants
        } - descendants
        if not added:
            return descendants
        descendants.update(added)


def gpu_compute_processes() -> tuple[list[dict[str, Any]], list[str]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory,process_name,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return [], ["nvidia-smi compute-process query failed"]
    rows = []
    issues = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 3)]
        if len(values) != 4:
            issues.append("nvidia-smi compute-process row is malformed")
            continue
        try:
            rows.append(
                {
                    "pid": int(values[0]),
                    "used_memory_mib": int(values[1]),
                    "process_name": values[2],
                    "gpu_uuid": values[3],
                }
            )
        except ValueError:
            issues.append("nvidia-smi compute-process values are malformed")
    return rows, issues


def gpu_inventory() -> tuple[list[dict[str, Any]], list[str]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return [], ["nvidia-smi GPU query failed"]
    rows = []
    issues = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 6)]
        if len(values) != 7:
            issues.append("nvidia-smi GPU row is malformed")
            continue
        try:
            rows.append(
                {
                    "index": int(values[0]),
                    "uuid": values[1],
                    "name": values[2],
                    "utilization_percent": int(values[3]),
                    "memory_used_mib": int(values[4]),
                    "memory_total_mib": int(values[5]),
                    "temperature_c": int(values[6]),
                }
            )
        except ValueError:
            issues.append("nvidia-smi GPU values are malformed")
    return rows, issues


def fdinfo_declares_write_flock(value: str, *, inode: int) -> bool:
    for line in value.splitlines():
        if not line.startswith("lock:"):
            continue
        fields = line.split()
        if "FLOCK" not in fields or "WRITE" not in fields:
            continue
        try:
            record_inode = int(fields[-3].rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if record_inode == inode:
            return True
    return False


def execution_lock_held_by_controller(path: str | Path, controller_pid: int) -> bool:
    target = Path(path)
    if not target.is_file():
        return False
    resolved_target = target.resolve()
    inode = target.stat().st_ino
    fd_root = Path("/proc") / str(controller_pid) / "fd"
    try:
        descriptors = list(fd_root.iterdir())
    except OSError:
        return False
    for descriptor in descriptors:
        try:
            if descriptor.resolve() != resolved_target:
                continue
            fdinfo = (
                Path("/proc")
                / str(controller_pid)
                / "fdinfo"
                / descriptor.name
            ).read_text(encoding="utf-8")
        except OSError:
            continue
        if fdinfo_declares_write_flock(fdinfo, inode=inode):
            return True
    return False


def evaluate_process_health(
    *,
    processes: list[dict[str, Any]],
    gpu_processes: list[dict[str, Any]],
    controller_pid: int,
    controller_start_ticks: int,
    controller_runbook: str,
    controller_phase: str,
    arms: dict[str, dict[str, Any]],
    phase_elapsed_seconds: float,
    startup_grace_seconds: float,
    stall_seconds: float,
    controller_completed: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    stall_issues: list[str] = []
    controllers = [
        record
        for record in processes
        if record["argv"] == ["bash", controller_runbook]
    ]
    controller = next(
        (record for record in controllers if record["pid"] == controller_pid), None
    )
    expected_controller_counts = {0, 1} if controller_completed else {1}
    if len(controllers) not in expected_controller_counts:
        issues.append(f"expected exactly one pilot controller, found {len(controllers)}")
    if controller is None:
        if not controller_completed:
            issues.append("bound pilot controller process is absent")
        descendants: set[int] = set()
    else:
        if controller["ppid"] != 1:
            issues.append("pilot controller parent PID is not 1")
        if controller["start_ticks"] != controller_start_ticks:
            issues.append("pilot controller start ticks differ")
        descendants = descendant_pids(processes, controller_pid)
    pilot_trainers = [
        record
        for record in processes
        if "scripts/train_generation.py" in record["cmdline"]
        and "min_snr_gamma5_matched_50k_pilot_v1" in record["cmdline"]
    ]
    outside_trainers = [
        record for record in pilot_trainers if record["pid"] not in descendants
    ]
    if outside_trainers:
        issues.append("pilot trainer exists outside the bound controller process tree")
    direct_trainers = [
        record for record in pilot_trainers if record["ppid"] == controller_pid
    ]
    if len(direct_trainers) > 1:
        issues.append(f"multiple direct pilot trainers found: {len(direct_trainers)}")
    active_method = None
    if controller_phase == "training_cofitok":
        active_method = "cofitok"
    elif controller_phase == "training_dense_identity":
        active_method = "dense_identity"
    if active_method is not None:
        arm = arms[active_method]
        if len(direct_trainers) == 1:
            trainer = direct_trainers[0]
            expected_config = CONFIG_NAMES[active_method]
            expected_run = f"/{RUN_NAMES[active_method]}"
            if expected_config not in trainer["cmdline"] or expected_run not in trainer["cmdline"]:
                issues.append("direct trainer command differs from the controller phase")
            age = arm["metrics"].get("activity_age_seconds")
            if age is not None and float(age) > stall_seconds:
                stall_issues.append(
                    f"{active_method} metrics have not advanced for {float(age):.1f} seconds"
                )
            elif age is None and phase_elapsed_seconds > stall_seconds:
                stall_issues.append(
                    f"{active_method} has not emitted metrics for "
                    f"{phase_elapsed_seconds:.1f} seconds"
                )
        elif (
            int(arm["metrics"].get("last_step", 0)) < PILOT_STEP
            and phase_elapsed_seconds > startup_grace_seconds
        ):
            issues.append(f"{active_method} direct trainer is absent before 50K")
    descendant_gpu = [row for row in gpu_processes if row["pid"] in descendants]
    external_gpu = [row for row in gpu_processes if row["pid"] not in descendants]
    if external_gpu:
        issues.append("GPU compute process exists outside the bound pilot controller tree")
    if active_method is not None and direct_trainers and descendant_gpu:
        direct_pid = int(direct_trainers[0]["pid"])
        if any(int(row["pid"]) != direct_pid for row in descendant_gpu):
            issues.append("training GPU owner is not the direct trainer")
    return {
        "controller": controller,
        "controller_count": len(controllers),
        "descendant_pids": sorted(descendants),
        "direct_trainers": direct_trainers,
        "pilot_training_process_count": len(pilot_trainers),
        "gpu_descendants": descendant_gpu,
        "gpu_external": external_gpu,
        "issues": issues,
        "stall_issues": stall_issues,
    }


def inspect_evaluations(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root) / "evaluations"
    arms = {}
    for arm in EVALUATION_ARMS:
        arm_root = root / arm
        expected = {
            "sampling_preflight": arm_root / "sampling_preflight.json",
            "sampling_report": arm_root / "samples/sampling_report.json",
            "generation_metrics": (
                arm_root / "samples/metrics/generation_metrics_report.json"
            ),
            "class_fidelity": (
                arm_root / "samples/class_fidelity/class_fidelity_report.json"
            ),
            "checkpoint_evaluation": (
                arm_root / "checkpoint_eval/checkpoint_evaluation_report.json"
            ),
        }
        progress_path = arm_root / "samples/sampling_progress.json"
        progress = None
        if progress_path.is_file():
            try:
                progress = _read_json(progress_path)
            except (OSError, ValueError, json.JSONDecodeError):
                progress = {"status": "unreadable"}
        arms[arm] = {
            "exists": arm_root.is_dir(),
            "artifacts": {name: path.is_file() for name, path in expected.items()},
            "complete": all(path.is_file() for path in expected.values()),
            "sampling_progress": progress,
        }
    return {"arms": arms, "completed_arm_count": sum(row["complete"] for row in arms.values())}


def inspect_result(output_root: str | Path) -> dict[str, Any]:
    path = Path(output_root) / "reports/min_snr_pilot_result.json"
    if not path.is_file():
        return {"exists": False, "verified_non_authorizing": False, "issues": []}
    try:
        result = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"exists": True, "verified_non_authorizing": False, "issues": [str(error)]}
    boundary = result.get("authorization_boundary")
    next_stage = result.get("recommended_next_stage")
    checks = {
        "status": result.get("status") == "completed",
        "screening_only": result.get("scientific_status") == "screening_only",
        "terminal_hold": result.get("terminal_status") == "hold",
        "advantage_unproven": result.get("generation_advantage_proven") is False,
        "authorization_boundary": boundary == EXPECTED_RESULT_BOUNDARY,
        "continuation_blocked": isinstance(next_stage, dict)
        and next_stage.get("execution_ready") is False
        and next_stage.get("gpu_execution_allowed") is False
        and next_stage.get("continuation_beyond_50000_allowed") is False,
    }
    issues = [f"pilot result {name} differs" for name, passed in checks.items() if not passed]
    return {
        "exists": True,
        "verified_non_authorizing": all(checks.values()),
        "sha256": _sha256(path),
        "selection_status": result.get("selection_status"),
        "checks": checks,
        "issues": issues,
    }


def classify_status(
    *,
    controller_status: str,
    controller_phase: str,
    issues: list[str],
    stall_issues: list[str],
    result_verified: bool,
) -> tuple[str, str]:
    if issues:
        return "failed", "health_failure"
    if stall_issues:
        return "stalled", controller_phase
    if controller_status == "failed":
        return "failed", "controller_failed"
    if controller_status == "completed":
        return (
            ("pass", "completed")
            if result_verified
            else ("failed", "completion_evidence_missing")
        )
    return "running", controller_phase


def _hash_binding(path: str | Path, expected: str, *, label: str) -> dict[str, Any]:
    source = Path(path)
    actual = _sha256(source) if source.is_file() else None
    return {
        "label": label,
        "path": source.resolve().as_posix(),
        "exists": source.is_file(),
        "bytes": source.stat().st_size if source.is_file() else None,
        "sha256": actual,
        "expected_sha256": expected,
        "verified": actual == expected,
    }


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    now = time.time()
    source_bindings = {
        "preparation": _hash_binding(
            args.preparation,
            args.expected_preparation_sha256,
            label="preparation",
        ),
        "execution_gate": _hash_binding(
            args.execution_gate,
            args.expected_execution_gate_sha256,
            label="execution gate",
        ),
        "controller_launch_receipt": _hash_binding(
            args.controller_launch_receipt,
            args.expected_controller_launch_receipt_sha256,
            label="controller launch receipt",
        ),
        "controller_process_tree_audit": _hash_binding(
            args.controller_process_tree_audit,
            args.expected_controller_process_tree_audit_sha256,
            label="controller process-tree audit",
        ),
        "controller_runbook": _hash_binding(
            args.controller_runbook,
            args.expected_controller_runbook_sha256,
            label="controller runbook",
        ),
        "monitor_source": _hash_binding(
            Path(__file__), args.expected_self_sha256, label="monitor source"
        ),
    }
    issues = [
        f"{name} hash differs"
        for name, binding in source_bindings.items()
        if not binding["verified"]
    ]
    training_git = git_identity(args.training_checkout)
    monitor_git = git_identity(args.monitor_checkout)
    expected_training_git = {
        "revision": args.expected_training_revision,
        "tree": args.expected_training_tree,
        "branch": args.expected_training_branch,
        "tracked_dirty": False,
    }
    expected_monitor_git = {
        "revision": args.expected_monitor_revision,
        "tree": args.expected_monitor_tree,
        "branch": args.expected_monitor_branch,
        "tracked_dirty": False,
    }
    if training_git != expected_training_git:
        issues.append("training checkout Git identity differs")
    if monitor_git != expected_monitor_git:
        issues.append("monitor checkout Git identity differs")
    try:
        controller_status = _read_json(args.controller_status)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        controller_status = {"status": "unreadable", "phase": "unknown"}
        issues.append(f"controller status is unreadable: {error}")
    phase = str(controller_status.get("phase", "unknown"))
    try:
        phase_updated_at = datetime.fromisoformat(
            str(controller_status["updated_at"])
        ).timestamp()
        phase_elapsed_seconds = max(0.0, now - phase_updated_at)
    except (KeyError, TypeError, ValueError):
        phase_elapsed_seconds = float("inf")
        issues.append("controller status updated_at is invalid")
    if controller_status.get("status") not in {"running", "completed", "failed"}:
        issues.append("controller status value is invalid")
    if phase not in ALLOWED_CONTROLLER_PHASES:
        issues.append("controller phase is invalid")
    if controller_status.get("pid") != args.expected_controller_pid:
        issues.append("controller status PID differs")
    if controller_status.get("generation_advantage_proven") is not False:
        issues.append("controller status claims generation advantage")
    for field in (
        "continuation_beyond_50000_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "release_allowed",
        "process_signals_allowed",
    ):
        if controller_status.get(field) is not False:
            issues.append(f"controller status {field} is not false")
    arms = {
        method: inspect_arm(
            args.output_root,
            method=method,
            expected_revision=args.expected_training_revision,
            expected_branch=args.expected_training_branch,
            now=now,
        )
        for method in METHODS
    }
    for method, arm in arms.items():
        issues.extend(f"{method}: {issue}" for issue in arm["issues"])
    processes = scan_processes()
    ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    uptime_seconds = float(Path("/proc/uptime").read_text().split()[0])
    controller_elapsed_seconds = max(
        0.0,
        uptime_seconds - args.expected_controller_start_ticks / ticks_per_second,
    )
    compute, gpu_issues = gpu_compute_processes()
    inventory, inventory_issues = gpu_inventory()
    issues.extend(gpu_issues)
    issues.extend(inventory_issues)
    process_health = evaluate_process_health(
        processes=processes,
        gpu_processes=compute,
        controller_pid=args.expected_controller_pid,
        controller_start_ticks=args.expected_controller_start_ticks,
        controller_runbook=Path(args.controller_runbook).resolve().as_posix(),
        controller_phase=phase,
        arms=arms,
        phase_elapsed_seconds=phase_elapsed_seconds,
        startup_grace_seconds=args.startup_grace_seconds,
        stall_seconds=args.stall_seconds,
        controller_completed=controller_status.get("status") == "completed",
    )
    issues.extend(process_health["issues"])
    execution_lock_held = execution_lock_held_by_controller(
        args.execution_lock, args.expected_controller_pid
    )
    if (
        not execution_lock_held
        and controller_status.get("status") != "completed"
    ):
        issues.append("execution flock is not held by the bound controller")
    usage = shutil.disk_usage(args.output_root)
    disk = {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "minimum_free_bytes": args.minimum_free_bytes,
    }
    if usage.free < args.minimum_free_bytes:
        issues.append("free disk is below the pilot monitor threshold")
    evaluations = inspect_evaluations(args.output_root)
    result = inspect_result(args.output_root)
    issues.extend(result["issues"])
    monitor_ionice = _run(["ionice", "-p", str(os.getpid())])
    monitor_runtime = {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "ionice": monitor_ionice.stdout.strip(),
    }
    if args.require_parent_pid_one and monitor_runtime["parent_pid"] != 1:
        issues.append("monitor parent PID is not 1")
    if monitor_runtime["cuda_visible_devices"] not in {"", "-1"}:
        issues.append("monitor CUDA visibility is not disabled")
    if monitor_runtime["omp_num_threads"] != "1" or monitor_runtime["mkl_num_threads"] != "1":
        issues.append("monitor CPU thread limits differ")
    if monitor_runtime["nice"] < 10:
        issues.append("monitor nice priority is too high")
    if monitor_ionice.returncode != 0 or monitor_runtime["ionice"] != "idle":
        issues.append("monitor ionice class is not idle")
    status, status_phase = classify_status(
        controller_status=str(controller_status.get("status", "invalid")),
        controller_phase=phase,
        issues=issues,
        stall_issues=process_health["stall_issues"],
        result_verified=bool(result["verified_non_authorizing"]),
    )
    relevant_processes = [
        record
        for record in processes
        if record["pid"] == args.expected_controller_pid
        or record["pid"] in set(process_health["descendant_pids"])
        or str(Path(__file__).resolve()) in record["cmdline"]
    ]
    return {
        "schema_version": 1,
        "role": "generation_matched_min_snr_50k_pilot_health_monitor",
        "status": status,
        "phase": status_phase,
        "issues": issues + process_health["stall_issues"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "monitor_process": monitor_runtime,
        "source_bindings": source_bindings,
        "git": {"training_checkout": training_git, "monitor_checkout": monitor_git},
        "controller_status": controller_status,
        "controller_elapsed_seconds": controller_elapsed_seconds,
        "execution_lock": {
            "path": Path(args.execution_lock).resolve().as_posix(),
            "held_by_controller": execution_lock_held,
            "release_allowed_only_after_controller_completion": True,
        },
        "arms": arms,
        "evaluations": evaluations,
        "result": result,
        "process_health": process_health,
        "processes": relevant_processes,
        "gpu": {"inventory": inventory, "compute_processes": compute},
        "disk": disk,
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "monitor_is_read_only_except_status_and_lock": True,
        "monitor_is_permanently_non_authorizing": True,
        "authorization_boundary": dict(NON_AUTHORIZING_BOUNDARY),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only, non-authorizing health monitor for the matched Min-SNR pilot."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--status-output", required=True)
    parser.add_argument("--monitor-lock", required=True)
    parser.add_argument("--training-checkout", required=True)
    parser.add_argument("--monitor-checkout", required=True)
    parser.add_argument("--controller-status", required=True)
    parser.add_argument("--controller-runbook", required=True)
    parser.add_argument("--controller-launch-receipt", required=True)
    parser.add_argument("--controller-process-tree-audit", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--execution-gate", required=True)
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-monitor-revision", required=True)
    parser.add_argument("--expected-monitor-tree", required=True)
    parser.add_argument("--expected-monitor-branch", required=True)
    parser.add_argument("--expected-controller-pid", type=int, required=True)
    parser.add_argument("--expected-controller-start-ticks", type=int, required=True)
    parser.add_argument("--expected-self-sha256", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--expected-execution-gate-sha256", required=True)
    parser.add_argument("--expected-controller-runbook-sha256", required=True)
    parser.add_argument("--expected-controller-launch-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-controller-process-tree-audit-sha256", required=True
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stall-seconds", type=float, default=1_800.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-free-bytes", type=int, default=128_849_018_880)
    parser.add_argument("--require-parent-pid-one", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if min(
        args.expected_controller_pid,
        args.expected_controller_start_ticks,
        args.poll_seconds,
        args.stall_seconds,
        args.startup_grace_seconds,
        args.minimum_free_bytes,
    ) <= 0:
        raise ValueError("pilot monitor thresholds and process identity must be positive")
    for value in (
        args.expected_self_sha256,
        args.expected_preparation_sha256,
        args.expected_execution_gate_sha256,
        args.expected_controller_runbook_sha256,
        args.expected_controller_launch_receipt_sha256,
        args.expected_controller_process_tree_audit_sha256,
    ):
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("pilot monitor expected SHA256 is malformed")


def _acquire_monitor_lock(path: str | Path):
    import fcntl

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("refusing duplicate Min-SNR pilot health monitor")
    return handle


def main() -> int:
    args = parse_args()
    _validate_args(args)
    lock_handle = _acquire_monitor_lock(args.monitor_lock)
    try:
        while True:
            try:
                report = build_snapshot(args)
            except Exception as error:
                report = {
                    "schema_version": 1,
                    "role": "generation_matched_min_snr_50k_pilot_health_monitor",
                    "status": "failed",
                    "phase": "monitor_exception",
                    "issues": [f"{type(error).__name__}: {error}"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "hostname": socket.gethostname(),
                    "terminal_status": "hold",
                    "generation_advantage_proven": False,
                    "monitor_is_read_only_except_status_and_lock": True,
                    "monitor_is_permanently_non_authorizing": True,
                    "authorization_boundary": dict(NON_AUTHORIZING_BOUNDARY),
                }
            _write_atomic(args.status_output, report)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "phase": report["phase"],
                        "issues": report["issues"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.once or report["status"] in {"pass", "failed", "stalled"}:
                return 0 if report["status"] in {"running", "pass"} else 1
            time.sleep(args.poll_seconds)
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
