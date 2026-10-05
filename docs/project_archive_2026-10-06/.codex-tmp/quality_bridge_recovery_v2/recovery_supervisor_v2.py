from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


BASE = Path("/tmp/cofitok-quality-bridge-execution-cf0e5fa")
PROJECT = BASE / "CoFiTok-internal"
FORMAL_PROJECT = Path("/root/autodl-tmp/CoFiTok/CoFiTok-internal")
PYTHON = Path("/root/autodl-tmp/conda/envs/pf-vlm/bin/python")
CHECKPOINT_ROOT = Path("/root/autodl-tmp/CoFiTok/checkpoints/generation")
OUTPUT_ROOT = CHECKPOINT_ROOT / "stability_full_data_100k_base128_quality_bridge_v1"
REPORT_ROOT = OUTPUT_ROOT / "reports"
RUNBOOK = PROJECT / "artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh"
PREPARATION = REPORT_ROOT / "preparation.json"
APPROVAL = BASE / "execution_approval.json"
STANDING_AUTHORIZATION = BASE / "standing_authorization.json"
PRELAUNCH_AUDIT = REPORT_ROOT / "prelaunch_contract_audit.json"
EXECUTION_STATUS = REPORT_ROOT / "execution_status.json"
PAIR_MONITOR = OUTPUT_ROOT / "pair_monitor.json"
LAUNCH_RECEIPT = REPORT_ROOT / "launch_receipt.json"
QUALITY_RESULT = REPORT_ROOT / "quality_bridge_result.json"
SUPERVISOR_STATUS = BASE / "recovery_supervisor_v2_status.json"
SUPERVISOR_LOG = BASE / "recovery_supervisor_v2.log"
SUPERVISOR_LOCK = BASE / "recovery_supervisor_v2.lock"
SUPERVISOR_PID = BASE / "recovery_supervisor_v2.pid"
RECOVERY_CONTROLLER_RECEIPT = BASE / "recovery_controller_v2_launch.json"
DIRECT_CONTROLLER_LOG = BASE / "exact_resume_20260817.log"
IDLE_WAITER_PID = BASE / "idle_waiter.pid"

REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
TREE = "6cef27723196fd363379bca2e7b85b1678ebd777"
BRANCH = "scale/generation-stability-quality-bridge-100k"
FORMAL_REVISION = "1ebcc15210e63a776a2ba448481cbd8bb94a4066"
FORMAL_BRANCH = "scale/generative-system"
PREPARATION_SHA256 = "7398d9a6f096ea9c178295c9016bb56fd38e28dff30f26662ae4225aded208ea"
APPROVAL_SHA256 = "e9da52a4e7ff1b4700b70aadaa8703ee40fb1a9862e847dcfb6e75933d295a4b"
STANDING_AUTHORIZATION_SHA256 = "5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df"
PRELAUNCH_AUDIT_SHA256 = "00dd284187a060be945e47d8ade4d1b3e1d51cbbfdd030413cf66232ffd979d4"
RUNBOOK_SHA256 = "f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73"
FROZEN_GATE_SHA256 = "2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90"
REQUIRED_FREE_BYTES = 103_826_920_100

POLL_SECONDS = 60
REQUIRED_IDLE_POLLS = 5
MAX_RECOVERY_ATTEMPTS = 3
BASE_RETRY_SECONDS = 300
MAX_RETRY_SECONDS = 1800
ORPHAN_GRACE_SECONDS = 180
UNCONDITIONAL_RETRYABLE_EXIT_CODES = {-15, -9, 9, 12, 15, 137, 143}
WATCHDOG_RETRYABLE_REASONS = {
    87: "monitor_did_not_publish_fresh_status",
    88: "monitor_status_stopped_updating",
    89: "monitor_process_missing",
}
RETRYABLE_EXIT_CODES = (
    UNCONDITIONAL_RETRYABLE_EXIT_CODES | set(WATCHDOG_RETRYABLE_REASONS) | {1}
)
PIN_MEMORY_FAILURE_MARKERS = (
    "RuntimeError: Pin memory thread exited unexpectedly",
    "multiprocessing/resource_sharer.py",
    "FileNotFoundError: [Errno 2] No such file or directory",
)
EXPECTED_SUPERVISOR_SHA256 = os.environ.get(
    "COFITOK_QUALITY_BRIDGE_RECOVERY_SUPERVISOR_V2_SHA256", ""
)


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    command: str
    cwd: str | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_identity(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required file is absent: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"file identity differs: {path}: expected {expected_sha256}, observed {observed}"
        )
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": observed}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def git_output(project: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=project,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def require_static_bindings() -> dict[str, Any]:
    if Path.cwd().resolve() != PROJECT.resolve():
        raise RuntimeError(
            f"recovery supervisor cwd differs: expected {PROJECT}, observed {Path.cwd()}"
        )
    supervisor_path = Path(__file__).resolve()
    supervisor_sha256 = sha256_file(supervisor_path)
    if not EXPECTED_SUPERVISOR_SHA256:
        raise RuntimeError("expected recovery supervisor SHA256 is not set")
    if supervisor_sha256 != EXPECTED_SUPERVISOR_SHA256:
        raise RuntimeError(
            "recovery supervisor identity differs: "
            f"expected {EXPECTED_SUPERVISOR_SHA256}, observed {supervisor_sha256}"
        )
    identities = {
        "recovery_supervisor": {
            "path": supervisor_path.as_posix(),
            "bytes": supervisor_path.stat().st_size,
            "sha256": supervisor_sha256,
        },
        "preparation": require_identity(PREPARATION, PREPARATION_SHA256),
        "approval": require_identity(APPROVAL, APPROVAL_SHA256),
        "standing_authorization": require_identity(
            STANDING_AUTHORIZATION, STANDING_AUTHORIZATION_SHA256
        ),
        "prelaunch_audit": require_identity(PRELAUNCH_AUDIT, PRELAUNCH_AUDIT_SHA256),
        "execution_runbook": require_identity(RUNBOOK, RUNBOOK_SHA256),
    }
    audit = read_json(PRELAUNCH_AUDIT)
    if (
        audit is None
        or audit.get("status") != "pass_waiting_for_gpu_idle"
        or audit.get("issues") != []
        or audit.get("authorization", {}).get("quality_bridge_execution_allowed")
        is not True
        or audit.get("authorization", {}).get("full_300k_launch_allowed") is not False
        or audit.get("authorization", {}).get("release_authorization_allowed") is not False
        or audit.get("recovery_contract", {}).get("passed_test_count") != 142
    ):
        raise RuntimeError("prelaunch contract audit is not a bounded pass")
    if str(git_output(PROJECT, "rev-parse", "HEAD")).strip() != REVISION:
        raise RuntimeError("execution checkout revision changed")
    if str(git_output(PROJECT, "rev-parse", "HEAD^{tree}")).strip() != TREE:
        raise RuntimeError("execution checkout tree changed")
    if str(git_output(PROJECT, "branch", "--show-current")).strip() != BRANCH:
        raise RuntimeError("execution checkout branch changed")
    if bytes(git_output(PROJECT, "status", "--porcelain=v1", binary=True)):
        raise RuntimeError("execution checkout became dirty")
    formal_porcelain = bytes(
        git_output(FORMAL_PROJECT, "status", "--porcelain=v1", binary=True)
    )
    formal_tracked_porcelain = bytes(
        git_output(
            FORMAL_PROJECT,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            binary=True,
        )
    )
    if str(git_output(FORMAL_PROJECT, "rev-parse", "HEAD")).strip() != FORMAL_REVISION:
        raise RuntimeError("formal checkout revision changed")
    if (
        str(git_output(FORMAL_PROJECT, "branch", "--show-current")).strip()
        != FORMAL_BRANCH
    ):
        raise RuntimeError("formal checkout branch changed")
    if formal_tracked_porcelain:
        raise RuntimeError("formal checkout has tracked changes")
    identities["formal_checkout"] = {
        "path": FORMAL_PROJECT.as_posix(),
        "revision": FORMAL_REVISION,
        "branch": FORMAL_BRANCH,
        "tracked_dirty": False,
        "porcelain_count": len(formal_porcelain.splitlines()),
        "porcelain_sha256": hashlib.sha256(formal_porcelain).hexdigest(),
    }
    if not (Path("/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/train").is_dir()):
        raise RuntimeError("ImageNet-256 train directory is absent")
    if not (Path("/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val").is_dir()):
        raise RuntimeError("ImageNet-256 validation directory is absent")
    current_free = shutil.disk_usage(OUTPUT_ROOT).free
    if current_free < REQUIRED_FREE_BYTES:
        raise RuntimeError(
            f"insufficient storage: required {REQUIRED_FREE_BYTES}, observed {current_free}"
        )
    identities["storage"] = {
        "required_free_bytes": REQUIRED_FREE_BYTES,
        "current_free_bytes": current_free,
        "headroom_bytes": current_free - REQUIRED_FREE_BYTES,
    }
    return identities


def process_records() -> list[ProcessRecord]:
    records: list[ProcessRecord] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
            stat = (entry / "stat").read_text(encoding="ascii").split()
            ppid = int(stat[3])
            cwd = os.readlink(entry / "cwd")
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, IndexError):
            continue
        if command:
            records.append(ProcessRecord(pid=pid, ppid=ppid, command=command, cwd=cwd))
    return records


def classify_processes(records: Iterable[ProcessRecord]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "waiter": [],
        "controller": [],
        "trainer": [],
        "watchdog": [],
        "evaluation": [],
        "monitor": [],
    }
    output_text = OUTPUT_ROOT.as_posix()
    project_text = PROJECT.as_posix()
    for record in records:
        command = record.command
        category: str | None = None
        if str(BASE / "idle_waiter.sh") in command:
            category = "waiter"
        elif "generation_stability_full_data_quality_bridge_100k_execute.sh" in command:
            category = "controller"
        elif "scripts/train_generation.py" in command and (
            output_text in command
            or "stability_quality_bridge" in command
            or (record.cwd == project_text and "runtime_preflight/training" in command)
        ):
            category = "trainer"
        elif "run_generation_training_watchdog.py" in command and output_text in command:
            category = "watchdog"
        elif "monitor_generation_pair.py" in command and output_text in command:
            category = "monitor"
        elif output_text in command and any(
            marker in command
            for marker in (
                "preflight_generation_sampling.py",
                "evaluate_generation_checkpoint.py",
                "generate_samples.py",
                "evaluate_generation_metrics.py",
                "evaluate_generation_class_fidelity.py",
                "build_generation_milestone_report.py",
                "build_generation_quality_bridge_result.py",
            )
        ):
            category = "evaluation"
        if category is not None:
            groups[category].append(
                {
                    "pid": record.pid,
                    "ppid": record.ppid,
                    "cwd": record.cwd,
                    "command": command,
                }
            )
    return groups


def blocking_processes(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for category in ("waiter", "controller", "trainer", "watchdog", "evaluation"):
        for process in groups[category]:
            blockers.append({"category": category, **process})
    return blockers


def gpu_rows() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [row.strip() for row in result.stdout.splitlines() if row.strip()]


def latest_watchdog_status() -> dict[str, Any] | None:
    candidates = [
        path
        for path in OUTPUT_ROOT.glob("*/training_watchdog_step_*.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))
    report = read_json(latest)
    if report is None:
        return None
    return {"path": latest.as_posix(), "report": report}


def latest_controller_log_text() -> str:
    if SUPERVISOR_LOG.is_file():
        text = SUPERVISOR_LOG.read_text(encoding="utf-8", errors="replace")
        marker = "launching bounded recovery controller"
        if marker in text:
            return text[text.rfind(marker) :]
    if DIRECT_CONTROLLER_LOG.is_file():
        return DIRECT_CONTROLLER_LOG.read_text(encoding="utf-8", errors="replace")
    return ""


def pin_memory_ipc_failure_is_bounded(
    *,
    watchdog: dict[str, Any] | None,
    monitor: dict[str, Any] | None,
    controller_log_text: str,
) -> tuple[bool, str]:
    watchdog_report = watchdog.get("report") if isinstance(watchdog, dict) else None
    if not isinstance(watchdog_report, dict):
        return False, "exit_1_lacks_watchdog_evidence"
    if (
        watchdog_report.get("status") != "failed"
        or watchdog_report.get("reason") != "child_failed"
        or watchdog_report.get("watchdog_exit_code") != 1
        or watchdog_report.get("child_exit_code") != 1
    ):
        return False, "exit_1_watchdog_evidence_differs"
    missing = [marker for marker in PIN_MEMORY_FAILURE_MARKERS if marker not in controller_log_text]
    if missing:
        return False, "exit_1_lacks_exact_pin_memory_ipc_signature"
    if isinstance(monitor, dict):
        issues = monitor.get("issues")
        if isinstance(issues, list):
            unsafe_terms = (
                "checkpoint integrity",
                "non-finite",
                "nonfinite",
                "non-monotonic",
                "revision",
                "branch",
                "samples_seen",
            )
            for issue in issues:
                normalized = str(issue).lower()
                if any(term in normalized for term in unsafe_terms):
                    return False, "exit_1_has_nontransient_monitor_issue"
    return True, "bounded_pin_memory_resource_sharer_failure"


def classify_retryable_exit(
    exit_code: Any,
    *,
    watchdog: dict[str, Any] | None,
    monitor: dict[str, Any] | None,
    controller_log_text: str | None = None,
) -> tuple[bool, str]:
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        return False, "missing_or_invalid_exit_code"
    if exit_code in UNCONDITIONAL_RETRYABLE_EXIT_CODES:
        return True, f"bounded_transient_exit_{exit_code}"
    if exit_code == 1:
        return pin_memory_ipc_failure_is_bounded(
            watchdog=watchdog,
            monitor=monitor,
            controller_log_text=(
                latest_controller_log_text()
                if controller_log_text is None
                else controller_log_text
            ),
        )
    expected_reason = WATCHDOG_RETRYABLE_REASONS.get(exit_code)
    if expected_reason is None:
        return False, f"exit_{exit_code}_is_not_retryable"
    watchdog_report = watchdog.get("report") if isinstance(watchdog, dict) else None
    if not isinstance(watchdog_report, dict):
        return False, f"exit_{exit_code}_lacks_watchdog_evidence"
    if (
        watchdog_report.get("status") != "failed"
        or watchdog_report.get("reason") != expected_reason
        or watchdog_report.get("watchdog_exit_code") != exit_code
    ):
        return False, f"exit_{exit_code}_watchdog_evidence_differs"
    if isinstance(monitor, dict):
        monitor_status = monitor.get("status")
        monitor_issues = monitor.get("issues")
        if monitor_status in {"failed", "stalled"}:
            return False, f"exit_{exit_code}_has_terminal_monitor_status_{monitor_status}"
        if isinstance(monitor_issues, list) and monitor_issues:
            return False, f"exit_{exit_code}_has_monitor_health_issues"
    return True, f"bounded_watchdog_transient_{expected_reason}"


def retry_decision(status: dict[str, Any] | None, *, now: datetime) -> tuple[str, str]:
    if status is None:
        if LAUNCH_RECEIPT.exists() or any(
            path.exists() and any(path.iterdir())
            for path in (
                OUTPUT_ROOT / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
                OUTPUT_ROOT / "dense_rollout_x0_u2_ema_teacher",
            )
        ):
            return "stop", "unreceipted_state_without_execution_status"
        return "recover", "initial_waiter_disappeared_before_launch"
    state = status.get("status")
    if state == "completed":
        return "stop", "execution_completed_without_verifiable_result"
    if state == "running":
        updated_at = parse_timestamp(status.get("updated_at"))
        if updated_at is None or now - updated_at < timedelta(seconds=ORPHAN_GRACE_SECONDS):
            return "wait", "recent_running_status_without_process"
        return "recover", "orphaned_running_execution"
    if state == "failed":
        exit_code = status.get("exit_code")
        retryable, reason = classify_retryable_exit(
            exit_code,
            watchdog=latest_watchdog_status(),
            monitor=read_json(PAIR_MONITOR),
        )
        return ("recover", reason) if retryable else ("stop", reason)
    return "stop", f"unsupported_execution_status_{state}"


def status_payload(
    *,
    status: str,
    detail: str,
    attempt: int,
    idle_polls: int,
    active: dict[str, list[dict[str, Any]]],
    observed_gpu_rows: list[str],
    child_pid: int | None = None,
    child_exit_code: int | None = None,
    next_retry_seconds: int | None = None,
    identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "cofitok_quality_bridge_100k_bounded_recovery_supervisor_v2",
        "status": status,
        "detail": detail,
        "attempt": attempt,
        "max_recovery_attempts": MAX_RECOVERY_ATTEMPTS,
        "idle_polls": idle_polls,
        "required_idle_polls": REQUIRED_IDLE_POLLS,
        "poll_seconds": POLL_SECONDS,
        "child_pid": child_pid,
        "child_exit_code": child_exit_code,
        "next_retry_seconds": next_retry_seconds,
        "active_processes": active,
        "gpu_compute_rows": observed_gpu_rows,
        "git": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "scope": {
            "quality_bridge_recovery_allowed": True,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
            "report_is_promotion_gate": False,
            "unrelated_gpu_processes_must_not_be_signaled": True,
        },
        "retryable_exit_codes": sorted(RETRYABLE_EXIT_CODES),
        "identities": identities,
        "hostname": socket.gethostname(),
        "supervisor_pid": os.getpid(),
        "updated_at": utc_now().isoformat(),
    }


def write_status(**kwargs: Any) -> None:
    atomic_write_json(SUPERVISOR_STATUS, status_payload(**kwargs))


def verify_terminal_result() -> bool:
    if not QUALITY_RESULT.is_file() or not LAUNCH_RECEIPT.is_file():
        return False
    result_sha = sha256_file(QUALITY_RESULT)
    receipt_sha = sha256_file(LAUNCH_RECEIPT)
    cofitok_run = OUTPUT_ROOT / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
    dense_run = OUTPUT_ROOT / "dense_rollout_x0_u2_ema_teacher"
    args = [
        str(PYTHON),
        "scripts/verify_generation_quality_bridge_result.py",
        "--preparation",
        str(PREPARATION),
        "--expected-preparation-sha256",
        PREPARATION_SHA256,
        "--launch-receipt",
        str(LAUNCH_RECEIPT),
        "--expected-launch-receipt-sha256",
        receipt_sha,
        "--cofitok-training",
        str(cofitok_run / "training_report.json"),
        "--dense-training",
        str(dense_run / "training_report.json"),
        "--training-pair-validation",
        str(REPORT_ROOT / "training_pair_validation.json"),
        "--cofitok-training-audit",
        str(REPORT_ROOT / "cofitok_training_audit.json"),
        "--dense-training-audit",
        str(REPORT_ROOT / "dense_training_audit.json"),
        "--milestone-50000",
        str(REPORT_ROOT / "milestones/step_00050000.json"),
        "--milestone-100000",
        str(REPORT_ROOT / "milestones/step_00100000.json"),
        "--cofitok-sampling-preflight",
        str(cofitok_run / "terminal_100k/sampling_preflight.json"),
        "--dense-sampling-preflight",
        str(dense_run / "terminal_100k/sampling_preflight.json"),
        "--cofitok-generation",
        str(cofitok_run / "terminal_100k/samples_10000_ddim100_cfg15/metrics/generation_metrics_report.json"),
        "--dense-generation",
        str(dense_run / "terminal_100k/samples_10000_ddim100_cfg15/metrics/generation_metrics_report.json"),
        "--cofitok-checkpoint-eval",
        str(cofitok_run / "terminal_100k/checkpoint_eval/checkpoint_evaluation_report.json"),
        "--dense-checkpoint-eval",
        str(dense_run / "terminal_100k/checkpoint_eval/checkpoint_evaluation_report.json"),
        "--class-fidelity-qualification",
        str(REPORT_ROOT / "class_fidelity/qualification_report.json"),
        "--cofitok-class-fidelity",
        str(cofitok_run / "terminal_100k/samples_10000_ddim100_cfg15/class_fidelity/class_fidelity_report.json"),
        "--dense-class-fidelity",
        str(dense_run / "terminal_100k/samples_10000_ddim100_cfg15/class_fidelity/class_fidelity_report.json"),
        "--expected-revision",
        REVISION,
        "--expected-branch",
        BRANCH,
        "--result",
        str(QUALITY_RESULT),
        "--expected-result-sha256",
        result_sha,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{PROJECT}:{PROJECT / 'src'}"
    with SUPERVISOR_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{utc_now().isoformat()}] verifying terminal result {result_sha}\n")
        handle.flush()
        result = subprocess.run(
            args,
            cwd=PROJECT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return result.returncode == 0


def launch_recovery_controller() -> subprocess.Popen[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": str(PYTHON),
            "EXPECTED_TARGET_REVISION": REVISION,
            "EXPECTED_TARGET_BRANCH": BRANCH,
            "EXPECTED_PREPARATION_SHA256": PREPARATION_SHA256,
            "QUALITY_BRIDGE_EXECUTION_APPROVAL": str(APPROVAL),
            "EXPECTED_EXECUTION_APPROVAL_SHA256": APPROVAL_SHA256,
            "QUALITY_BRIDGE_EXECUTION_ALLOWED": "true",
        }
    )
    if LAUNCH_RECEIPT.is_file():
        environment["EXPECTED_LAUNCH_RECEIPT_SHA256"] = sha256_file(LAUNCH_RECEIPT)
    if QUALITY_RESULT.is_file():
        environment["EXPECTED_QUALITY_BRIDGE_RESULT_SHA256"] = sha256_file(QUALITY_RESULT)
    handle = SUPERVISOR_LOG.open("a", encoding="utf-8", newline="\n")
    handle.write(f"[{utc_now().isoformat()}] launching bounded recovery controller\n")
    handle.flush()
    process = subprocess.Popen(
        ["bash", str(RUNBOOK)],
        cwd=PROJECT,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    atomic_write_json(
        RECOVERY_CONTROLLER_RECEIPT,
        {
            "schema_version": 1,
            "pid": process.pid,
            "revision": REVISION,
            "branch": BRANCH,
            "launched_at": utc_now().isoformat(),
        },
    )
    return process


def self_test() -> None:
    now = utc_now()
    original_receipt_exists = LAUNCH_RECEIPT.exists()
    original_state_exists = any(
        path.exists() and any(path.iterdir())
        for path in (
            OUTPUT_ROOT / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
            OUTPUT_ROOT / "dense_rollout_x0_u2_ema_teacher",
        )
    )
    expected_absent_status_decision = (
        "stop" if original_receipt_exists or original_state_exists else "recover"
    )
    assert retry_decision(None, now=now)[0] == expected_absent_status_decision
    assert classify_retryable_exit(86, watchdog=None, monitor=None) == (
        False,
        "exit_86_is_not_retryable",
    )
    assert classify_retryable_exit(
        87,
        watchdog={
            "report": {
                "status": "failed",
                "reason": "monitor_did_not_publish_fresh_status",
                "watchdog_exit_code": 87,
            }
        },
        monitor={"status": "waiting", "issues": []},
    ) == (True, "bounded_watchdog_transient_monitor_did_not_publish_fresh_status")
    assert classify_retryable_exit(
        87,
        watchdog={
            "report": {
                "status": "failed",
                "reason": "monitor_did_not_publish_fresh_status",
                "watchdog_exit_code": 87,
            }
        },
        monitor={"status": "failed", "issues": ["checkpoint integrity mismatch"]},
    )[0] is False
    assert classify_retryable_exit(137, watchdog=None, monitor=None)[0] is True
    pin_watchdog = {
        "report": {
            "status": "failed",
            "reason": "child_failed",
            "watchdog_exit_code": 1,
            "child_exit_code": 1,
        }
    }
    pin_log = "\n".join(PIN_MEMORY_FAILURE_MARKERS)
    assert classify_retryable_exit(
        1,
        watchdog=pin_watchdog,
        monitor={
            "status": "failed",
            "issues": ["matched queue incomplete without active trainer/runbook"],
        },
        controller_log_text=pin_log,
    ) == (True, "bounded_pin_memory_resource_sharer_failure")
    assert classify_retryable_exit(
        1,
        watchdog=pin_watchdog,
        monitor={"status": "failed", "issues": []},
        controller_log_text="RuntimeError: another failure",
    )[0] is False
    assert classify_retryable_exit(
        1,
        watchdog=pin_watchdog,
        monitor={"status": "failed", "issues": ["checkpoint integrity mismatch"]},
        controller_log_text=pin_log,
    )[0] is False
    assert retry_decision(
        {"status": "running", "updated_at": now.isoformat()}, now=now
    )[0] == "wait"
    assert retry_decision(
        {
            "status": "running",
            "updated_at": (now - timedelta(seconds=ORPHAN_GRACE_SECONDS + 1)).isoformat(),
        },
        now=now,
    )[0] == "recover"
    fake = [
        ProcessRecord(
            pid=111,
            ppid=1,
            command=f"bash {BASE / 'idle_waiter.sh'}",
            cwd="/tmp",
        ),
        ProcessRecord(
            pid=222,
            ppid=1,
            command=(
                f"python scripts/monitor_generation_pair.py --output-root {OUTPUT_ROOT}"
            ),
            cwd=PROJECT.as_posix(),
        ),
    ]
    groups = classify_processes(fake)
    assert [item["pid"] for item in groups["waiter"]] == [111]
    assert [item["pid"] for item in groups["monitor"]] == [222]
    assert [item["pid"] for item in blocking_processes(groups)] == [111]
    unrelated = ProcessRecord(
        pid=333,
        ppid=1,
        command="python -m fieldscope.cli run-readout-matrix",
        cwd="/root/autodl-tmp/FieldScope/FieldScope-internal",
    )
    unrelated_groups = classify_processes([unrelated])
    assert all(not values for values in unrelated_groups.values())
    assert blocking_processes(unrelated_groups) == []
    print("self-test passed")


def run_supervisor(*, once: bool) -> int:
    SUPERVISOR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = SUPERVISOR_LOCK.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("quality bridge recovery supervisor already owns its lock", file=sys.stderr)
        return 75
    SUPERVISOR_PID.write_text(f"{os.getpid()}\n", encoding="ascii")
    previous = read_json(SUPERVISOR_STATUS) or {}
    attempt = int(previous.get("attempt", 0))
    idle_polls = 0
    while True:
        try:
            identities = require_static_bindings()
        except Exception as error:
            active = classify_processes(process_records())
            write_status(
                status="failed",
                detail=f"static_binding_failure:{type(error).__name__}:{error}",
                attempt=attempt,
                idle_polls=idle_polls,
                active=active,
                observed_gpu_rows=gpu_rows(),
            )
            return 70
        active = classify_processes(process_records())
        blockers = blocking_processes(active)
        observed_gpu = gpu_rows()
        if verify_terminal_result():
            write_status(
                status="pass",
                detail="quality_bridge_terminal_result_physically_verified",
                attempt=attempt,
                idle_polls=idle_polls,
                active=active,
                observed_gpu_rows=observed_gpu,
                identities=identities,
            )
            return 0
        if blockers:
            idle_polls = 0
            write_status(
                status="observing",
                detail="existing_quality_bridge_execution_is_active",
                attempt=attempt,
                idle_polls=idle_polls,
                active=active,
                observed_gpu_rows=observed_gpu,
                identities=identities,
            )
            if once:
                return 0
            time.sleep(POLL_SECONDS)
            continue
        decision, reason = retry_decision(read_json(EXECUTION_STATUS), now=utc_now())
        if decision == "stop":
            write_status(
                status="failed",
                detail=reason,
                attempt=attempt,
                idle_polls=idle_polls,
                active=active,
                observed_gpu_rows=observed_gpu,
                identities=identities,
            )
            return 71
        if decision == "wait":
            idle_polls = 0
            write_status(
                status="waiting",
                detail=reason,
                attempt=attempt,
                idle_polls=idle_polls,
                active=active,
                observed_gpu_rows=observed_gpu,
                identities=identities,
            )
            if once:
                return 0
            time.sleep(POLL_SECONDS)
            continue
        if observed_gpu:
            idle_polls = 0
            write_status(
                status="waiting",
                detail=f"{reason}:waiting_for_gpu_idle",
                attempt=attempt,
                idle_polls=idle_polls,
                active=active,
                observed_gpu_rows=observed_gpu,
                identities=identities,
            )
            if once:
                return 0
            time.sleep(POLL_SECONDS)
            continue
        idle_polls += 1
        write_status(
            status="waiting",
            detail=f"{reason}:confirming_gpu_idle",
            attempt=attempt,
            idle_polls=idle_polls,
            active=active,
            observed_gpu_rows=observed_gpu,
            identities=identities,
        )
        if once or idle_polls < REQUIRED_IDLE_POLLS:
            if once:
                return 0
            time.sleep(POLL_SECONDS)
            continue
        require_static_bindings()
        active = classify_processes(process_records())
        if blocking_processes(active) or gpu_rows():
            idle_polls = 0
            continue
        attempt += 1
        if attempt > MAX_RECOVERY_ATTEMPTS:
            write_status(
                status="failed",
                detail="bounded_recovery_attempts_exhausted",
                attempt=attempt - 1,
                idle_polls=idle_polls,
                active=active,
                observed_gpu_rows=[],
                identities=identities,
            )
            return 72
        child = launch_recovery_controller()
        write_status(
            status="recovering",
            detail=reason,
            attempt=attempt,
            idle_polls=idle_polls,
            active=classify_processes(process_records()),
            observed_gpu_rows=gpu_rows(),
            child_pid=child.pid,
            identities=identities,
        )
        while child.poll() is None:
            time.sleep(POLL_SECONDS)
            write_status(
                status="recovering",
                detail="bounded_recovery_controller_is_active",
                attempt=attempt,
                idle_polls=0,
                active=classify_processes(process_records()),
                observed_gpu_rows=gpu_rows(),
                child_pid=child.pid,
                identities=identities,
            )
        exit_code = int(child.returncode)
        if verify_terminal_result():
            write_status(
                status="pass",
                detail="quality_bridge_terminal_result_physically_verified",
                attempt=attempt,
                idle_polls=0,
                active=classify_processes(process_records()),
                observed_gpu_rows=gpu_rows(),
                child_pid=child.pid,
                child_exit_code=exit_code,
                identities=identities,
            )
            return 0
        retryable, retry_reason = classify_retryable_exit(
            exit_code,
            watchdog=latest_watchdog_status(),
            monitor=read_json(PAIR_MONITOR),
        )
        if not retryable:
            write_status(
                status="failed",
                detail=f"nonretryable_recovery_controller_exit_{exit_code}:{retry_reason}",
                attempt=attempt,
                idle_polls=0,
                active=classify_processes(process_records()),
                observed_gpu_rows=gpu_rows(),
                child_pid=child.pid,
                child_exit_code=exit_code,
                identities=identities,
            )
            return exit_code if exit_code != 0 else 73
        delay = min(BASE_RETRY_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_SECONDS)
        write_status(
            status="retrying",
            detail=f"bounded_retryable_recovery_controller_exit_{exit_code}:{retry_reason}",
            attempt=attempt,
            idle_polls=0,
            active=classify_processes(process_records()),
            observed_gpu_rows=gpu_rows(),
            child_pid=child.pid,
            child_exit_code=exit_code,
            next_retry_seconds=delay,
            identities=identities,
        )
        time.sleep(delay)
        idle_polls = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    raise SystemExit(run_supervisor(once=args.once))


if __name__ == "__main__":
    main()
