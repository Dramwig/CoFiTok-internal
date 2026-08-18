from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.gpu_contention import update_gpu_contention_evidence
from cofitok.monitoring import (
    CHECKPOINT_INTEGRITY_POLICIES,
    build_monitor_report,
    inspect_run,
)
from cofitok.process_identity import (
    linux_process_identity_mismatches,
    read_linux_process_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXACT_RUNBOOK_IDENTITY_SCHEMA_VERSION = 1
EXACT_RUNBOOK_IDENTITY_ROLE = "generation_exact_runbook_process_identity"


def _processes(pattern: str) -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _read_process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any] | None:
    return read_linux_process_identity(pid, proc_root=proc_root)


def _process_records(
    lines: list[str],
    *,
    proc_root: Path = Path("/proc"),
) -> list[dict[str, Any]]:
    records = []
    for line in lines:
        raw_pid, separator, _ = line.strip().partition(" ")
        if not separator:
            continue
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        identity = _read_process_identity(pid, proc_root=proc_root)
        if identity is not None:
            records.append(identity)
    return records


def _runbook_identity_mismatches(
    observed: dict[str, Any] | None,
    *,
    expected: dict[str, Any],
) -> list[str]:
    return linux_process_identity_mismatches(observed, expected=expected)


def inspect_exact_runbook_identity(
    *,
    expected: dict[str, Any],
    pattern_processes: list[str],
    proc_root: Path = Path("/proc"),
) -> tuple[list[str], dict[str, Any]]:
    observed = _read_process_identity(int(expected["pid"]), proc_root=proc_root)
    mismatches = _runbook_identity_mismatches(observed, expected=expected)
    pattern_records = _process_records(pattern_processes, proc_root=proc_root)
    pattern_pids = sorted({int(record["pid"]) for record in pattern_records})
    bound_pid = int(expected["pid"])
    active = not mismatches
    evidence = {
        "schema_version": EXACT_RUNBOOK_IDENTITY_SCHEMA_VERSION,
        "role": EXACT_RUNBOOK_IDENTITY_ROLE,
        "mode": "exact_process_identity",
        "status": "active" if active else "invalid",
        "expected": dict(expected),
        "observed": dict(observed) if observed is not None else None,
        "mismatches": mismatches,
        "pattern_candidates": {
            "pids": pattern_pids,
            "bound_pid_reported": bound_pid in pattern_pids,
            "nonbound_pids": [pid for pid in pattern_pids if pid != bound_pid],
            "substring_candidate_contamination_observed": any(
                pid != bound_pid for pid in pattern_pids
            ),
        },
    }
    runbook_processes = (
        [f"{bound_pid} {observed['argv']}"] if active and observed is not None else []
    )
    return runbook_processes, evidence


def enforce_exact_runbook_identity(
    report: dict[str, Any],
    *,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    report["runbook_identity"] = evidence
    if report.get("status") == "pass" or evidence.get("status") == "active":
        return report
    mismatches = evidence.get("mismatches", [])
    issue = "exact runbook identity is unavailable before pair completion"
    if isinstance(mismatches, list) and mismatches:
        issue += ": " + ", ".join(str(value) for value in mismatches)
    issues = report.setdefault("issues", [])
    if issue not in issues:
        issues.append(issue)
    report["status"] = "failed"
    return report


def _exact_runbook_binding(args: argparse.Namespace) -> dict[str, Any] | None:
    values = {
        "pid": int(args.runbook_process_pid),
        "start_ticks": int(args.runbook_process_start_ticks),
        "executable": str(args.runbook_process_executable),
        "cwd": str(args.runbook_process_cwd),
        "cmdline_sha256": str(args.runbook_process_cmdline_sha256).lower(),
    }
    requested = any(
        (
            values["pid"] > 0,
            values["start_ticks"] > 0,
            bool(values["executable"]),
            bool(values["cwd"]),
            bool(values["cmdline_sha256"]),
        )
    )
    if not requested:
        return None
    if (
        values["pid"] < 1
        or values["start_ticks"] < 1
        or not PurePosixPath(values["executable"]).is_absolute()
        or not PurePosixPath(values["cwd"]).is_absolute()
        or len(values["cmdline_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in values["cmdline_sha256"]
        )
    ):
        raise ValueError("exact runbook process identity is incomplete")
    return values


def _gpu_compute_processes() -> tuple[list[dict[str, Any]], bool]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [], False
    compute = []
    complete = True
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) != 3:
            complete = False
            continue
        try:
            pid = int(values[0])
            used_memory_mib = int(values[1])
            proc = Path("/proc") / str(pid)
            argv = (
                (proc / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            stat = (proc / "stat").read_text(encoding="utf-8").rsplit(")", 1)[
                1
            ].split()
            compute.append(
                {
                    "pid": pid,
                    "start_ticks": int(stat[19]),
                    "argv": argv,
                    "cwd": (proc / "cwd").resolve().as_posix(),
                    "process_name": values[2],
                    "used_memory_mib": used_memory_mib,
                }
            )
        except (FileNotFoundError, IndexError, OSError, PermissionError, ValueError):
            complete = False
            continue
    return compute, complete


def _previous_gpu_contention(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prior pair monitor report is not a JSON object")
    evidence = payload.get("gpu_contention")
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise ValueError("prior pair monitor GPU contention evidence is malformed")
    return evidence


def _gpu_status() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            rows.append(
                {
                    "index": int(values[0]),
                    "utilization_percent": int(values[1]),
                    "memory_used_mib": int(values[2]),
                    "memory_total_mib": int(values[3]),
                    "temperature_c": int(values[4]),
                }
            )
    return rows


def _git_state() -> dict[str, str | bool]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(defaults: dict[str, Any] | None = None) -> argparse.Namespace:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Read-only monitor for a generation pair.")
    parser.add_argument(
        "--output-root",
        default=defaults.get(
            "output_root", "/root/autodl-tmp/CoFiTok/checkpoints/generation"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--monitor-name",
        default=defaults.get("monitor_name"),
        required="monitor_name" not in defaults,
    )
    parser.add_argument(
        "--cofitok-run",
        default=defaults.get("cofitok_run"),
        required="cofitok_run" not in defaults,
    )
    parser.add_argument(
        "--dense-run",
        default=defaults.get("dense_run"),
        required="dense_run" not in defaults,
    )
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=defaults.get("expected_steps"),
        required="expected_steps" not in defaults,
    )
    parser.add_argument(
        "--training-process-pattern",
        default=defaults.get("training_process_pattern"),
        required="training_process_pattern" not in defaults,
    )
    parser.add_argument(
        "--runbook-process-pattern",
        default=defaults.get("runbook_process_pattern"),
        required="runbook_process_pattern" not in defaults,
    )
    parser.add_argument(
        "--runbook-process-pid",
        type=int,
        default=defaults.get("runbook_process_pid", 0),
        help=(
            "Optional exact controller PID. When set, all exact runbook "
            "identity fields are required and the pattern is diagnostic only."
        ),
    )
    parser.add_argument(
        "--runbook-process-start-ticks",
        type=int,
        default=defaults.get("runbook_process_start_ticks", 0),
    )
    parser.add_argument(
        "--runbook-process-executable",
        default=defaults.get("runbook_process_executable", ""),
    )
    parser.add_argument(
        "--runbook-process-cwd",
        default=defaults.get("runbook_process_cwd", ""),
    )
    parser.add_argument(
        "--runbook-process-cmdline-sha256",
        default=defaults.get("runbook_process_cmdline_sha256", ""),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=defaults.get("checkpoint_interval", 0),
    )
    parser.add_argument(
        "--checkpoint-grace-steps",
        type=int,
        default=defaults.get("checkpoint_grace_steps", 250),
    )
    parser.add_argument(
        "--checkpoint-integrity-policy",
        choices=CHECKPOINT_INTEGRITY_POLICIES,
        default=defaults.get("checkpoint_integrity_policy", "optional"),
    )
    parser.add_argument(
        "--expected-checkpoint-revision",
        default=defaults.get("expected_checkpoint_revision", ""),
        help=(
            "Expected checkpoint sidecar Git revision. Defaults to the monitor "
            "checkout revision."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--stall-seconds", type=float, default=1_800.0)
    parser.add_argument("--idle-failure-grace-seconds", type=float, default=600.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main(defaults: dict[str, Any] | None = None) -> None:
    args = parse_args(defaults)
    if min(
        args.expected_steps,
        args.poll_seconds,
        args.stall_seconds,
        args.idle_failure_grace_seconds,
    ) <= 0 or min(args.checkpoint_interval, args.checkpoint_grace_steps) < 0:
        raise ValueError("monitor steps and timing thresholds are invalid")
    exact_runbook_binding = _exact_runbook_binding(args)

    output_root = Path(args.output_root)
    output = Path(args.output)
    previous_gpu_contention = _previous_gpu_contention(output)
    while True:
        now = time.time()
        git_state = _git_state()
        expected_checkpoint_revision = (
            args.expected_checkpoint_revision or str(git_state["revision"])
        )
        runs = {
            "cofitok": inspect_run(
                output_root / args.cofitok_run,
                expected_steps=args.expected_steps,
                now=now,
                checkpoint_interval=args.checkpoint_interval,
                checkpoint_grace_steps=args.checkpoint_grace_steps,
                checkpoint_integrity_policy=args.checkpoint_integrity_policy,
                expected_checkpoint_revision=expected_checkpoint_revision,
            ),
            "dense_identity": inspect_run(
                output_root / args.dense_run,
                expected_steps=args.expected_steps,
                now=now,
                checkpoint_interval=args.checkpoint_interval,
                checkpoint_grace_steps=args.checkpoint_grace_steps,
                checkpoint_integrity_policy=args.checkpoint_integrity_policy,
                expected_checkpoint_revision=expected_checkpoint_revision,
            ),
        }
        usage = shutil.disk_usage(output_root)
        training_processes = _processes(args.training_process_pattern)
        pattern_runbook_processes = _processes(args.runbook_process_pattern)
        runbook_identity = None
        if exact_runbook_binding is None:
            runbook_processes = pattern_runbook_processes
        else:
            runbook_processes, runbook_identity = inspect_exact_runbook_identity(
                expected=exact_runbook_binding,
                pattern_processes=pattern_runbook_processes,
            )
        updated_at = datetime.now(timezone.utc).isoformat()
        report = build_monitor_report(
            runs=runs,
            training_processes=training_processes,
            runbook_processes=runbook_processes,
            stall_seconds=args.stall_seconds,
            idle_failure_grace_seconds=args.idle_failure_grace_seconds,
            disk={
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            },
            gpu=_gpu_status(),
            updated_at=updated_at,
            hostname=socket.gethostname(),
            monitor_name=args.monitor_name,
            git=git_state,
        )
        if runbook_identity is not None:
            report = enforce_exact_runbook_identity(
                report,
                evidence=runbook_identity,
            )
        gpu_compute_processes, gpu_query_complete = _gpu_compute_processes()
        report["gpu_contention"] = update_gpu_contention_evidence(
            previous_gpu_contention,
            pair_report=report,
            training_processes=_process_records(training_processes),
            gpu_compute_processes=gpu_compute_processes,
            observed_at=updated_at,
            poll_seconds=args.poll_seconds,
            gpu_query_complete=gpu_query_complete,
        )
        previous_gpu_contention = report["gpu_contention"]
        _write_atomic(output, report)
        print(json.dumps({"status": report["status"], "stage": report["stage"]}))
        if args.once or report["status"] in {"pass", "failed", "stalled"}:
            raise SystemExit(0 if report["status"] in {"pass", "running", "waiting"} else 1)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
