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
from pathlib import Path
from typing import Any

from cofitok.monitoring import (
    CHECKPOINT_INTEGRITY_POLICIES,
    build_monitor_report,
    inspect_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _processes(pattern: str) -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


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

    output_root = Path(args.output_root)
    output = Path(args.output)
    while True:
        now = time.time()
        runs = {
            "cofitok": inspect_run(
                output_root / args.cofitok_run,
                expected_steps=args.expected_steps,
                now=now,
                checkpoint_interval=args.checkpoint_interval,
                checkpoint_grace_steps=args.checkpoint_grace_steps,
                checkpoint_integrity_policy=args.checkpoint_integrity_policy,
            ),
            "dense_identity": inspect_run(
                output_root / args.dense_run,
                expected_steps=args.expected_steps,
                now=now,
                checkpoint_interval=args.checkpoint_interval,
                checkpoint_grace_steps=args.checkpoint_grace_steps,
                checkpoint_integrity_policy=args.checkpoint_integrity_policy,
            ),
        }
        usage = shutil.disk_usage(output_root)
        report = build_monitor_report(
            runs=runs,
            training_processes=_processes(args.training_process_pattern),
            runbook_processes=_processes(args.runbook_process_pattern),
            stall_seconds=args.stall_seconds,
            idle_failure_grace_seconds=args.idle_failure_grace_seconds,
            disk={
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            },
            gpu=_gpu_status(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            hostname=socket.gethostname(),
            monitor_name=args.monitor_name,
            git=_git_state(),
        )
        _write_atomic(output, report)
        print(json.dumps({"status": report["status"], "stage": report["stage"]}))
        if args.once or report["status"] in {"pass", "failed", "stalled"}:
            raise SystemExit(0 if report["status"] in {"pass", "running", "waiting"} else 1)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
