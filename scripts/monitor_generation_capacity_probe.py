from __future__ import annotations

import argparse
import json
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from cofitok.generation.capacity_probe_monitor import (
    CAPACITY_PROBE_MONITOR_NAME,
    build_capacity_probe_monitor_report,
    inspect_capacity_probe_run,
)
from scripts.monitor_generation_pair import (
    _git_state,
    _gpu_status,
    _processes,
    _write_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only monitor for the matched 250M capacity probe that treats "
            "the exact step-10K partial stop as completion."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monitor-name", default=CAPACITY_PROBE_MONITOR_NAME)
    parser.add_argument("--cofitok-run", required=True)
    parser.add_argument("--dense-run", required=True)
    parser.add_argument("--cofitok-validation", type=Path, required=True)
    parser.add_argument("--dense-validation", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-stop-step", type=int, default=10_000)
    parser.add_argument("--expected-configured-steps", type=int, default=100_000)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--checkpoint-grace-steps", type=int, default=250)
    parser.add_argument(
        "--training-process-pattern",
        default=r"[s]cripts/train_generation.py.*stability_capacity_probe.*100k",
    )
    parser.add_argument(
        "--runbook-process-pattern",
        default=r"[g]eneration_stability_capacity_probe_250m_10k_execute.sh",
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--stall-seconds", type=float, default=1_800.0)
    parser.add_argument("--idle-failure-grace-seconds", type=float, default=600.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if min(
        args.expected_stop_step,
        args.expected_configured_steps,
        args.checkpoint_interval,
        args.poll_seconds,
        args.stall_seconds,
        args.idle_failure_grace_seconds,
    ) <= 0 or args.checkpoint_grace_steps < 0:
        parser.error("capacity probe monitor thresholds are invalid")
    return args


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    while True:
        now = time.time()
        git = _git_state()
        runs = {
            "cofitok": inspect_capacity_probe_run(
                output_root / args.cofitok_run,
                validation_report=args.cofitok_validation,
                now=now,
                expected_revision=args.expected_revision,
                expected_branch=args.expected_branch,
                expected_stop_step=args.expected_stop_step,
                expected_configured_steps=args.expected_configured_steps,
                checkpoint_interval=args.checkpoint_interval,
                checkpoint_grace_steps=args.checkpoint_grace_steps,
            ),
            "dense_identity": inspect_capacity_probe_run(
                output_root / args.dense_run,
                validation_report=args.dense_validation,
                now=now,
                expected_revision=args.expected_revision,
                expected_branch=args.expected_branch,
                expected_stop_step=args.expected_stop_step,
                expected_configured_steps=args.expected_configured_steps,
                checkpoint_interval=args.checkpoint_interval,
                checkpoint_grace_steps=args.checkpoint_grace_steps,
            ),
        }
        usage = shutil.disk_usage(output_root)
        report = build_capacity_probe_monitor_report(
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
            git=git,
            monitor_name=args.monitor_name,
        )
        _write_atomic(args.output, report)
        print(json.dumps({"status": report["status"], "stage": report["stage"]}))
        if args.once or report["status"] in {"pass", "failed", "stalled"}:
            raise SystemExit(
                0 if report["status"] in {"pass", "running", "waiting"} else 1
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
