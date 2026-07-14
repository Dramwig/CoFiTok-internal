from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.training.watchdog import TrainingWatchdogConfig, run_training_watchdog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one generation-training command and fail it when its read-only "
            "pair monitor stalls, fails, or disappears."
        )
    )
    parser.add_argument("--monitor-report", required=True)
    parser.add_argument("--monitor-pid-file", required=True)
    parser.add_argument("--expected-monitor-name", required=True)
    parser.add_argument("--status-output", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=600.0)
    parser.add_argument("--monitor-silence-seconds", type=float, default=900.0)
    parser.add_argument("--monitor-process-grace-seconds", type=float, default=120.0)
    parser.add_argument("--termination-grace-seconds", type=float, default=60.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    config = TrainingWatchdogConfig(
        monitor_report=Path(args.monitor_report),
        monitor_pid_file=Path(args.monitor_pid_file),
        expected_monitor_name=args.expected_monitor_name,
        status_output=Path(args.status_output),
        command=tuple(command),
        poll_seconds=args.poll_seconds,
        startup_grace_seconds=args.startup_grace_seconds,
        monitor_silence_seconds=args.monitor_silence_seconds,
        monitor_process_grace_seconds=args.monitor_process_grace_seconds,
        termination_grace_seconds=args.termination_grace_seconds,
    )
    raise SystemExit(run_training_watchdog(config))


if __name__ == "__main__":
    main()
