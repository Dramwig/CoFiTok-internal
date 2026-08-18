from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cofitok.process_identity import read_linux_process_identity


MONITOR_ARGUMENT_FIELDS = (
    ("--runbook-process-pid", "pid"),
    ("--runbook-process-start-ticks", "start_ticks"),
    ("--runbook-process-executable", "executable"),
    ("--runbook-process-cwd", "cwd"),
    ("--runbook-process-cmdline-sha256", "cmdline_sha256"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an exact Linux process identity for generation monitor "
            "controller binding."
        )
    )
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument(
        "--format",
        choices=("json", "monitor-args0"),
        default="json",
        help=(
            "json emits the full identity; monitor-args0 emits NUL-delimited "
            "argument/value tokens for a Bash array."
        ),
    )
    parser.add_argument(
        "--proc-root",
        type=Path,
        default=Path("/proc"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def monitor_argument_tokens(identity: dict[str, object]) -> list[str]:
    return [
        token
        for option, field in MONITOR_ARGUMENT_FIELDS
        for token in (option, str(identity[field]))
    ]


def main() -> None:
    args = parse_args()
    identity = read_linux_process_identity(args.pid, proc_root=args.proc_root)
    if identity is None:
        raise SystemExit(f"process identity is unavailable for PID {args.pid}")
    if args.format == "json":
        print(json.dumps(identity, indent=2, sort_keys=True))
        return
    tokens = monitor_argument_tokens(identity)
    sys.stdout.buffer.write(
        b"".join(token.encode("utf-8") + b"\0" for token in tokens)
    )


if __name__ == "__main__":
    main()
