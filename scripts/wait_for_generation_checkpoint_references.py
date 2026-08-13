from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from cofitok.reporting import write_json_report
from scripts.preserve_generation_checkpoint_reference import (
    preserve_checkpoint_reference,
    verify_checkpoint_reference,
)


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_checkpoint_reference_waiter"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_source(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise ValueError("source must use NAME=CHECKPOINT_PATH")
    return name.strip(), Path(raw_path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for and hardlink-preserve a matched checkpoint reference set."
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--receipt-root", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    return args


def _atomic_status(path: Path, payload: dict[str, Any]) -> None:
    write_json_report(path, payload)


def main() -> None:
    args = parse_args()
    parsed = [_parse_source(value) for value in args.source]
    if len(parsed) < 1 or len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("checkpoint reference source names must be unique")
    reference_root = Path(args.reference_root).resolve()
    receipt_root = Path(args.receipt_root).resolve()
    status_path = Path(args.status).resolve()
    base = {
        "schema_version": WAITER_SCHEMA_VERSION,
        "role": WAITER_ROLE,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected_step": args.expected_step,
        "expected_git": {
            "revision": args.expected_revision,
            "branch": args.expected_branch,
            "tracked_dirty": False,
        },
        "reason": args.reason,
        "poll_seconds": args.poll_seconds,
        "sources": {name: path.as_posix() for name, path in parsed},
        "reference_root": reference_root.as_posix(),
        "receipt_root": receipt_root.as_posix(),
        "authorization_boundary": {
            "training_launch_allowed": False,
            "gpu_use_allowed": False,
            "checkpoint_mutation_allowed": False,
            "checkpoint_deletion_allowed": False,
            "full_300k_launch_allowed": False,
            "release_allowed": False,
        },
    }
    polls = 0
    while True:
        polls += 1
        rows: dict[str, Any] = {}
        all_preserved = True
        for name, checkpoint in parsed:
            receipt_path = receipt_root / f"{name}_step_{args.expected_step:08d}.json"
            if receipt_path.is_file():
                with receipt_path.open(encoding="utf-8") as handle:
                    receipt = json.load(handle)
                verify_checkpoint_reference(
                    receipt,
                    expected_step=args.expected_step,
                    expected_revision=args.expected_revision,
                    expected_branch=args.expected_branch,
                    expected_reason=args.reason,
                )
                rows[name] = {
                    "status": "preserved",
                    "receipt": receipt_path.as_posix(),
                    "reference_checkpoint": receipt["reference"]["checkpoint"]["path"],
                    "checkpoint_sha256": receipt["reference"]["checkpoint"]["sha256"],
                }
                continue
            if checkpoint.is_file() and checkpoint.with_name(
                f"{checkpoint.name}.integrity.json"
            ).is_file():
                report = preserve_checkpoint_reference(
                    checkpoint=checkpoint,
                    reference_dir=reference_root / name,
                    expected_step=args.expected_step,
                    expected_revision=args.expected_revision,
                    expected_branch=args.expected_branch,
                    reason=args.reason,
                )
                write_json_report(receipt_path, report)
                rows[name] = {
                    "status": "preserved",
                    "receipt": receipt_path.as_posix(),
                    "reference_checkpoint": report["reference"]["checkpoint"]["path"],
                    "checkpoint_sha256": report["reference"]["checkpoint"]["sha256"],
                }
            else:
                all_preserved = False
                rows[name] = {
                    "status": "waiting",
                    "checkpoint_exists": checkpoint.is_file(),
                    "integrity_exists": checkpoint.with_name(
                        f"{checkpoint.name}.integrity.json"
                    ).is_file(),
                }
        _atomic_status(
            status_path,
            {
                **base,
                "status": "completed" if all_preserved else "waiting",
                "detail": (
                    "matched_checkpoint_references_preserved"
                    if all_preserved
                    else "waiting_for_source_checkpoints"
                ),
                "polls": polls,
                "updated_at": _utc_now(),
                "methods": rows,
            },
        )
        if all_preserved:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
