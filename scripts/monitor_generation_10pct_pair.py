from __future__ import annotations

import argparse
import json
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


CHECKPOINT_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt$")
RUNS = {
    "cofitok": "imagenet256_10pct_cofitok_k8_50k_2026-07-12",
    "dense_identity": "imagenet256_10pct_dense_50k_2026-07-12",
}


def _read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inspect_run(run_dir: str | Path, *, expected_steps: int, now: float) -> dict[str, Any]:
    root = Path(run_dir)
    metrics_path = root / "train_metrics.jsonl"
    last = _read_last_jsonl(metrics_path)
    report = _read_json(root / "training_report.json")
    checkpoints = []
    for path in root.glob("checkpoint_step_*.pt") if root.is_dir() else ():
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            checkpoints.append(
                {
                    "step": int(match.group(1)),
                    "bytes": path.stat().st_size,
                    "name": path.name,
                }
            )
    checkpoints.sort(key=lambda row: row["step"])
    last_step = int(last.get("step", 0)) if last is not None else 0
    complete = bool(
        report is not None
        and report.get("training_complete") is True
        and int(report.get("completed_steps", -1)) == expected_steps
        and int(report.get("target_steps", -1)) == expected_steps
    )
    activity_mtime = max(
        [
            path.stat().st_mtime
            for path in (metrics_path, root / "training_report.json")
            if path.is_file()
        ]
        or [0.0]
    )
    return {
        "run_dir": root.resolve().as_posix(),
        "exists": root.is_dir(),
        "complete": complete,
        "expected_steps": expected_steps,
        "last_step": last_step,
        "progress_fraction": last_step / expected_steps,
        "last_metric": last,
        "activity_age_seconds": now - activity_mtime if activity_mtime > 0.0 else None,
        "checkpoints": checkpoints,
        "training_report": report,
    }


def build_monitor_report(
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
) -> dict[str, Any]:
    cofitok = runs["cofitok"]
    dense = runs["dense_identity"]
    if cofitok["complete"] and dense["complete"]:
        status, stage, issues = "pass", "complete", []
    else:
        active = bool(training_processes or runbook_processes)
        current_name = "dense_identity" if cofitok["complete"] else "cofitok"
        current = runs[current_name]
        age = current.get("activity_age_seconds")
        if active and age is not None and age > stall_seconds:
            status, stage = "stalled", f"{current_name}_training"
            issues = [f"{current_name} metrics have not advanced for {age:.1f} seconds"]
        elif active:
            status, stage, issues = "running", f"{current_name}_training", []
        else:
            observed_ages = [
                float(run["activity_age_seconds"])
                for run in runs.values()
                if run.get("activity_age_seconds") is not None
            ]
            freshest_age = min(observed_ages, default=float("inf"))
            if freshest_age <= idle_failure_grace_seconds:
                status, stage, issues = "waiting", f"{current_name}_transition", []
            else:
                status, stage = "failed", f"{current_name}_training"
                issues = ["matched queue is incomplete without an active training or runbook process"]
    return {
        "schema_version": 1,
        "monitor": "generation_10pct_matched_pair",
        "status": status,
        "stage": stage,
        "updated_at": updated_at,
        "hostname": hostname,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only monitor for the pinned 10% pair.")
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/CoFiTok/checkpoints/generation",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-steps", type=int, default=50_000)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--stall-seconds", type=float, default=1_800.0)
    parser.add_argument("--idle-failure-grace-seconds", type=float, default=600.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if min(
        args.expected_steps,
        args.poll_seconds,
        args.stall_seconds,
        args.idle_failure_grace_seconds,
    ) <= 0:
        raise ValueError("monitor steps and timing thresholds must be positive")

    output_root = Path(args.output_root)
    output = Path(args.output)
    while True:
        now = time.time()
        runs = {
            name: inspect_run(
                output_root / directory,
                expected_steps=args.expected_steps,
                now=now,
            )
            for name, directory in RUNS.items()
        }
        usage = shutil.disk_usage(output_root)
        report = build_monitor_report(
            runs=runs,
            training_processes=_processes(
                r"[s]cripts/train_generation.py.*imagenet256_10pct_"
            ),
            runbook_processes=_processes(
                r"[g]eneration_10pct_matched_50k_2026-07-12.sh"
            ),
            stall_seconds=args.stall_seconds,
            idle_failure_grace_seconds=args.idle_failure_grace_seconds,
            disk={"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
            gpu=_gpu_status(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            hostname=socket.gethostname(),
        )
        _write_atomic(output, report)
        print(json.dumps({"status": report["status"], "stage": report["stage"]}))
        if args.once or report["status"] in {"pass", "failed", "stalled"}:
            raise SystemExit(0 if report["status"] in {"pass", "running", "waiting"} else 1)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
