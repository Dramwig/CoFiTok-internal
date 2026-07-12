#!/usr/bin/env python
"""Run or resume the AAAI-27 ablation suite on the experiment server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config(workspace: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else workspace / path


def run_logged(command: list[str], cwd: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd / "src")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("command: " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - start,
        "log": str(log_path),
    }


def train_entry(entry: dict[str, Any], workspace: Path, force: bool) -> dict[str, Any]:
    output_dir = Path(entry["checkpoint_dir"])
    checkpoint = output_dir / "checkpoint_final.pt"
    report = output_dir / "report.json"
    if checkpoint.exists() and report.exists() and not force:
        return {"id": entry["id"], "stage": "train", "status": "skipped_complete"}
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/train_short.py",
        "--config",
        str(resolve_config(workspace, entry["config"])),
        "--output-dir",
        str(output_dir),
    ]
    result = run_logged(command, workspace, output_dir / "train.log")
    status = "completed" if result["returncode"] == 0 and checkpoint.exists() and report.exists() else "failed"
    return {"id": entry["id"], "stage": "train", "status": status, **result}


def eval_entry(
    entry: dict[str, Any],
    workspace: Path,
    suite_root: Path,
    protocol: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    output_dir = suite_root / "eval" / str(entry["id"])
    report = output_dir / "quality_report.json"
    if report.exists() and not force:
        return {"id": entry["id"], "stage": "eval", "status": "skipped_complete"}
    checkpoint = Path(entry["checkpoint_dir"]) / "checkpoint_final.pt"
    if not checkpoint.exists():
        return {
            "id": entry["id"],
            "stage": "eval",
            "status": "failed",
            "reason": f"missing checkpoint: {checkpoint}",
        }
    token_count = int(entry["token_count"])
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/evaluate_quality.py",
        "--config",
        str(resolve_config(workspace, entry["config"])),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--split",
        str(protocol["split"]),
        "--max-batches",
        str(max(32, int(protocol["image_count"]))),
        "--max-images",
        str(protocol["image_count"]),
        "--timestep",
        str(protocol["timestep"]),
        "--prefix-budgets",
        ",".join(str(value) for value in range(1, token_count + 1)),
        "--component-order",
        str(protocol["component_order"]),
        "--evaluation-progress-power",
        str(protocol["evaluation_progress_power"]),
    ]
    result = run_logged(command, workspace, output_dir / "eval.log")
    status = "completed" if result["returncode"] == 0 and report.exists() else "failed"
    return {"id": entry["id"], "stage": "eval", "status": status, **result}


def run_parallel(function, entries: list[dict[str, Any]], jobs: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(function, entry): entry for entry in entries}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    return sorted(results, key=lambda row: (str(row["stage"]), str(row["id"])))


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.train_only and args.eval_only:
        raise ValueError("--train-only and --eval-only are mutually exclusive")
    manifest = read_json(args.manifest)
    suite_root = Path(manifest["suite_root"])
    suite_root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    if not args.eval_only:
        train_entries = [entry for entry in manifest["entries"] if entry.get("train")]
        all_results.extend(
            run_parallel(
                lambda entry: train_entry(entry, args.workspace, args.force),
                train_entries,
                args.jobs,
            )
        )

    if not args.train_only:
        all_results.extend(
            run_parallel(
                lambda entry: eval_entry(
                    entry,
                    args.workspace,
                    suite_root,
                    manifest["evaluation_protocol"],
                    args.force,
                ),
                list(manifest["entries"]),
                args.jobs,
            )
        )

    failed = [row for row in all_results if row["status"] == "failed"]
    status = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "entry_count": len(manifest["entries"]),
        "results": all_results,
        "failed_count": len(failed),
        "complete": not failed,
    }
    (suite_root / "suite_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failed:
        raise SystemExit(f"{len(failed)} suite tasks failed")


if __name__ == "__main__":
    main()
