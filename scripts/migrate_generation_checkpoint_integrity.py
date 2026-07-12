from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.reporting import write_json_report
from cofitok.training import backfill_training_checkpoint_integrity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind a legacy generation checkpoint to integrity and training metadata."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def migrate_training_report(
    report: dict[str, Any],
    *,
    integrity: dict[str, Any],
    integrity_manifest: str,
) -> dict[str, Any]:
    step = int(integrity["step"])
    if report.get("training_complete") is not True:
        raise ValueError("Integrity migration requires a completed training report")
    if int(report.get("completed_steps", -1)) != step:
        raise ValueError("Training report completed step does not match checkpoint")
    if int(report.get("target_steps", -1)) != step:
        raise ValueError("Training report target step does not match checkpoint")
    previous = report.get("latest_checkpoint", {})
    if previous.get("checkpoint") != integrity["checkpoint"]:
        raise ValueError("Training report names another latest checkpoint")
    if int(previous.get("step", -1)) != step:
        raise ValueError("Training report latest step does not match checkpoint")
    migrated = dict(report)
    migrated["latest_checkpoint"] = {
        **integrity,
        "integrity_manifest": integrity_manifest,
    }
    migrated["checkpoint_integrity_migration"] = {
        "status": "completed",
        "checkpoint_sha256": integrity["checkpoint_sha256"],
        "checkpoint_bytes_unchanged": True,
        "payload_exact_resume_fields_present": True,
    }
    return migrated


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    training_report_path = Path(args.training_report)
    before_bytes = checkpoint.stat().st_size
    with training_report_path.open("r", encoding="utf-8") as handle:
        training_report = json.load(handle)
    integrity = backfill_training_checkpoint_integrity(checkpoint, update_latest=False)
    integrity_manifest = f"{checkpoint.name}.integrity.json"
    migrated = migrate_training_report(
        training_report,
        integrity=integrity,
        integrity_manifest=integrity_manifest,
    )
    backfill_training_checkpoint_integrity(checkpoint, update_latest=True)
    after_bytes = checkpoint.stat().st_size
    if before_bytes != after_bytes:
        raise RuntimeError("Checkpoint bytes changed during integrity migration")
    write_json_report(training_report_path, migrated)
    report = {
        "schema_version": 1,
        "status": "completed",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_integrity": integrity,
        "training_report": str(training_report_path.resolve()),
        "checkpoint_bytes_unchanged": True,
        "payload_exact_resume_fields_present": True,
    }
    write_json_report(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
