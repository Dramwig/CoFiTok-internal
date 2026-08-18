from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ORPHAN_NAME = re.compile(
    r"^train_metrics_orphaned_at_resume_(\d{8})_([0-9a-f]{12})\.jsonl$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def read_jsonl_with_raw(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    rows: list[tuple[dict[str, Any], bytes]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append((row, raw))
    if not rows:
        raise ValueError(f"metrics file is empty: {path}")
    return rows


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    effective_batch: int,
    label: str,
) -> None:
    steps = [int(row.get("step", -1)) for row in rows]
    if not all(left < right for left, right in zip(steps, steps[1:])):
        raise ValueError(f"{label} steps are not strictly increasing")
    cumulative = [float(row.get("cumulative_elapsed_seconds", math.nan)) for row in rows]
    if not all(math.isfinite(value) for value in cumulative):
        raise ValueError(f"{label} cumulative elapsed is non-finite")
    if not all(left < right for left, right in zip(cumulative, cumulative[1:])):
        raise ValueError(f"{label} cumulative elapsed is not strictly increasing")
    for row in rows:
        step = int(row.get("step", -1))
        if int(row.get("samples_seen", -1)) != step * effective_batch:
            raise ValueError(f"{label} samples_seen binding differs")


def build_adjustment_report(
    *,
    canonical_metrics: Path,
    orphan_metrics: list[Path],
    effective_batch: int,
    continuity_end_step: int,
) -> dict[str, Any]:
    if effective_batch < 1 or continuity_end_step < 1:
        raise ValueError("effective batch and continuity end step must be positive")
    if not orphan_metrics:
        raise ValueError("at least one orphan metrics archive is required")

    canonical_raw = read_jsonl_with_raw(canonical_metrics)
    canonical_rows = [row for row, _ in canonical_raw]
    validate_rows(
        canonical_rows,
        effective_batch=effective_batch,
        label="canonical metrics",
    )
    canonical_by_step = {int(row["step"]): row for row in canonical_rows}
    if continuity_end_step not in canonical_by_step:
        raise ValueError("canonical metrics do not contain continuity end step")
    prefix_raw = b"".join(
        raw
        for row, raw in canonical_raw
        if int(row["step"]) <= continuity_end_step
    )
    prefix_rows = [
        row for row in canonical_rows if int(row["step"]) <= continuity_end_step
    ]

    events: list[dict[str, Any]] = []
    adjustment_seconds = 0.0
    adjustment_steps = 0
    adjustment_images = 0
    for orphan_path in sorted(orphan_metrics, key=lambda path: path.as_posix()):
        match = ORPHAN_NAME.match(orphan_path.name)
        if match is None:
            raise ValueError(f"orphan metrics filename is malformed: {orphan_path.name}")
        checkpoint_step = int(match.group(1))
        orphan_identity = file_identity(orphan_path)
        if not orphan_identity["sha256"].startswith(match.group(2)):
            raise ValueError("orphan filename digest does not match physical content")

        orphan_raw = read_jsonl_with_raw(orphan_path)
        orphan_rows = [row for row, _ in orphan_raw]
        validate_rows(
            orphan_rows,
            effective_batch=effective_batch,
            label=f"orphan metrics {orphan_path.name}",
        )
        orphan_steps = [int(row["step"]) for row in orphan_rows]
        if orphan_steps[0] <= checkpoint_step:
            raise ValueError("orphan metrics do not begin after the resume checkpoint")
        if checkpoint_step not in canonical_by_step:
            raise ValueError("canonical metrics omit the resume checkpoint row")
        if any(step not in canonical_by_step for step in orphan_steps):
            raise ValueError("canonical metrics omit an orphan replacement step")

        replacement_rows = [canonical_by_step[step] for step in orphan_steps]
        resume_bases = [
            float(row["cumulative_elapsed_seconds"]) - float(row["elapsed_seconds"])
            for row in replacement_rows
        ]
        resume_base = resume_bases[0]
        if not all(
            math.isclose(value, resume_base, rel_tol=0.0, abs_tol=1e-6)
            for value in resume_bases
        ):
            raise ValueError("canonical replacement rows have inconsistent resume bases")

        checkpoint_cumulative = float(
            canonical_by_step[checkpoint_step]["cumulative_elapsed_seconds"]
        )
        checkpoint_finalization_seconds = resume_base - checkpoint_cumulative
        if checkpoint_finalization_seconds < 0.0:
            raise ValueError("canonical resume base predates checkpoint accounting")
        orphan_end_cumulative = float(orphan_rows[-1]["cumulative_elapsed_seconds"])
        orphaned_seconds = orphan_end_cumulative - resume_base
        if orphaned_seconds <= 0.0:
            raise ValueError("orphaned compute adjustment is not positive")
        optimizer_steps = orphan_steps[-1] - checkpoint_step
        if optimizer_steps <= 0:
            raise ValueError("orphaned optimizer-step lower bound is not positive")
        images = optimizer_steps * effective_batch

        adjustment_seconds += orphaned_seconds
        adjustment_steps += optimizer_steps
        adjustment_images += images
        events.append(
            {
                "resume_checkpoint_step": checkpoint_step,
                "checkpoint_cumulative_elapsed_seconds": checkpoint_cumulative,
                "canonical_resume_base_seconds": resume_base,
                "checkpoint_finalization_seconds_in_canonical": (
                    checkpoint_finalization_seconds
                ),
                "orphan_metrics": orphan_identity,
                "orphan_first_step": orphan_steps[0],
                "orphan_last_step": orphan_steps[-1],
                "orphan_logged_row_count": len(orphan_rows),
                "orphan_end_cumulative_elapsed_seconds": orphan_end_cumulative,
                "orphaned_compute_seconds_lower_bound": orphaned_seconds,
                "orphaned_optimizer_steps_lower_bound": optimizer_steps,
                "orphaned_images_lower_bound": images,
                "orphaned_images_per_second": images / orphaned_seconds,
                "canonical_replacement_steps": orphan_steps,
                "canonical_replacement_resume_base_constant": True,
                "canonical_replacement_first_row": replacement_rows[0],
                "canonical_replacement_last_row": replacement_rows[-1],
            }
        )

    observed_last = canonical_rows[-1]
    observed_canonical_elapsed = float(observed_last["cumulative_elapsed_seconds"])
    return {
        "schema_version": 1,
        "role": "generation_resume_compute_adjustment",
        "status": "pass",
        "created_at": utc_now(),
        "claim_boundary": {
            "adjusts_compute_accounting_only": True,
            "changes_training_trajectory": False,
            "changes_checkpoint": False,
            "changes_quality_metrics": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
        },
        "canonical_metrics": {
            "path": str(canonical_metrics.resolve()),
            "observed_file": file_identity(canonical_metrics),
            "observed_last_step": int(observed_last["step"]),
            "observed_canonical_elapsed_seconds": observed_canonical_elapsed,
            "continuity_prefix": {
                "end_step": continuity_end_step,
                "row_count": len(prefix_rows),
                "bytes": len(prefix_raw),
                "sha256": sha256_bytes(prefix_raw),
                "steps_strictly_increasing": True,
                "cumulative_elapsed_strictly_increasing": True,
                "samples_seen_binding_verified": True,
            },
        },
        "recovery_events": events,
        "summary": {
            "event_count": len(events),
            "orphaned_compute_seconds_lower_bound": adjustment_seconds,
            "orphaned_compute_hours_lower_bound": adjustment_seconds / 3600.0,
            "orphaned_optimizer_steps_lower_bound": adjustment_steps,
            "orphaned_images_lower_bound": adjustment_images,
            "observed_adjusted_elapsed_seconds_lower_bound": (
                observed_canonical_elapsed + adjustment_seconds
            ),
            "observed_adjusted_elapsed_hours_lower_bound": (
                observed_canonical_elapsed + adjustment_seconds
            )
            / 3600.0,
            "final_cost_formula": (
                "training_report.cumulative_elapsed_seconds + "
                "orphaned_compute_seconds_lower_bound"
            ),
            "matched_compute_report_must_add_adjustment": True,
            "reason": (
                "canonical exact-resume metrics correctly exclude rolled-back work, "
                "but physical compute already consumed by orphaned rows must remain "
                "visible in the CoFiTok cost row"
            ),
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound adjustment for compute consumed by metrics rows "
            "orphaned during exact resume."
        )
    )
    parser.add_argument("--canonical-metrics", type=Path, required=True)
    parser.add_argument(
        "--orphan-metrics",
        type=Path,
        action="append",
        default=[],
        required=True,
    )
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--continuity-end-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_adjustment_report(
        canonical_metrics=args.canonical_metrics,
        orphan_metrics=args.orphan_metrics,
        effective_batch=args.effective_batch,
        continuity_end_step=args.continuity_end_step,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
