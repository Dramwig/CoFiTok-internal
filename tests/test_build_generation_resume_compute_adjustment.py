from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_generation_resume_compute_adjustment import (
    build_adjustment_report,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    return payload


def fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    canonical = tmp_path / "train_metrics.jsonl"
    canonical_rows = [
        {
            "step": 100,
            "samples_seen": 800,
            "elapsed_seconds": 1000.0,
            "cumulative_elapsed_seconds": 1000.0,
        },
        {
            "step": 110,
            "samples_seen": 880,
            "elapsed_seconds": 10.0,
            "cumulative_elapsed_seconds": 1010.0,
        },
        {
            "step": 120,
            "samples_seen": 960,
            "elapsed_seconds": 20.0,
            "cumulative_elapsed_seconds": 1020.0,
        },
        {
            "step": 130,
            "samples_seen": 1040,
            "elapsed_seconds": 30.0,
            "cumulative_elapsed_seconds": 1030.0,
        },
    ]
    write_jsonl(canonical, canonical_rows)
    orphan_rows = [
        {
            "step": 110,
            "samples_seen": 880,
            "elapsed_seconds": 1012.0,
            "cumulative_elapsed_seconds": 1012.0,
        },
        {
            "step": 120,
            "samples_seen": 960,
            "elapsed_seconds": 1022.0,
            "cumulative_elapsed_seconds": 1022.0,
        },
    ]
    orphan_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in orphan_rows
    ).encode()
    digest = hashlib.sha256(orphan_payload).hexdigest()
    orphan = tmp_path / f"train_metrics_orphaned_at_resume_00000100_{digest[:12]}.jsonl"
    orphan.write_bytes(orphan_payload)
    return canonical, orphan


def test_report_adds_orphaned_compute_without_changing_trajectory(
    tmp_path: Path,
) -> None:
    canonical, orphan = fixture_paths(tmp_path)

    report = build_adjustment_report(
        canonical_metrics=canonical,
        orphan_metrics=[orphan],
        effective_batch=8,
        continuity_end_step=130,
    )

    event = report["recovery_events"][0]
    summary = report["summary"]
    assert report["status"] == "pass"
    assert event["canonical_resume_base_seconds"] == 1000.0
    assert event["orphaned_compute_seconds_lower_bound"] == 22.0
    assert event["orphaned_optimizer_steps_lower_bound"] == 20
    assert event["orphaned_images_lower_bound"] == 160
    assert summary["matched_compute_report_must_add_adjustment"] is True
    assert summary["observed_adjusted_elapsed_seconds_lower_bound"] == 1052.0
    assert report["claim_boundary"]["changes_training_trajectory"] is False


def test_report_rejects_inconsistent_canonical_resume_base(tmp_path: Path) -> None:
    canonical, orphan = fixture_paths(tmp_path)
    rows = [json.loads(line) for line in canonical.read_text().splitlines()]
    rows[2]["cumulative_elapsed_seconds"] = 1021.0
    write_jsonl(canonical, rows)

    with pytest.raises(ValueError, match="inconsistent resume bases"):
        build_adjustment_report(
            canonical_metrics=canonical,
            orphan_metrics=[orphan],
            effective_batch=8,
            continuity_end_step=130,
        )


def test_report_rejects_orphan_filename_digest_drift(tmp_path: Path) -> None:
    canonical, orphan = fixture_paths(tmp_path)
    orphan.write_bytes(orphan.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="filename digest"):
        build_adjustment_report(
            canonical_metrics=canonical,
            orphan_metrics=[orphan],
            effective_batch=8,
            continuity_end_step=130,
        )
