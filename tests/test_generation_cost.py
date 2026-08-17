from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cofitok.generation_cost import (
    resume_compute_adjustment_source_identity,
    training_cost_summary,
    verify_resume_compute_adjustment_source,
)
from scripts.build_generation_resume_compute_adjustment import (
    build_adjustment_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _report(
    output_dir: Path,
    *,
    target_steps: int,
    elapsed_seconds: float,
    reconciliation: dict[str, object] | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "output_dir": str(output_dir.resolve()),
        "config": {
            "data": {"batch_size": 4},
            "optimization": {"gradient_accumulation_steps": 2},
            "runtime": {"device": "cuda"},
        },
        "completed_steps": target_steps,
        "target_steps": target_steps,
        "final_metrics": {"samples_seen": target_steps * 8},
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_bytes": 1024,
    }
    if reconciliation is not None:
        report["metrics_resume_reconciliation"] = reconciliation
    return report


def _recovery_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path, Path, Path]:
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
    _write_jsonl(canonical, canonical_rows)
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
    ).encode("utf-8")
    orphan_sha256 = hashlib.sha256(orphan_payload).hexdigest()
    orphan = tmp_path / (
        "train_metrics_orphaned_at_resume_00000100_"
        f"{orphan_sha256[:12]}.jsonl"
    )
    orphan.write_bytes(orphan_payload)
    adjustment = build_adjustment_report(
        canonical_metrics=canonical,
        orphan_metrics=[orphan],
        effective_batch=8,
        continuity_end_step=130,
    )
    adjustment_path = tmp_path / "resume_compute_adjustment.json"
    adjustment_path.write_text(
        json.dumps(adjustment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reconciliation: dict[str, object] = {
        "schema_version": 1,
        "status": "reconciled",
        "resume_step": 100,
        "metrics": str(canonical.resolve()),
        "retained_rows": 1,
        "orphaned_rows": 2,
        "orphan_archive": str(orphan.resolve()),
        "orphan_sha256": orphan_sha256,
    }
    return (
        _report(
            tmp_path,
            target_steps=130,
            elapsed_seconds=1030.0,
            reconciliation=reconciliation,
        ),
        adjustment,
        adjustment_path,
        canonical,
        orphan,
    )


def test_legacy_training_cost_without_recovery_is_unchanged(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path, target_steps=10, elapsed_seconds=20.0)

    cost = training_cost_summary(report)

    assert cost["valid"] is True
    assert cost["reported_elapsed_seconds"] == 20.0
    assert cost["elapsed_seconds"] == 20.0
    assert cost["images_per_second"] == 4.0
    assert cost["resume_compute_adjustment"]["required"] is False
    assert cost["elapsed_seconds_role"] == "reported_training_elapsed_seconds"


def test_reconciled_or_discovered_orphan_compute_requires_adjustment(
    tmp_path: Path,
) -> None:
    report, _, _, _, _ = _recovery_fixture(tmp_path)

    cost = training_cost_summary(report)

    assert cost["valid"] is False
    assert cost["resume_compute_adjustment"]["required"] is True
    assert cost["resume_compute_adjustment"]["discovered_orphan_archive_count"] == 1
    assert cost["resume_compute_adjustment"]["issues"] == [
        "training orphaned compute lacks an adjustment report"
    ]

    report["metrics_resume_reconciliation"] = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 130,
        "metrics": str((tmp_path / "train_metrics.jsonl").resolve()),
        "retained_rows": 4,
        "orphaned_rows": 0,
        "orphan_archive": None,
        "orphan_sha256": None,
    }
    later_cost = training_cost_summary(report)
    assert later_cost["valid"] is False
    assert later_cost["resume_compute_adjustment"]["required"] is True


def test_adjustment_adds_physical_recovery_compute_lower_bound(
    tmp_path: Path,
) -> None:
    report, adjustment, adjustment_path, _, _ = _recovery_fixture(tmp_path)

    cost = training_cost_summary(report, adjustment)
    verified = verify_resume_compute_adjustment_source(
        resume_compute_adjustment_source_identity(adjustment_path),
        method="cofitok",
        training_report=report,
    )

    assert cost["valid"] is True
    assert cost["reported_elapsed_seconds"] == 1030.0
    assert cost["resume_compute_adjustment"]["seconds"] == 22.0
    assert cost["resume_compute_adjustment"][
        "orphaned_optimizer_steps_lower_bound"
    ] == 20
    assert cost["resume_compute_adjustment"]["orphaned_images_lower_bound"] == 160
    assert cost["elapsed_seconds"] == 1052.0
    assert cost["images_per_second"] == pytest.approx(1040 / 1052.0)
    assert cost["elapsed_seconds_role"] == (
        "physical_lower_bound_including_orphaned_recovery_compute"
    )
    assert verified["status"] == "verified"
    assert verified["summary"] == cost["resume_compute_adjustment"]


def test_adjustment_verification_rejects_canonical_prefix_drift(
    tmp_path: Path,
) -> None:
    report, _, adjustment_path, canonical, _ = _recovery_fixture(tmp_path)
    rows = [json.loads(line) for line in canonical.read_text().splitlines()]
    rows[1]["epsilon"] = 999.0
    _write_jsonl(canonical, rows)

    with pytest.raises(ValueError, match="continuity prefix changed"):
        verify_resume_compute_adjustment_source(
            resume_compute_adjustment_source_identity(adjustment_path),
            method="cofitok",
            training_report=report,
        )


def test_adjustment_verification_rejects_orphan_archive_drift(
    tmp_path: Path,
) -> None:
    report, _, adjustment_path, _, orphan = _recovery_fixture(tmp_path)
    orphan.write_bytes(orphan.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="orphan metrics identity changed"):
        verify_resume_compute_adjustment_source(
            resume_compute_adjustment_source_identity(adjustment_path),
            method="cofitok",
            training_report=report,
        )
