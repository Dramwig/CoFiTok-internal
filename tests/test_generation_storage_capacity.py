from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_generation_storage_capacity import build_storage_capacity_report


def _report(*, free_bytes: int) -> dict:
    return build_storage_capacity_report(
        stage="full_training",
        path=Path("."),
        total_bytes=10_000,
        used_bytes=1_000,
        free_bytes=free_bytes,
        checkpoint_count=2,
        checkpoint_bytes=100,
        sample_count=3,
        estimated_sample_bytes=10,
        additional_bytes=20,
        safety_margin_bytes=50,
        git={"revision": "a" * 40, "tracked_dirty": False},
        hostname="server",
        checked_at="2026-07-13T00:00:00+00:00",
    )


def test_storage_capacity_passes_at_exact_required_boundary() -> None:
    report = _report(free_bytes=300)

    assert report["status"] == "pass"
    assert report["plan"]["required_free_bytes"] == 300
    assert report["headroom_bytes"] == 0


def test_storage_capacity_fails_below_required_boundary() -> None:
    report = _report(free_bytes=299)

    assert report["status"] == "fail"
    assert report["headroom_bytes"] == -1


def test_storage_capacity_rejects_inconsistent_filesystem_usage() -> None:
    with pytest.raises(ValueError, match="exceeds total"):
        build_storage_capacity_report(
            stage="full_posteval",
            path=Path("."),
            total_bytes=100,
            used_bytes=80,
            free_bytes=30,
            checkpoint_count=0,
            checkpoint_bytes=0,
            sample_count=0,
            estimated_sample_bytes=0,
            additional_bytes=0,
            safety_margin_bytes=0,
            git={},
            hostname="server",
            checked_at="2026-07-13T00:00:00+00:00",
        )
