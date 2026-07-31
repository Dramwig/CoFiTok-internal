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
        checkpoint_size_multiplier=1.25,
        sample_count=3,
        estimated_sample_bytes=10,
        additional_bytes=20,
        safety_margin_bytes=50,
        git={"revision": "a" * 40, "tracked_dirty": False},
        hostname="server",
        checked_at="2026-07-13T00:00:00+00:00",
    )


def test_storage_capacity_passes_at_exact_required_boundary() -> None:
    report = _report(free_bytes=350)

    assert report["status"] == "pass"
    assert report["schema_version"] == 2
    assert report["plan"]["reference_checkpoint_bytes_each"] == 100
    assert report["plan"]["checkpoint_bytes_each"] == 125
    assert report["plan"]["required_free_bytes"] == 350
    assert report["headroom_bytes"] == 0


def test_storage_capacity_fails_below_required_boundary() -> None:
    report = _report(free_bytes=349)

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


def test_storage_capacity_rounds_scaled_checkpoint_bytes_up() -> None:
    report = build_storage_capacity_report(
        stage="full_training",
        path=Path("."),
        total_bytes=10_000,
        used_bytes=1_000,
        free_bytes=9_000,
        checkpoint_count=2,
        checkpoint_bytes=101,
        checkpoint_size_multiplier=4.0,
        sample_count=0,
        estimated_sample_bytes=0,
        additional_bytes=0,
        safety_margin_bytes=0,
        git={},
        hostname="server",
        checked_at="2026-07-13T00:00:00+00:00",
    )

    assert report["plan"]["checkpoint_bytes_each"] == 404
    assert report["plan"]["checkpoint_reserve_bytes"] == 808


def test_storage_capacity_rejects_checkpoint_downscaling() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_storage_capacity_report(
            stage="full_training",
            path=Path("."),
            total_bytes=10_000,
            used_bytes=1_000,
            free_bytes=9_000,
            checkpoint_count=1,
            checkpoint_bytes=100,
            checkpoint_size_multiplier=0.99,
            sample_count=0,
            estimated_sample_bytes=0,
            additional_bytes=0,
            safety_margin_bytes=0,
            git={},
            hostname="server",
            checked_at="2026-07-13T00:00:00+00:00",
        )
