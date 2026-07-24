from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "operations"
    / "generation"
    / "fixed_basis_v3_milestone_waiter.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fixed_basis_v3_milestone_waiter", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
waiter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(waiter)


def progress_report(
    *,
    last_step: int,
    validation_events: int,
    integrity_status: str = "not_available",
    integrity_step: int | None = None,
) -> dict:
    checkpoint_steps = [5000] if integrity_step == 5000 else []
    return {
        "status": "healthy",
        "last_step": last_step,
        "issues": [],
        "warnings": [],
        "validation": {
            "event_count": validation_events,
            "logging_complete": True,
        },
        "checkpoint": {
            "steps": checkpoint_steps,
            "missing_required_steps": [],
            "latest_integrity": {
                "status": integrity_status,
                "step": integrity_step,
            },
        },
    }


def test_accepts_complete_step_1000_validation() -> None:
    waiter.validate_audit(
        progress_report(last_step=1000, validation_events=1),
        1000,
    )


def test_rejects_incomplete_step_1000_validation() -> None:
    with pytest.raises(ValueError, match="validation evidence is incomplete"):
        waiter.validate_audit(
            progress_report(last_step=1000, validation_events=0),
            1000,
        )


def test_rejects_progress_warnings() -> None:
    report = progress_report(last_step=1000, validation_events=1)
    report["warnings"] = ["validation row is unavailable"]
    with pytest.raises(ValueError, match="issues or warnings"):
        waiter.validate_audit(report, 1000)


def test_accepts_verified_step_5000_checkpoint() -> None:
    waiter.validate_audit(
        progress_report(
            last_step=5000,
            validation_events=5,
            integrity_status="verified",
            integrity_step=5000,
        ),
        5000,
    )


@pytest.mark.parametrize(
    ("integrity_status", "integrity_step"),
    [
        ("missing_manifest", 5000),
        ("invalid", 5000),
        ("verified", 4999),
    ],
)
def test_rejects_unverified_step_5000_checkpoint(
    integrity_status: str,
    integrity_step: int,
) -> None:
    report = progress_report(
        last_step=5000,
        validation_events=5,
        integrity_status=integrity_status,
        integrity_step=5000,
    )
    report["checkpoint"]["latest_integrity"]["step"] = integrity_step
    with pytest.raises(ValueError):
        waiter.validate_audit(report, 5000)
