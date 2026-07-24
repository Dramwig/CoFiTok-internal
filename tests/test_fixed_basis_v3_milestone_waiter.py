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
    checkpoint_steps = [integrity_step] if integrity_step is not None else []
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


@pytest.mark.parametrize("milestone", [2000, 3000, 4000])
def test_accepts_complete_intermediate_validation(milestone: int) -> None:
    waiter.validate_audit(
        progress_report(
            last_step=milestone,
            validation_events=milestone // 1000,
        ),
        milestone,
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


def test_accepts_verified_step_10000_checkpoint() -> None:
    waiter.validate_audit(
        progress_report(
            last_step=10000,
            validation_events=10,
            integrity_status="verified",
            integrity_step=10000,
        ),
        10000,
    )


def test_rejects_stale_checkpoint_at_step_10000() -> None:
    report = progress_report(
        last_step=10000,
        validation_events=10,
        integrity_status="verified",
        integrity_step=5000,
    )
    with pytest.raises(ValueError, match="step-10000 checkpoint is unavailable"):
        waiter.validate_audit(report, 10000)


def test_final_milestone_can_require_complete_status() -> None:
    report = progress_report(
        last_step=50000,
        validation_events=50,
        integrity_status="verified",
        integrity_step=50000,
    )
    with pytest.raises(ValueError, match="progress audit is not healthy"):
        waiter.validate_audit(
            report,
            50000,
            require_complete_final=True,
        )
    report["status"] = "complete"
    waiter.validate_audit(
        report,
        50000,
        require_complete_final=True,
    )


def test_validates_long_horizon_milestones() -> None:
    assert waiter.validate_milestones(
        [10000, 25000, 50000],
        expected_steps=50000,
        checkpoint_interval=5000,
        evaluation_interval=1000,
    ) == (10000, 25000, 50000)


@pytest.mark.parametrize(
    "milestones",
    [
        [10000, 10000],
        [10500],
        [55000],
    ],
)
def test_rejects_invalid_long_horizon_milestones(
    milestones: list[int],
) -> None:
    with pytest.raises(ValueError):
        waiter.validate_milestones(
            milestones,
            expected_steps=50000,
            checkpoint_interval=5000,
            evaluation_interval=1000,
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


def test_milestone_entry_preserves_report_identity(tmp_path: Path) -> None:
    report_path = tmp_path / "cofitok_progress_step_00001000.json"
    report_path.write_text("{}\n", encoding="utf-8")
    report = progress_report(last_step=1000, validation_events=1)
    entry = waiter.milestone_entry(report, report_path=report_path)
    assert entry["status"] == "pass"
    assert entry["report"] == report_path.as_posix()
    assert entry["report_bytes"] == report_path.stat().st_size
    assert entry["last_step"] == 1000
    assert entry["validation_event_count"] == 1
