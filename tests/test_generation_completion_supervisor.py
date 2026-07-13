from __future__ import annotations

from pathlib import Path

import pytest

from scripts.classify_generation_pipeline_exit import classify_pipeline_exit
from scripts.write_generation_supervisor_status import build_status


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_classifies_only_recoverable_execution_stages_for_retry() -> None:
    for stage in (
        "scaling_training",
        "posteval_10pct",
        "full_training",
        "full_posteval",
        "inference_export",
    ):
        report = classify_pipeline_exit(
            {"stage": stage, "status": "failed"}, exit_code=137
        )
        assert report["decision"] == "retry"


def test_supervisor_never_retries_scientific_or_completion_gates() -> None:
    for stage in (
        "preconditions",
        "promotion_gate",
        "final_gate",
        "completion_audit",
    ):
        report = classify_pipeline_exit(
            {"stage": stage, "status": "failed"}, exit_code=1
        )
        assert report["decision"] == "stop"


def test_supervisor_never_retries_storage_capacity_failure() -> None:
    report = classify_pipeline_exit(
        {"stage": "full_training", "status": "failed"}, exit_code=78
    )

    assert report["decision"] == "stop"
    assert report["reason"] == "insufficient_storage_capacity"


def test_supervisor_requires_pipeline_pass_and_complete_stage() -> None:
    assert classify_pipeline_exit(
        {"stage": "complete", "status": "pass"}, exit_code=0
    )["decision"] == "pass"
    assert classify_pipeline_exit(
        {"stage": "full_training", "status": "running"}, exit_code=0
    )["decision"] == "retry"
    assert classify_pipeline_exit({}, exit_code=0)["decision"] == "stop"


def test_supervisor_status_enforces_retry_evidence() -> None:
    report = build_status(
        status="retrying",
        detail="retry",
        attempt=2,
        max_attempts=4,
        pipeline_stage="full_training",
        pipeline_status="failed",
        child_exit_code=137,
        next_retry_seconds=120,
        git_commit="a" * 40,
        hostname="server",
        updated_at="2026-07-12T00:00:00+00:00",
    )

    assert report["status"] == "retrying"
    assert report["attempt"] == 2
    assert report["next_retry_seconds"] == 120

    with pytest.raises(ValueError, match="positive delay"):
        build_status(
            status="retrying",
            detail="retry",
            attempt=1,
            max_attempts=4,
            pipeline_stage="full_training",
            pipeline_status="failed",
            child_exit_code=1,
            next_retry_seconds=None,
            git_commit="a" * 40,
            hostname="server",
            updated_at="2026-07-12T00:00:00+00:00",
        )


def test_supervisor_runbook_is_bounded_locked_and_uses_structured_classifier() -> None:
    runbook = (
        ROOT / "artifacts/runbooks/generation_completion_supervisor.sh"
    ).read_text(encoding="utf-8")

    assert "flock -n 8" in runbook
    assert "COFITOK_SUPERVISOR_MAX_ATTEMPTS:-4" in runbook
    assert "classify_generation_pipeline_exit.py" in runbook
    assert "write_generation_supervisor_status.py" in runbook
    assert "attempt<=MAX_ATTEMPTS" in runbook
    assert "delay > 900" in runbook
    assert 'bash "$PIPELINE"' in runbook
    assert "promotion_gate" not in runbook
    assert "final_gate" not in runbook
