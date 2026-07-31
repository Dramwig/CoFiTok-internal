from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.run_generation_stability_50k_posteval_waiter import (
    validate_monitor,
    validate_pair_summary,
)


TRAINING_REVISION = "1" * 40
TRAINING_BRANCH = "scale/generation-stability-50k-preflight"
MONITOR_NAME = "generation_stability_ema_teacher_matched_50k"


def _monitor(*, status: str = "pass", stage: str = "complete") -> dict:
    return {
        "schema_version": 1,
        "monitor": MONITOR_NAME,
        "status": status,
        "stage": stage,
        "issues": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        },
    }


def _summary() -> dict:
    source = {"path": "/tmp/source.json", "bytes": 1, "sha256": "a" * 64}
    return {
        "schema_version": 1,
        "status": "completed",
        "stage": "stability_matched_50k",
        "completed_steps_per_method": 50_000,
        "images_seen_per_method": 3_200_000,
        "effective_batch_size": 64,
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
        },
        "training_pair": {"status": "pass"},
        "formal_300k_authorization_allowed": False,
        "formal_ema_sampling_gate_required": True,
        "sources": {
            "cofitok_training": source,
            "dense_training": source,
            "decision_validation": source,
            "config_validation": source,
        },
    }


def test_waiter_accepts_bound_completed_monitor_and_pair() -> None:
    monitor = validate_monitor(
        _monitor(),
        expected_name=MONITOR_NAME,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )
    summary = validate_pair_summary(
        _summary(),
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )

    assert monitor["status"] == "pass"
    assert monitor["stage"] == "complete"
    assert summary["status"] == "verified"


def test_waiter_rejects_stale_running_monitor() -> None:
    payload = _monitor(status="running", stage="cofitok_training")
    now = datetime.now(timezone.utc)
    payload["updated_at"] = (now - timedelta(seconds=901)).isoformat()

    with pytest.raises(RuntimeError, match="stale"):
        validate_monitor(
            payload,
            expected_name=MONITOR_NAME,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            now=now,
            silence_seconds=900,
        )


def test_waiter_rejects_monitor_revision_mismatch() -> None:
    payload = _monitor()
    payload["git"]["revision"] = "2" * 40

    with pytest.raises(ValueError, match="Git identity"):
        validate_monitor(
            payload,
            expected_name=MONITOR_NAME,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "failed"),
        ("completed_steps_per_method", 49_999),
        ("formal_300k_authorization_allowed", True),
        ("formal_ema_sampling_gate_required", False),
    ],
)
def test_waiter_rejects_incomplete_or_promoting_pair(field: str, value: object) -> None:
    payload = _summary()
    payload[field] = value

    with pytest.raises(ValueError, match="contract mismatch"):
        validate_pair_summary(
            payload,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )


def test_waiter_runbook_is_non_promoting_and_identity_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/"
        "generation_stability_ema_teacher_50k_posteval_waiter.sh"
    ).read_text(encoding="utf-8")

    assert "EXPECTED_TRAINING_REVISION" in source
    assert "EXPECTED_TRAINING_BRANCH" in source
    assert "EXPECTED_TARGET_REVISION" in source
    assert "EXPECTED_TARGET_BRANCH" in source
    assert "run_generation_stability_50k_posteval_waiter.py" in source
    assert "generation_full_matched_300k_after_gate.sh" not in source
    assert "train_generation.py" not in source
