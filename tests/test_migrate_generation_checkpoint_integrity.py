from __future__ import annotations

import pytest

from scripts.migrate_generation_checkpoint_integrity import migrate_training_report


def _report() -> dict:
    return {
        "training_complete": True,
        "completed_steps": 50_000,
        "target_steps": 50_000,
        "latest_checkpoint": {
            "checkpoint": "checkpoint_step_00050000.pt",
            "step": 50_000,
        },
    }


def _integrity() -> dict:
    return {
        "schema_version": 1,
        "checkpoint": "checkpoint_step_00050000.pt",
        "checkpoint_bytes": 123,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_format_version": 1,
        "step": 50_000,
    }


def test_migrate_training_report_binds_legacy_report_to_integrity() -> None:
    migrated = migrate_training_report(
        _report(),
        integrity=_integrity(),
        integrity_manifest="checkpoint_step_00050000.pt.integrity.json",
    )

    assert migrated["latest_checkpoint"]["checkpoint_sha256"] == "a" * 64
    assert migrated["latest_checkpoint"]["checkpoint_bytes"] == 123
    assert migrated["checkpoint_integrity_migration"]["checkpoint_bytes_unchanged"] is True
    assert migrated["checkpoint_integrity_migration"]["payload_exact_resume_fields_present"] is True


def test_migrate_training_report_rejects_incomplete_training() -> None:
    report = _report()
    report["training_complete"] = False
    with pytest.raises(ValueError, match="completed training"):
        migrate_training_report(
            report,
            integrity=_integrity(),
            integrity_manifest="checkpoint_step_00050000.pt.integrity.json",
        )
