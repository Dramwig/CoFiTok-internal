from __future__ import annotations

import hashlib
import json

from scripts.audit_generation_training_progress import audit_progress


def _write_metrics(run_dir, steps, elapsed=None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    elapsed = elapsed or [float(step * 2) for step in steps]
    rows = [
        {
            "step": step,
            "total": 0.1,
            "epsilon": 0.08,
            "grad_norm": 0.2,
            "learning_rate": 1e-4,
            "elapsed_seconds": seconds,
        }
        for step, seconds in zip(steps, elapsed)
    ]
    (run_dir / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_checkpoint_with_integrity(run_dir, step: int, payload: bytes = b"checkpoint"):
    checkpoint = run_dir / f"checkpoint_step_{step:08d}.pt"
    checkpoint.write_bytes(payload)
    integrity = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_format_version": 1,
        "step": step,
    }
    integrity_name = f"{checkpoint.name}.integrity.json"
    (run_dir / integrity_name).write_text(json.dumps(integrity), encoding="utf-8")
    (run_dir / "latest.json").write_text(
        json.dumps({**integrity, "integrity_manifest": integrity_name}),
        encoding="utf-8",
    )
    return checkpoint


def test_audit_reports_healthy_before_first_checkpoint(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 50, 100, 450])

    report = audit_progress(tmp_path, expected_steps=1_000, checkpoint_interval=500)

    assert report["status"] == "healthy"
    assert report["checkpoint"]["status"] == "not_due"
    assert report["seconds_per_step"] == 2.0
    assert report["eta_seconds"] == 1_100.0
    assert report["gradient_clipping"]["logged_norm_semantics"] == "pre_clip_total_norm"
    assert report["gradient_clipping"]["pre_clip_exceedance_count"] == 0


def test_audit_validates_latest_checkpoint_pointer(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 550])
    checkpoint = tmp_path / "checkpoint_step_00000500.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint.name, "step": 500}),
        encoding="utf-8",
    )

    report = audit_progress(tmp_path, expected_steps=1_000, checkpoint_interval=500)

    assert report["status"] == "healthy"
    assert report["checkpoint"]["status"] == "available"
    assert report["checkpoint"]["steps"] == [500]
    assert report["checkpoint"]["latest_integrity"]["status"] == "legacy_computed"
    assert report["checkpoint"]["latest_integrity"]["checkpoint_bytes"] == 10
    assert len(report["checkpoint"]["latest_integrity"]["checkpoint_sha256"]) == 64
    assert len(report["warnings"]) == 1


def test_audit_requires_and_verifies_checkpoint_integrity(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 550])
    _write_checkpoint_with_integrity(tmp_path, 500)

    report = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=500,
        integrity_policy="required",
    )

    assert report["status"] == "healthy"
    assert report["schema_version"] == 2
    integrity = report["checkpoint"]["latest_integrity"]
    assert integrity["status"] == "verified"
    assert integrity["checkpoint_bytes"] == 10
    assert report["warnings"] == []


def test_audit_rejects_missing_required_checkpoint_integrity(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 550])
    checkpoint = tmp_path / "checkpoint_step_00000500.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint.name, "step": 500}),
        encoding="utf-8",
    )

    report = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=500,
        integrity_policy="required",
    )

    assert report["status"] == "invalid"
    assert report["checkpoint"]["latest_integrity"]["status"] == "missing_manifest"


def test_audit_rejects_checkpoint_tampering(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 550])
    checkpoint = _write_checkpoint_with_integrity(tmp_path, 500)
    checkpoint.write_bytes(b"changed-checkpoint")

    report = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=500,
        integrity_policy="required",
    )

    assert report["status"] == "invalid"
    assert report["checkpoint"]["latest_integrity"]["status"] == "invalid"
    assert "Checkpoint size mismatch" in report["checkpoint"]["latest_integrity"]["error"]


def test_audit_rejects_nonmonotonic_metrics_and_overdue_checkpoint(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 550, 500, 600])

    report = audit_progress(tmp_path, expected_steps=1_000, checkpoint_interval=500)

    assert report["status"] == "invalid"
    assert "metrics steps are not strictly increasing" in report["issues"]
    assert "checkpoint for step 500 is overdue" in report["issues"]


def test_audit_uses_latest_elapsed_segment_after_resume(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 550, 600], elapsed=[2.0, 1_000.0, 100.0, 200.0])
    checkpoint = tmp_path / "checkpoint_step_00000500.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint.name, "step": 500}),
        encoding="utf-8",
    )

    report = audit_progress(tmp_path, expected_steps=1_000, checkpoint_interval=500)

    assert report["status"] == "healthy"
    assert report["current_segment_start_step"] == 550
    assert report["seconds_per_step"] == 2.0


def test_audit_warns_when_scheduled_validation_is_not_logged(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 100, 200])

    report = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=500,
        evaluation_interval=100,
    )

    assert report["status"] == "healthy"
    assert report["validation"] == {
        "configured_interval": 100,
        "event_count": 0,
        "expected_event_count": 2,
        "logging_complete": False,
    }
    assert len(report["warnings"]) == 1


def test_audit_accepts_complete_validation_logging(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 100, 200])
    metrics_path = tmp_path / "train_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["validation_epsilon_mse"] = 0.09
    rows[2]["validation_epsilon_mse"] = 0.08
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=500,
        evaluation_interval=100,
    )

    assert report["validation"]["logging_complete"] is True
    assert report["warnings"] == []


def test_audit_requires_reached_protected_checkpoints(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 1_000])
    latest = tmp_path / "checkpoint_step_00001000.pt"
    latest.write_bytes(b"checkpoint")
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": latest.name, "step": 1_000}),
        encoding="utf-8",
    )

    report = audit_progress(
        tmp_path,
        expected_steps=2_000,
        checkpoint_interval=500,
        required_checkpoint_steps=[500, 1_000, 2_000],
    )

    assert report["status"] == "invalid"
    assert report["checkpoint"]["missing_required_steps"] == [500]
    assert "required checkpoints are missing: 500" in report["issues"]


def test_audit_does_not_require_future_protected_checkpoint(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500])
    checkpoint = tmp_path / "checkpoint_step_00000500.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint.name, "step": 500}),
        encoding="utf-8",
    )

    report = audit_progress(
        tmp_path,
        expected_steps=2_000,
        checkpoint_interval=500,
        required_checkpoint_steps=[500, 1_000, 2_000],
    )

    assert report["status"] == "healthy"
    assert report["checkpoint"]["missing_required_steps"] == []
