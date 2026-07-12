from __future__ import annotations

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
