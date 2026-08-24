from __future__ import annotations

import hashlib
import json

import pytest

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


def test_audit_explicitly_accepts_a_bound_stale_pause_report_after_resume(
    tmp_path,
) -> None:
    _write_metrics(tmp_path, [1, 500, 550])
    (tmp_path / "training_report.json").write_text(
        json.dumps(
            {
                "completed_steps": 500,
                "target_steps": 1_000,
                "training_complete": False,
                "stop_requested": True,
                "latest_checkpoint": {"step": 500},
            }
        ),
        encoding="utf-8",
    )

    strict = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=1_000,
    )
    resumed = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=1_000,
        allow_stale_incomplete_training_report=True,
    )

    assert strict["status"] == "invalid"
    assert resumed["status"] == "healthy"
    assert resumed["issues"] == []
    assert resumed["training_report"]["status"] == (
        "stale_incomplete_resume_report"
    )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("training_complete", True),
        ("stop_requested", False),
        ("target_steps", 2_000),
        ("latest_checkpoint", {"step": 499}),
    ],
)
def test_audit_does_not_mask_an_unbound_stale_training_report(
    tmp_path, mutation, value
) -> None:
    _write_metrics(tmp_path, [1, 500, 550])
    report = {
        "completed_steps": 500,
        "target_steps": 1_000,
        "training_complete": False,
        "stop_requested": True,
        "latest_checkpoint": {"step": 500},
    }
    report[mutation] = value
    (tmp_path / "training_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    audited = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=1_000,
        allow_stale_incomplete_training_report=True,
    )

    assert audited["status"] == "invalid"
    assert audited["training_report"]["status"] == "mismatched"
    assert "training report completed_steps does not match metrics" in audited["issues"]


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
        "provenance_metadata_status": "not_applicable",
        "provenance_fields": [
            "validation_event_index",
            "validation_batch_index",
            "validation_num_images",
            "validation_noise_seed",
        ],
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
    assert report["validation"]["provenance_metadata_status"] == "legacy_absent"
    assert report["warnings"] == []


def test_audit_validates_complete_validation_provenance(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 100, 200])
    metrics_path = tmp_path / "train_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    for event_index, row in enumerate(rows[1:]):
        row.update(
            {
                "validation_epsilon_mse": 0.09 - event_index * 0.01,
                "validation_event_index": event_index,
                "validation_batch_index": event_index,
                "validation_num_images": 16,
                "validation_noise_seed": 102_030,
            }
        )
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

    assert report["status"] == "healthy"
    assert report["validation"]["provenance_metadata_status"] == "complete"
    assert report["issues"] == []


def test_audit_rejects_noncontiguous_validation_event_indices(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 100, 200])
    metrics_path = tmp_path / "train_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    for event_index, row in enumerate(rows[1:]):
        row.update(
            {
                "validation_epsilon_mse": 0.09,
                "validation_event_index": event_index + 1,
                "validation_batch_index": event_index,
                "validation_num_images": 16,
                "validation_noise_seed": 102_030,
            }
        )
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

    assert report["status"] == "invalid"
    assert "validation event indices are not contiguous from zero" in report["issues"]


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


def test_audit_physically_verifies_every_reached_protected_checkpoint(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 750, 1_000])
    expected = {}
    for step in (500, 750, 1_000):
        payload = f"checkpoint-{step}".encode()
        _write_checkpoint_with_integrity(tmp_path, step, payload)
        expected[step] = hashlib.sha256(payload).hexdigest()

    report = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=250,
        required_checkpoint_steps=[500, 750, 1_000],
        integrity_policy="required",
    )

    required = report["checkpoint"]["required_integrity"]
    assert report["status"] == "healthy"
    assert required["status"] == "verified"
    assert required["reached_steps"] == [500, 750, 1_000]
    assert [item["step"] for item in required["checkpoints"]] == [500, 750, 1_000]
    assert all(item["status"] == "verified" for item in required["checkpoints"])
    assert {
        item["step"]: item["checkpoint_sha256"] for item in required["checkpoints"]
    } == expected
    assert required["checkpoints"][-1]["latest_binding"] == "verified"
    assert all(
        item["latest_binding"] == "not_applicable"
        for item in required["checkpoints"][:-1]
    )


def test_audit_rejects_tampered_historical_protected_checkpoint(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 1_000])
    historical = _write_checkpoint_with_integrity(tmp_path, 500)
    _write_checkpoint_with_integrity(tmp_path, 1_000)
    historical.write_bytes(b"tampered-historical-checkpoint")

    report = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=500,
        required_checkpoint_steps=[500, 1_000],
        integrity_policy="required",
    )

    required = report["checkpoint"]["required_integrity"]
    assert report["status"] == "invalid"
    assert required["status"] == "invalid"
    assert required["checkpoints"][0]["status"] == "invalid"
    assert any(
        issue.startswith("required checkpoint step 500 integrity verification failed")
        for issue in report["issues"]
    )


def test_audit_rejects_missing_historical_protected_sidecar(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 500, 1_000])
    historical = _write_checkpoint_with_integrity(tmp_path, 500)
    _write_checkpoint_with_integrity(tmp_path, 1_000)
    (tmp_path / f"{historical.name}.integrity.json").unlink()

    report = audit_progress(
        tmp_path,
        expected_steps=1_000,
        checkpoint_interval=500,
        required_checkpoint_steps=[500, 1_000],
        integrity_policy="required",
    )

    required = report["checkpoint"]["required_integrity"]
    assert report["status"] == "invalid"
    assert required["status"] == "invalid"
    assert required["checkpoints"][0]["status"] == "missing_manifest"
    assert any(
        issue.startswith("required checkpoint step 500 integrity manifest is missing")
        for issue in report["issues"]
    )


def _write_schedule_config(
    path,
    *,
    rollout_weight: float = 0.1,
    teacher_weight: float = 0.25,
    ranking_weight: float = 0.0,
    residual_weight: float = 0.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "schedule-audit",
                "data": {},
                "diffusion": {},
                "model": {},
                "loss": {
                    "rollout_consistency_weight": rollout_weight,
                    "rollout_consistency_start_step": 0,
                    "rollout_consistency_warmup_steps": 1_000,
                    "ema_teacher_consistency_weight": teacher_weight,
                    "ema_teacher_consistency_start_step": 2_000,
                    "ema_teacher_consistency_warmup_steps": 1_000,
                    "class_conditioning_ranking_weight": ranking_weight,
                    "class_conditioning_ranking_start_step": 1_000,
                    "class_conditioning_ranking_warmup_steps": 1_000,
                    "class_conditioning_residual_alignment_weight": residual_weight,
                    "class_conditioning_residual_alignment_start_step": 1_000,
                    "class_conditioning_residual_alignment_warmup_steps": 1_000,
                },
                "runtime": {},
                "optimization": {},
            }
        ),
        encoding="utf-8",
    )


def _write_schedule_metrics(
    run_dir,
    *,
    corrupt: str = "",
    include_ranking: bool = False,
    include_residual: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in (1, 500, 1_000, 2_000, 2_500, 3_000):
        rollout_scale = min(step / 1_000, 1.0)
        teacher_scale = 0.0 if step <= 2_000 else min((step - 2_000) / 1_000, 1.0)
        rows.append(
            {
                "step": step,
                "total": 0.1,
                "epsilon": 0.08,
                "grad_norm": 0.2,
                "learning_rate": 1e-4,
                "elapsed_seconds": float(step * 2),
                "rollout_consistency_scale": rollout_scale,
                "rollout_consistency": 0.0 if rollout_scale == 0.0 else 0.02,
                "ema_teacher_consistency_scale": teacher_scale,
                "ema_teacher_consistency": 0.0 if teacher_scale == 0.0 else 0.01,
                **(
                    {
                        "class_conditioning_ranking_scale": (
                            0.0
                            if step <= 1_000
                            else min((step - 1_000) / 1_000, 1.0)
                        ),
                        "class_conditioning_ranking": 0.0,
                    }
                    if include_ranking
                    else {}
                ),
                **(
                    {
                        "class_conditioning_residual_alignment_scale": (
                            0.0
                            if step <= 1_000
                            else min((step - 1_000) / 1_000, 1.0)
                        ),
                        "class_conditioning_residual_alignment": (
                            0.0 if step <= 1_000 else 0.03
                        ),
                    }
                    if include_residual
                    else {}
                ),
            }
        )
    if corrupt == "rollout_scale":
        rows[1]["rollout_consistency_scale"] = 0.75
    elif corrupt == "teacher_early_loss":
        rows[2]["ema_teacher_consistency"] = 0.5
    elif corrupt == "rollout_active_zero":
        for row in rows:
            row["rollout_consistency"] = 0.0
    elif corrupt == "residual_active_zero":
        for row in rows:
            row["class_conditioning_residual_alignment"] = 0.0
    (run_dir / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_audit_verifies_consistency_schedules_against_config(tmp_path) -> None:
    config = tmp_path / "config.json"
    _write_schedule_config(config)
    _write_schedule_metrics(tmp_path)
    report = audit_progress(
        tmp_path,
        expected_steps=4_000,
        checkpoint_interval=5_000,
        config_path=config,
    )

    assert report["status"] == "healthy"
    schedules = report["consistency_schedules"]
    assert schedules["status"] == "verified"
    assert len(schedules["config_sha256"]) == 64
    assert schedules["schedules"]["rollout_consistency"][
        "expected_scale_at_last_step"
    ] == 1.0
    assert schedules["schedules"]["ema_teacher_consistency"][
        "expected_scale_at_last_step"
    ] == 1.0
    assert schedules["schedules"]["rollout_consistency"]["active_rows"] == 6
    assert schedules["schedules"]["rollout_consistency"]["nonzero_loss_rows"] == 6


def test_audit_rejects_consistency_schedule_drift(tmp_path) -> None:
    config = tmp_path / "config.json"
    _write_schedule_config(config)
    _write_schedule_metrics(tmp_path, corrupt="rollout_scale")
    report = audit_progress(
        tmp_path,
        expected_steps=4_000,
        checkpoint_interval=5_000,
        config_path=config,
    )

    assert report["status"] == "invalid"
    assert "row 1 rollout_consistency_scale differs from config schedule" in report["issues"]


def test_audit_rejects_teacher_loss_before_schedule_start(tmp_path) -> None:
    config = tmp_path / "config.json"
    _write_schedule_config(config)
    _write_schedule_metrics(tmp_path, corrupt="teacher_early_loss")
    report = audit_progress(
        tmp_path,
        expected_steps=4_000,
        checkpoint_interval=5_000,
        config_path=config,
    )

    assert report["status"] == "invalid"
    assert "row 2 ema_teacher_consistency is nonzero while disabled" in report["issues"]


def test_audit_rejects_consistency_loss_that_is_zero_after_activation(tmp_path) -> None:
    config = tmp_path / "config.json"
    _write_schedule_config(config)
    _write_schedule_metrics(tmp_path, corrupt="rollout_active_zero")
    report = audit_progress(
        tmp_path,
        expected_steps=4_000,
        checkpoint_interval=5_000,
        config_path=config,
    )

    assert report["status"] == "invalid"
    assert "rollout_consistency is zero for all active schedule rows" in report["issues"]


def test_audit_accepts_zero_hinge_after_ranking_activation(tmp_path) -> None:
    config = tmp_path / "config.json"
    _write_schedule_config(config, ranking_weight=0.05)
    _write_schedule_metrics(tmp_path, include_ranking=True)

    report = audit_progress(
        tmp_path,
        expected_steps=4_000,
        checkpoint_interval=5_000,
        config_path=config,
    )

    assert report["status"] == "healthy"
    ranking = report["consistency_schedules"]["schedules"][
        "class_conditioning_ranking"
    ]
    assert ranking["active_rows"] == 3
    assert ranking["nonzero_loss_rows"] == 0


def test_audit_verifies_residual_alignment_schedule(tmp_path) -> None:
    config = tmp_path / "config.json"
    _write_schedule_config(config, residual_weight=0.05)
    _write_schedule_metrics(tmp_path, include_residual=True)

    report = audit_progress(
        tmp_path,
        expected_steps=4_000,
        checkpoint_interval=5_000,
        config_path=config,
    )

    assert report["status"] == "healthy"
    residual = report["consistency_schedules"]["schedules"][
        "class_conditioning_residual_alignment"
    ]
    assert residual["active_rows"] == 3
    assert residual["nonzero_loss_rows"] == 3


def test_audit_rejects_inactive_residual_alignment_objective(tmp_path) -> None:
    config = tmp_path / "config.json"
    _write_schedule_config(config, residual_weight=0.05)
    _write_schedule_metrics(
        tmp_path,
        include_residual=True,
        corrupt="residual_active_zero",
    )

    report = audit_progress(
        tmp_path,
        expected_steps=4_000,
        checkpoint_interval=5_000,
        config_path=config,
    )

    assert report["status"] == "invalid"
    assert (
        "class_conditioning_residual_alignment is zero for all active schedule rows"
        in report["issues"]
    )


def test_audit_uses_zero_effective_scale_when_consistency_weight_is_zero(tmp_path) -> None:
    config = tmp_path / "config.json"
    _write_schedule_config(config, rollout_weight=0.0, teacher_weight=0.0)
    _write_schedule_metrics(tmp_path)
    metrics_path = tmp_path / "train_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["rollout_consistency_scale"] = 0.0
        row["rollout_consistency"] = 0.0
        row["ema_teacher_consistency_scale"] = 0.0
        row["ema_teacher_consistency"] = 0.0
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = audit_progress(
        tmp_path,
        expected_steps=4_000,
        checkpoint_interval=5_000,
        config_path=config,
    )

    assert report["status"] == "healthy"
    rollout = report["consistency_schedules"]["schedules"]["rollout_consistency"]
    assert rollout["scheduled_scale_at_last_step"] == 1.0
    assert rollout["expected_scale_at_last_step"] == 0.0
    assert rollout["active_rows"] == 0


def test_audit_keeps_schedule_check_optional_for_legacy_callers(tmp_path) -> None:
    _write_metrics(tmp_path, [1, 50])

    report = audit_progress(tmp_path, expected_steps=100, checkpoint_interval=500)

    assert report["status"] == "healthy"
    assert report["consistency_schedules"] == {"status": "not_requested"}
