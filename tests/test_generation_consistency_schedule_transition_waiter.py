from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cofitok.training.rollout import consistency_weight_scale
from scripts import wait_generation_consistency_schedule_transition as waiter

START_STEP = 30_000
WARMUP_STEPS = 10_000
WEIGHT = 0.25
FIRST_ACTIVE_STEP = 30_050
TARGET_STEP = 30_250
EFFECTIVE_BATCH = 64


def _row(step: int) -> dict:
    scale = consistency_weight_scale(
        step,
        start_step=START_STEP,
        warmup_steps=WARMUP_STEPS,
    )
    return {
        "step": step,
        "samples_seen": step * EFFECTIVE_BATCH,
        "ema_teacher_consistency_scale": scale,
        "ema_teacher_consistency": 0.0 if scale == 0.0 else 0.01 + step / 1_000_000,
        "total": 0.05,
        "epsilon": 0.03,
        "grad_norm": 0.4,
    }


def _rows() -> list[dict]:
    return [_row(step) for step in range(29_800, TARGET_STEP + 1, 50)]


def _audit(rows: list[dict]) -> dict:
    return waiter.audit_transition_rows(
        rows,
        start_step=START_STEP,
        warmup_steps=WARMUP_STEPS,
        weight=WEIGHT,
        target_step=TARGET_STEP,
        first_active_step=FIRST_ACTIVE_STEP,
        minimum_active_rows=5,
        effective_batch=EFFECTIVE_BATCH,
    )


def test_transition_audit_verifies_disabled_boundary_and_five_active_rows() -> None:
    report = _audit(_rows())

    assert report["status"] == "verified"
    assert report["start_row"]["ema_teacher_consistency_scale"] == 0.0
    assert report["start_row"]["ema_teacher_consistency"] == 0.0
    assert report["first_active_row"]["step"] == FIRST_ACTIVE_STEP
    assert report["first_active_row"]["ema_teacher_consistency_scale"] == pytest.approx(
        0.005
    )
    assert report["active_row_count"] == 5
    assert report["expected_target_scale"] == pytest.approx(0.025)
    assert report["target_row"]["ema_teacher_consistency"] > 0.0
    assert report["samples_seen_binding_verified"] is True


def test_transition_audit_rejects_scale_drift() -> None:
    rows = _rows()
    rows[-1]["ema_teacher_consistency_scale"] = 0.5

    with pytest.raises(ValueError, match="scale differs"):
        _audit(rows)


def test_transition_audit_rejects_nonzero_loss_before_activation() -> None:
    rows = _rows()
    rows[4]["ema_teacher_consistency"] = 0.1

    with pytest.raises(ValueError, match="nonzero before activation"):
        _audit(rows)


def test_transition_audit_rejects_zero_loss_after_activation() -> None:
    rows = _rows()
    rows[-1]["ema_teacher_consistency"] = 0.0

    with pytest.raises(ValueError, match="not positive"):
        _audit(rows)


def test_transition_audit_rejects_missing_exact_target_row() -> None:
    rows = [row for row in _rows() if row["step"] != TARGET_STEP]
    rows.append(_row(TARGET_STEP + 50))

    with pytest.raises(ValueError, match="omit required step"):
        _audit(rows)


def test_transition_audit_rejects_samples_seen_drift() -> None:
    rows = _rows()
    rows[-1]["samples_seen"] += 1

    with pytest.raises(ValueError, match="samples_seen binding differs"):
        _audit(rows)


def test_target_readiness_uses_canonical_metrics_tail(tmp_path: Path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text(
        "\n".join(json.dumps(row) for row in _rows()[:-1]) + "\n",
        encoding="utf-8",
    )
    assert waiter.target_is_ready(metrics, TARGET_STEP) == (
        False,
        "transition_target_not_reached",
    )

    with metrics.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_rows()[-1]) + "\n")
    assert waiter.target_is_ready(metrics, TARGET_STEP) == (True, "ready")


def test_run_manifest_validation_binds_schedule_and_training_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "git": {
                    "revision": "a" * 40,
                    "branch": "scale/test",
                    "dirty": False,
                },
                "config": {
                    "loss": {
                        "ema_teacher_consistency_start_step": START_STEP,
                        "ema_teacher_consistency_warmup_steps": WARMUP_STEPS,
                        "ema_teacher_consistency_weight": WEIGHT,
                    },
                    "runtime": {"steps": 100_000},
                    "data": {"batch_size": 16},
                    "optimization": {"gradient_accumulation_steps": 4},
                },
            }
        ),
        encoding="utf-8",
    )

    report = waiter.validate_run_manifest(
        path,
        expected_revision="a" * 40,
        expected_branch="scale/test",
        expected_start_step=START_STEP,
        expected_warmup_steps=WARMUP_STEPS,
        expected_weight=WEIGHT,
        expected_target_steps=100_000,
        effective_batch=EFFECTIVE_BATCH,
    )

    assert report["schedule"]["weight"] == WEIGHT
    assert report["micro_batch_size"] == 16
    assert report["gradient_accumulation_steps"] == 4
    assert len(report["source"]["sha256"]) == 64


def test_checked_in_quality_bridge_config_has_expected_transition_contract() -> None:
    project = Path(__file__).resolve().parents[1]
    config = (
        project
        / "configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
    )

    report = waiter.validate_config_source(
        config,
        expected_start_step=START_STEP,
        expected_warmup_steps=WARMUP_STEPS,
        expected_weight=WEIGHT,
        expected_target_steps=100_000,
        effective_batch=EFFECTIVE_BATCH,
        expected_log_interval=50,
    )

    assert report["name"].endswith("k8_100k")
    assert report["log_interval"] == 50


def test_once_waiter_writes_non_authorizing_waiting_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_metrics.jsonl").write_text(
        json.dumps(_row(29_950)) + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        run_dir=run_dir,
        target_step=TARGET_STEP,
        first_active_step=FIRST_ACTIVE_STEP,
        minimum_active_rows=5,
        report_output=tmp_path / "report.json",
        status_output=tmp_path / "status.json",
        poll_seconds=1,
        timeout_seconds=10,
        once=True,
    )

    assert waiter.run_waiter(args) == 0
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "waiting"
    assert status["detail"] == "transition_target_not_reached"
    assert status["scope"]["gpu_required"] is False
    assert status["scope"]["training_process_signals_allowed"] is False
