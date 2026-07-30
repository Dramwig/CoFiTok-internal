from __future__ import annotations

import json

import pytest

from scripts.monitor_generation_10pct_pair import build_monitor_report, inspect_run


def _run(*, complete: bool, age: float, last_step: int = 10_000) -> dict:
    return {
        "complete": complete,
        "activity_age_seconds": age,
        "last_step": last_step,
    }


def _report(runs, *, training=True, stall_seconds=1_800.0):
    return build_monitor_report(
        runs=runs,
        training_processes=["train"] if training else [],
        runbook_processes=["runbook"] if training else [],
        stall_seconds=stall_seconds,
        idle_failure_grace_seconds=600.0,
        disk={"total_bytes": 10, "used_bytes": 5, "free_bytes": 5},
        gpu=[],
        updated_at="2026-07-12T00:00:00+00:00",
        hostname="pro6000",
    )


def test_monitor_reports_running_current_stage() -> None:
    report = _report(
        {
            "cofitok": _run(complete=False, age=10.0),
            "dense_identity": _run(complete=False, age=10.0, last_step=0),
        }
    )

    assert report["status"] == "running"
    assert report["stage"] == "cofitok_training"
    assert report["issues"] == []


def test_monitor_advances_to_dense_after_cofitok_completion() -> None:
    report = _report(
        {
            "cofitok": _run(complete=True, age=100.0, last_step=50_000),
            "dense_identity": _run(complete=False, age=10.0),
        }
    )

    assert report["status"] == "running"
    assert report["stage"] == "dense_identity_training"


def test_monitor_tracks_freshest_run_during_alternating_full_training() -> None:
    report = _report(
        {
            "cofitok": _run(complete=False, age=1_000.0, last_step=50_000),
            "dense_identity": _run(complete=False, age=10.0, last_step=1_000),
        }
    )

    assert report["status"] == "running"
    assert report["stage"] == "dense_identity_training"


def test_monitor_uses_process_identity_before_new_run_emits_metrics() -> None:
    report = build_monitor_report(
        runs={
            "cofitok": _run(complete=False, age=4_000.0, last_step=50_000),
            "dense_identity": _run(complete=False, age=None, last_step=0),
        },
        training_processes=[
            "python /root/CoFiTok/CoFiTok-internal/scripts/train_generation.py "
            "--config imagenet256_dense_300k.json"
        ],
        runbook_processes=["runbook"],
        stall_seconds=1_800.0,
        idle_failure_grace_seconds=600.0,
        disk={"total_bytes": 10, "used_bytes": 5, "free_bytes": 5},
        gpu=[],
        updated_at="2026-07-12T00:00:00+00:00",
        hostname="pro6000",
    )

    assert report["status"] == "running"
    assert report["stage"] == "dense_identity_training"


def test_monitor_fails_closed_on_stall_or_dead_queue() -> None:
    runs = {
        "cofitok": _run(complete=False, age=2_000.0),
        "dense_identity": _run(complete=False, age=2_000.0, last_step=0),
    }
    assert _report(runs)["status"] == "stalled"
    assert _report(runs, training=False)["status"] == "failed"


def test_monitor_passes_only_when_both_reports_complete() -> None:
    runs = {
        "cofitok": _run(complete=True, age=100.0, last_step=50_000),
        "dense_identity": _run(complete=True, age=100.0, last_step=50_000),
    }

    assert _report(runs, training=False)["status"] == "pass"


def test_inspect_run_reads_metrics_report_and_checkpoint(tmp_path) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        json.dumps({"step": 5_000, "total": 0.1}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "training_report.json").write_text(
        json.dumps(
            {
                "training_complete": True,
                "completed_steps": 5_000,
                "target_steps": 5_000,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint_step_00005000.pt"
    checkpoint.write_bytes(b"checkpoint")

    report = inspect_run(tmp_path, expected_steps=5_000, now=checkpoint.stat().st_mtime + 1)

    assert report["complete"] is True
    assert report["last_step"] == 5_000
    assert report["checkpoints"] == [
        {"step": 5_000, "bytes": 10, "name": checkpoint.name}
    ]


def test_monitor_fails_on_nonfinite_or_nonmonotonic_metrics(tmp_path) -> None:
    metrics = [
        {"step": 50, "total": 0.1, "grad_norm": 0.2},
        {"step": 50, "total": float("nan"), "grad_norm": 0.3},
    ]
    (tmp_path / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
    )

    run = inspect_run(tmp_path, expected_steps=100, now=0.0)
    report = _report(
        {
            "cofitok": run,
            "dense_identity": _run(complete=False, age=10.0, last_step=0),
        }
    )

    assert report["status"] == "failed"
    assert any("strictly increasing" in issue for issue in report["issues"])
    assert any("non-finite" in issue for issue in report["issues"])


def test_inspect_run_detects_late_checkpoint_cadence(tmp_path) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        json.dumps({"step": 5_300, "total": 0.1}) + "\n", encoding="utf-8"
    )

    report = inspect_run(
        tmp_path,
        expected_steps=10_000,
        now=0.0,
        checkpoint_interval=5_000,
        checkpoint_grace_steps=250,
    )

    assert report["health_issues"] == [
        "checkpoint cadence is late: required>=5000, latest=0"
    ]


def test_required_checkpoint_integrity_waits_for_grace(tmp_path) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        json.dumps({"step": 1_250, "total": 0.1}) + "\n", encoding="utf-8"
    )
    (tmp_path / "checkpoint_step_00001250.pt").write_bytes(b"checkpoint")

    report = inspect_run(
        tmp_path,
        expected_steps=5_000,
        now=0.0,
        checkpoint_interval=1_250,
        checkpoint_grace_steps=100,
        checkpoint_integrity_policy="required",
    )

    assert report["health_issues"] == []
    integrity = report["checkpoint_integrity"]
    assert integrity["verification"] == "metadata_only_no_payload_hash"
    assert integrity["manifests"][0]["status"] == "optional_missing"
    assert integrity["latest_binding"]["status"] == "not_available"


def test_required_checkpoint_integrity_rejects_missing_sidecar_after_grace(
    tmp_path,
) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        json.dumps({"step": 1_350, "total": 0.1}) + "\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoint_step_00001250.pt"
    checkpoint.write_bytes(b"checkpoint")

    report = inspect_run(
        tmp_path,
        expected_steps=5_000,
        now=0.0,
        checkpoint_interval=1_250,
        checkpoint_grace_steps=100,
        checkpoint_integrity_policy="required",
    )

    assert any(
        "integrity manifest is missing" in issue
        for issue in report["health_issues"]
    )
    assert any(
        "latest.json is missing" in issue for issue in report["health_issues"]
    )


def test_required_checkpoint_integrity_verifies_metadata_and_latest_binding(
    tmp_path,
) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        json.dumps({"step": 1_350, "total": 0.1}) + "\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoint_step_00001250.pt"
    checkpoint.write_bytes(b"checkpoint")
    integrity = {
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": "a" * 64,
        "step": 1_250,
    }
    integrity_name = checkpoint.name + ".integrity.json"
    (tmp_path / integrity_name).write_text(
        json.dumps(integrity) + "\n", encoding="utf-8"
    )
    (tmp_path / "latest.json").write_text(
        json.dumps({**integrity, "integrity_manifest": integrity_name}) + "\n",
        encoding="utf-8",
    )

    report = inspect_run(
        tmp_path,
        expected_steps=5_000,
        now=0.0,
        checkpoint_interval=1_250,
        checkpoint_grace_steps=100,
        checkpoint_integrity_policy="required",
    )

    assert report["health_issues"] == []
    assert (
        report["checkpoint_integrity"]["manifests"][0]["status"]
        == "metadata_verified"
    )
    assert (
        report["checkpoint_integrity"]["latest_binding"]["status"]
        == "metadata_verified"
    )


def test_required_checkpoint_integrity_rejects_metadata_mismatch(tmp_path) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        json.dumps({"step": 1_350, "total": 0.1}) + "\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoint_step_00001250.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / f"{checkpoint.name}.integrity.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "checkpoint_bytes": checkpoint.stat().st_size + 1,
                "checkpoint_sha256": "b" * 64,
                "step": 1_250,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_run(
        tmp_path,
        expected_steps=5_000,
        now=0.0,
        checkpoint_interval=1_250,
        checkpoint_grace_steps=100,
        checkpoint_integrity_policy="required",
    )

    assert any(
        "mismatched checkpoint_bytes" in issue
        for issue in report["health_issues"]
    )


def test_required_integrity_does_not_compare_latest_to_previous_checkpoint(
    tmp_path,
) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        json.dumps({"step": 2_500, "total": 0.1}) + "\n", encoding="utf-8"
    )
    previous = tmp_path / "checkpoint_step_00001250.pt"
    previous.write_bytes(b"previous")
    previous_integrity = {
        "checkpoint": previous.name,
        "checkpoint_bytes": previous.stat().st_size,
        "checkpoint_sha256": "c" * 64,
        "step": 1_250,
    }
    (tmp_path / f"{previous.name}.integrity.json").write_text(
        json.dumps(previous_integrity) + "\n", encoding="utf-8"
    )
    current = tmp_path / "checkpoint_step_00002500.pt"
    current.write_bytes(b"current")
    (tmp_path / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint": current.name,
                "checkpoint_bytes": current.stat().st_size,
                "checkpoint_sha256": "d" * 64,
                "integrity_manifest": f"{current.name}.integrity.json",
                "step": 2_500,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_run(
        tmp_path,
        expected_steps=5_000,
        now=0.0,
        checkpoint_interval=1_250,
        checkpoint_grace_steps=100,
        checkpoint_integrity_policy="required",
    )

    assert report["health_issues"] == []
    assert (
        report["checkpoint_integrity"]["latest_binding"]["status"]
        == "pending_checkpoint_grace"
    )


def test_required_integrity_enforces_latest_after_new_checkpoint_grace(
    tmp_path,
) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        json.dumps({"step": 2_600, "total": 0.1}) + "\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoint_step_00002500.pt"
    checkpoint.write_bytes(b"current")
    integrity = {
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": "e" * 64,
        "step": 2_500,
    }
    (tmp_path / f"{checkpoint.name}.integrity.json").write_text(
        json.dumps(integrity) + "\n", encoding="utf-8"
    )
    (tmp_path / "latest.json").write_text(
        json.dumps({**integrity, "checkpoint": "checkpoint_step_00001250.pt"})
        + "\n",
        encoding="utf-8",
    )

    report = inspect_run(
        tmp_path,
        expected_steps=5_000,
        now=0.0,
        checkpoint_interval=1_250,
        checkpoint_grace_steps=100,
        checkpoint_integrity_policy="required",
    )

    assert "latest.json has mismatched checkpoint" in report["health_issues"]
    assert report["checkpoint_integrity"]["latest_binding"]["status"] == "invalid"


def test_monitor_waits_during_runbook_only_milestone_transition() -> None:
    runs = {
        "cofitok": _run(complete=False, age=10_000.0),
        "dense_identity": _run(complete=False, age=10_000.0, last_step=0),
    }
    report = build_monitor_report(
        runs=runs,
        training_processes=[],
        runbook_processes=["runbook"],
        stall_seconds=1_800.0,
        idle_failure_grace_seconds=600.0,
        disk={"total_bytes": 10, "used_bytes": 5, "free_bytes": 5},
        gpu=[],
        updated_at="2026-07-12T00:00:00+00:00",
        hostname="pro6000",
    )

    assert report["status"] == "waiting"
    assert report["stage"] == "cofitok_transition"


def test_inspect_run_rejects_invalid_thresholds(tmp_path) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        inspect_run(tmp_path, expected_steps=0, now=0.0)
