from __future__ import annotations

import json

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
