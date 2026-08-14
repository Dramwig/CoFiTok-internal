from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from cofitok.generation.capacity_probe_monitor import (
    build_capacity_probe_monitor_report,
    inspect_capacity_probe_run,
)


REVISION = "a" * 40
BRANCH = "scale/generation-capacity-probe-v1"


def _run(*, complete: bool, age: float | None, last_step: int) -> dict:
    return {
        "complete": complete,
        "activity_age_seconds": age,
        "last_step": last_step,
        "health_issues": [],
    }


def _report(runs: dict, *, training: list[str], runbook: list[str]) -> dict:
    return build_capacity_probe_monitor_report(
        runs=runs,
        training_processes=training,
        runbook_processes=runbook,
        stall_seconds=1_800,
        idle_failure_grace_seconds=600,
        disk={"total_bytes": 10, "used_bytes": 5, "free_bytes": 5},
        gpu=[],
        updated_at="2026-08-14T00:00:00+00:00",
        hostname="pro6000",
        git={"revision": REVISION, "branch": BRANCH, "tracked_dirty": False},
    )


def _make_partial_run(root: Path) -> Path:
    root.mkdir()
    environment = {"python": "test", "device": {"type": "cuda"}}
    environment_sha = hashlib.sha256(
        json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    dataset = {
        "schema_version": 1,
        "formal": True,
        "status": "pass",
        "issues": [],
        "dataset": "imagenet_256",
        "identity_sha256": "b" * 64,
    }
    manifest = {
        "config": {
            "runtime": {"steps": 100_000, "checkpoint_interval": 5_000},
            "loss": {
                "rollout_consistency_weight": 0.0,
                "rollout_consistency_start_step": 0,
                "rollout_consistency_warmup_steps": 0,
                "ema_teacher_consistency_weight": 0.0,
                "ema_teacher_consistency_start_step": 30_000,
                "ema_teacher_consistency_warmup_steps": 10_000,
            },
        },
        "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
        "runtime_environment_sha256": environment_sha,
        "dataset_provenance": dataset,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    metric = {
        "step": 10_000,
        "total": 0.1,
        "epsilon": 0.1,
        "samples_seen": 640_000,
        "elapsed_seconds": 1.0,
        "rollout_consistency_scale": 0.0,
        "ema_teacher_consistency_scale": 0.0,
    }
    (root / "train_metrics.jsonl").write_text(
        json.dumps(metric) + "\n",
        encoding="utf-8",
    )
    training_report = {
        **manifest,
        "training_complete": False,
        "completed_steps": 10_000,
        "target_steps": 100_000,
        "stop_requested": False,
        "stop_signal": None,
    }
    (root / "training_report.json").write_text(
        json.dumps(training_report) + "\n",
        encoding="utf-8",
    )
    checkpoint = root / "checkpoint_step_00010000.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()
    integrity = {
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
        "git_revision": REVISION,
        "git_branch": BRANCH,
        "git_dirty": False,
        "step": 10_000,
    }
    integrity_path = root / f"{checkpoint.name}.integrity.json"
    integrity_path.write_text(json.dumps(integrity) + "\n", encoding="utf-8")
    integrity_sha = hashlib.sha256(integrity_path.read_bytes()).hexdigest()
    (root / "latest.json").write_text(
        json.dumps(
            {
                **integrity,
                "integrity_manifest": integrity_path.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation = root / "partial_validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "role": "generation_capacity_probe_partial_training_validation",
                "run_dir": root.resolve().as_posix(),
                "git": {
                    "revision": REVISION,
                    "branch": BRANCH,
                    "tracked_dirty": False,
                },
                "configured_steps": 100_000,
                "completed_steps": 10_000,
                "training_complete": False,
                "intentional_partial_stop": True,
                "effective_batch_size": 64,
                "images_seen": 640_000,
                "checkpoint": {
                    "path": checkpoint.resolve().as_posix(),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": checkpoint_sha,
                    "integrity_manifest": {
                        "path": integrity_path.resolve().as_posix(),
                        "bytes": integrity_path.stat().st_size,
                        "sha256": integrity_sha,
                    },
                },
                "authorization_boundary": {
                    "capacity_probe_training_complete": True,
                    "full_training_complete": False,
                    "full_300k_launch_allowed": False,
                    "formal_generation_claim_allowed": False,
                    "release_allowed": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return validation


def test_monitor_tracks_capacity_probe_training_and_transition() -> None:
    runs = {
        "cofitok": _run(complete=False, age=10, last_step=100),
        "dense_identity": _run(complete=False, age=None, last_step=0),
    }
    report = _report(
        runs,
        training=["train capacity_probe_rgbtail3_100k --output base256_cofitok"],
        runbook=["runbook"],
    )
    assert report["status"] == "running"
    assert report["stage"] == "cofitok_training"
    assert report["scope"]["configured_100k_completion_allowed"] is False

    report = _report(runs, training=[], runbook=["runbook"])
    assert report["status"] == "waiting"


def test_monitor_passes_only_two_validated_intentional_stops() -> None:
    runs = {
        "cofitok": _run(complete=True, age=10, last_step=10_000),
        "dense_identity": _run(complete=True, age=10, last_step=10_000),
    }
    report = _report(runs, training=[], runbook=[])
    assert report["status"] == "pass"
    assert report["stage"] == "complete"


def test_monitor_fails_on_health_issue_or_stall() -> None:
    runs = {
        "cofitok": _run(complete=False, age=2_000, last_step=100),
        "dense_identity": _run(complete=False, age=None, last_step=0),
    }
    assert _report(runs, training=["train"], runbook=["runbook"])["status"] == "stalled"
    bad = copy.deepcopy(runs)
    bad["cofitok"]["health_issues"] = ["metric drift"]
    assert _report(bad, training=["train"], runbook=["runbook"])["status"] == "failed"


def test_inspect_capacity_probe_run_accepts_exact_partial_validation(
    tmp_path: Path,
) -> None:
    run = tmp_path / "base256_cofitok"
    validation = _make_partial_run(run)
    result = inspect_capacity_probe_run(
        run,
        validation_report=validation,
        now=validation.stat().st_mtime + 1,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )
    assert result["complete"] is True
    assert result["last_step"] == 10_000
    assert result["health_issues"] == []
    assert result["progress_fraction"] == 1.0


def test_inspect_capacity_probe_run_rejects_step_after_stop(tmp_path: Path) -> None:
    run = tmp_path / "base256_cofitok"
    validation = _make_partial_run(run)
    with (run / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 10_001, "total": 0.1}) + "\n")
    result = inspect_capacity_probe_run(
        run,
        validation_report=validation,
        now=validation.stat().st_mtime + 1,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )
    assert result["complete"] is False
    assert any("exceeds intentional stop" in issue for issue in result["health_issues"])


def test_inspect_capacity_probe_run_requires_partial_validation(tmp_path: Path) -> None:
    run = tmp_path / "base256_cofitok"
    validation = _make_partial_run(run)
    validation.unlink()
    result = inspect_capacity_probe_run(
        run,
        validation_report=validation,
        now=0,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )
    assert result["complete"] is False
    assert result["partial_validation"] is None


def test_inspect_capacity_probe_run_allows_recoverable_pre10k_report(
    tmp_path: Path,
) -> None:
    run = tmp_path / "base256_cofitok"
    validation = _make_partial_run(run)
    validation.unlink()
    training_path = run / "training_report.json"
    report = json.loads(training_path.read_text(encoding="utf-8"))
    report.update(
        completed_steps=5_000,
        stop_requested=True,
        stop_signal=15,
    )
    training_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    (run / "train_metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 5_000,
                "total": 0.1,
                "epsilon": 0.1,
                "samples_seen": 320_000,
                "elapsed_seconds": 1.0,
                "rollout_consistency_scale": 0.0,
                "ema_teacher_consistency_scale": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = inspect_capacity_probe_run(
        run,
        validation_report=validation,
        now=training_path.stat().st_mtime + 1,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )
    assert result["complete"] is False
    assert not any(
        "training report is not the exact intentional stop" in issue
        for issue in result["health_issues"]
    )


def test_capacity_probe_monitor_entrypoint_imports() -> None:
    __import__("scripts.monitor_generation_capacity_probe")
