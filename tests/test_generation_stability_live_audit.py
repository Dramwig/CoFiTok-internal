from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.audit_generation_stability_live import (
    classify_roles,
    evaluate_snapshot,
    metrics_evidence,
    read_metrics,
)


TRAINING_REVISION = "2" * 40
CONTROL_REVISION = "5" * 40
EVALUATION_REVISION = "c" * 40
READINESS_REVISION = "d" * 40
RUNTIME_SHA = "e" * 64


def _process(
    pid: int,
    ppid: int,
    argv: str,
    cwd: Path,
    *,
    fd6: Path | None = None,
) -> dict:
    payload = {
        "pid": pid,
        "ppid": ppid,
        "start_ticks": pid * 100,
        "argv": argv,
        "cwd": cwd.resolve().as_posix(),
    }
    if fd6 is not None:
        payload["fd6"] = fd6.resolve().as_posix()
    return payload


def _snapshot(tmp_path: Path) -> dict:
    output_root = tmp_path / "stability"
    dense_run = output_root / "dense_rollout_x0_u2_ema_teacher"
    control = tmp_path / "control"
    training = tmp_path / "training"
    posteval = tmp_path / "posteval"
    readiness_project = tmp_path / "readiness"
    lock = output_root / "dense_recovery.lock"
    for path in (output_root, dense_run, control, training, posteval, readiness_project):
        path.mkdir(parents=True, exist_ok=True)
    processes = [
        _process(
            11,
            1,
            f"bash {control}/artifacts/runbooks/generation_stability_ema_teacher_dense_recovery_after_transition_failure.sh",
            tmp_path,
            fd6=lock,
        ),
        _process(
            12,
            11,
            f"python {training}/scripts/monitor_generation_pair.py --output-root {output_root}",
            tmp_path,
        ),
        _process(
            13,
            11,
            f"python scripts/run_generation_training_watchdog.py --status-output {dense_run}/training_watchdog.json",
            training,
        ),
        _process(
            14,
            13,
            f"python scripts/train_generation.py --output-dir {dense_run}",
            training,
        ),
        # DataLoader child shares trainer argv but is not a second GPU trainer.
        _process(
            15,
            14,
            f"python scripts/train_generation.py --output-dir {dense_run}",
            training,
        ),
        _process(
            16,
            1,
            f"python scripts/run_generation_stability_50k_posteval_waiter.py --monitor-report {output_root}/pair_monitor.json",
            posteval,
        ),
        _process(
            17,
            1,
            f"python scripts/run_generation_stability_full_readiness_waiter.py --posteval-status {output_root}/reports/posteval_waiter.json",
            readiness_project,
        ),
        _process(99, 1, "python -m fieldscope.cli extract-dataset", tmp_path),
    ]
    roles, unrelated = classify_roles(
        processes,
        gpu_compute_pids={14, 99},
        output_root=output_root,
        dense_run=dense_run,
        control_project=control,
        training_project=training,
        posteval_project=posteval,
        readiness_project=readiness_project,
    )
    assert [item["pid"] for item in roles["trainer"]] == [14]
    assert [item["pid"] for item in unrelated] == [99]
    return {
        "pair_monitor": {
            "status": "running",
            "stage": "dense_identity_training",
            "updated_at": "2026-08-02T00:00:00+00:00",
            "issues": [],
        },
        "dense_inspection": {
            "health_issues": [],
            "run_manifest": {"status": "verified"},
            "checkpoint_integrity": {
                "policy": "required",
                "verification": "metadata_only_no_payload_hash",
                "manifests": [],
                "latest_binding": {"status": "not_available"},
            },
        },
        "dense_metrics": {
            "strictly_increasing": True,
            "samples_seen_exact": True,
            "all_finite": True,
            "last": {"step": 2_000, "samples_seen": 128_000},
        },
        "dense_manifest": {
            "git": {
                "revision": TRAINING_REVISION,
                "branch": "training-branch",
                "dirty": False,
            },
            "runtime_environment_sha256": RUNTIME_SHA,
            "parameter_count": 62_824_707,
        },
        "watchdog": {"status": "running", "child_pid": 14},
        "recovery": {
            "status": "running",
            "full_training_launch_allowed": False,
        },
        "posteval": {"status": "waiting", "child_pid": None},
        "readiness": {
            "status": "waiting",
            "child_pid": None,
            "full_training_launch_allowed": False,
        },
        "git_identities": {
            "control": {
                "revision": CONTROL_REVISION,
                "branch": "control-branch",
                "tracked_dirty": False,
            },
            "training": {
                "revision": TRAINING_REVISION,
                "branch": "training-branch",
                "tracked_dirty": False,
            },
            "evaluation": {
                "revision": EVALUATION_REVISION,
                "branch": "evaluation-branch",
                "tracked_dirty": False,
            },
            "readiness": {
                "revision": READINESS_REVISION,
                "branch": "readiness-branch",
                "tracked_dirty": False,
            },
        },
        "roles": roles,
        "expected_role_pids": {
            "controller": 11,
            "pair_monitor": 12,
            "watchdog": 13,
            "trainer": 14,
            "posteval_waiter": 16,
            "readiness_waiter": 17,
        },
        "unrelated_gpu_processes": unrelated,
        "gpu": {
            "devices": [{"index": 0, "memory_free_mib": 3_853}],
            "compute_processes": [
                {"pid": 14, "used_memory_mib": 77_970},
                {"pid": 99, "used_memory_mib": 15_412},
            ],
            "query_status": {"gpu": 0, "compute": 0, "pmon": 0},
        },
        "lock_path": lock,
        "flock_probe_returncode": 1,
        "free_bytes": 176_000_000_000,
        "total_bytes": 1_000_000_000_000,
        "matched_required_bytes": 118_000_000_000,
        "aggregate_required_bytes": 180_000_000_000,
        "expected_control_revision": CONTROL_REVISION,
        "expected_control_branch": "control-branch",
        "expected_training_revision": TRAINING_REVISION,
        "expected_training_branch": "training-branch",
        "expected_evaluation_revision": EVALUATION_REVISION,
        "expected_evaluation_branch": "evaluation-branch",
        "expected_readiness_revision": READINESS_REVISION,
        "expected_readiness_branch": "readiness-branch",
        "expected_runtime_sha256": RUNTIME_SHA,
        "expected_parameter_count": 62_824_707,
    }


def test_live_audit_distinguishes_safe_matched_runway_from_aggregate_risk(
    tmp_path: Path,
) -> None:
    report = evaluate_snapshot(**_snapshot(tmp_path))

    assert report["status"] == "warning"
    assert report["issues"] == []
    assert report["storage"]["matched_50k"]["status"] == "pass"
    assert report["storage"]["aggregate_completion"]["status"] == "fail"
    assert report["warnings"] == ["aggregate completion storage runway is negative"]
    assert report["full_training_launch_allowed"] is False
    assert report["checkpoint_payload_hashes_performed"] is False
    assert report["process_roles"]["trainer"]["count"] == 1


def test_live_audit_fails_on_duplicate_gpu_trainer_and_unheld_lock(
    tmp_path: Path,
) -> None:
    values = _snapshot(tmp_path)
    duplicate = dict(values["roles"]["trainer"][0], pid=18)
    values["roles"]["trainer"].append(duplicate)
    values["flock_probe_returncode"] = 0

    report = evaluate_snapshot(**values)

    assert report["status"] == "failed"
    assert "expected exactly one trainer, found 2" in report["issues"]
    assert "dense_recovery.lock is not held by the active controller" in report["issues"]


def test_live_audit_treats_pid_change_as_material_warning_when_identity_matches(
    tmp_path: Path,
) -> None:
    values = _snapshot(tmp_path)
    values["aggregate_required_bytes"] = 170_000_000_000
    values["roles"]["watchdog"][0]["pid"] = 113

    report = evaluate_snapshot(**values)

    assert report["status"] == "warning"
    assert report["issues"] == []
    assert report["warnings"] == [
        "watchdog PID changed from launch identity 13 to 113"
    ]


def test_metrics_evidence_requires_exact_samples_and_finite_monotonic_steps() -> None:
    summary, issues = metrics_evidence(
        [
            {"step": 50, "samples_seen": 3_200, "elapsed_seconds": 10.0},
            {"step": 100, "samples_seen": 6_399, "elapsed_seconds": 20.0},
            {"step": 100, "samples_seen": 6_400, "total": float("nan")},
        ],
        effective_batch=64,
        metric_age_seconds=1.0,
        metric_stale_seconds=1_800.0,
    )

    assert summary["strictly_increasing"] is False
    assert summary["samples_seen_exact"] is False
    assert summary["all_finite"] is False
    assert any("step * 64" in issue for issue in issues)
    assert any("strictly increasing" in issue for issue in issues)
    assert any("non-finite" in issue for issue in issues)


def test_read_metrics_binds_identity_to_the_exact_parsed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    raw = b'{"step":50,"samples_seen":3200}\n'
    path.write_bytes(raw)

    rows, identity = read_metrics(path)

    assert rows == [{"step": 50, "samples_seen": 3_200}]
    assert identity["bytes"] == len(raw)
    assert identity["sha256"] == hashlib.sha256(raw).hexdigest()


def test_live_audit_fails_if_readiness_authorizes_full_training(tmp_path: Path) -> None:
    values = _snapshot(tmp_path)
    values["aggregate_required_bytes"] = 170_000_000_000
    values["readiness"]["full_training_launch_allowed"] = True

    report = evaluate_snapshot(**values)

    assert report["status"] == "failed"
    assert "readiness waiter unexpectedly allows full training" in report["issues"]
