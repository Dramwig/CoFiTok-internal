from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import wait_generation_matched_training_trajectory as waiter


REVISION = "a" * 40
BRANCH = "scale/generation-stability-quality-bridge-100k"
TREE = "b" * 40
WAITER_SHA = "c" * 64
CUTOFF = 40_000
BATCH = 64


def _identity(path: Path) -> dict:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": waiter.file_sha256(path),
    }


def _schedule_report(tmp_path: Path) -> tuple[Path, Path, Path]:
    training = tmp_path / "training"
    dense = tmp_path / "dense"
    training.mkdir()
    dense.mkdir()
    manifest = dense / "run_manifest.json"
    metrics = dense / "train_metrics.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    metrics.write_text("{}\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "role": waiter.SCHEDULE_ROLE,
        "status": "pass",
        "scope": {
            "read_only_metrics_verification": True,
            "gpu_required": False,
            "training_process_signals_allowed": False,
            "unrelated_process_signals_allowed": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
        },
        "run_dir": dense.resolve().as_posix(),
        "training_checkout": {
            "path": training.resolve().as_posix(),
            "revision": REVISION,
            "branch": BRANCH,
            "tree": TREE,
            "tracked_dirty": False,
        },
        "waiter_source": {"sha256": WAITER_SHA},
        "run_manifest": {
            "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
            "source": _identity(manifest),
            "effective_batch": BATCH,
            "target_steps": 100_000,
            "schedule": {
                "start_step": 30_000,
                "warmup_steps": 10_000,
                "weight": 0.25,
            },
        },
        "metrics": {
            "last_step_at_read": CUTOFF,
            "source": _identity(metrics),
        },
        "transition": {
            "status": "verified",
            "start_step": 30_000,
            "warmup_steps": 10_000,
            "target_step": CUTOFF,
            "expected_target_scale": 1.0,
            "samples_seen_binding_verified": True,
            "strictly_increasing_metrics": True,
            "target_row": {
                "step": CUTOFF,
                "samples_seen": CUTOFF * BATCH,
                "ema_teacher_consistency_scale": 1.0,
                "validation_event_index": 39,
            },
        },
    }
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, dense, training


def _validate(path: Path, dense: Path, training: Path) -> dict:
    return waiter.validate_schedule_report(
        path,
        dense_run_dir=dense,
        training_checkout=training,
        expected_training_revision=REVISION,
        expected_training_branch=BRANCH,
        expected_training_tree=TREE,
        expected_waiter_source_sha256=WAITER_SHA,
        cutoff_step=CUTOFF,
        expected_effective_batch=BATCH,
        expected_training_steps=100_000,
        expected_start_step=30_000,
        expected_warmup_steps=10_000,
        expected_weight=0.25,
    )


def test_schedule_report_validation_binds_exact_full_warmup_source(
    tmp_path: Path,
) -> None:
    path, dense, training = _schedule_report(tmp_path)

    evidence = _validate(path, dense, training)

    assert evidence["report"]["transition"]["target_step"] == CUTOFF
    assert evidence["identity"]["sha256"] == waiter.file_sha256(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["scope"].update(
                {"promotion_authorization_allowed": True}
            ),
            "scope",
        ),
        (
            lambda report: report["training_checkout"].update({"tree": "d" * 40}),
            "tree differs",
        ),
        (
            lambda report: report["waiter_source"].update({"sha256": "e" * 64}),
            "source SHA256 differs",
        ),
        (
            lambda report: report["transition"]["target_row"].update(
                {"ema_teacher_consistency_scale": 0.99}
            ),
            "not at full strength",
        ),
    ],
)
def test_schedule_report_validation_fails_closed(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    path, dense, training = _schedule_report(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _validate(path, dense, training)


def test_schedule_readiness_requires_passing_report(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert waiter.schedule_is_ready(missing) == (
        False,
        "dense_schedule_report_missing",
    )
    missing.write_text(json.dumps({"status": "waiting"}), encoding="utf-8")
    assert waiter.schedule_is_ready(missing) == (
        False,
        "dense_schedule_report_not_pass",
    )
    missing.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    assert waiter.schedule_is_ready(missing) == (True, "ready")


def test_atomic_copy_preserves_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "nested" / "destination.jsonl"
    source.write_bytes(b'{"step":1}\n{"step":50}\n')

    waiter._atomic_copy(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert not list(destination.parent.glob("*.tmp"))


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cutoff_step=CUTOFF,
        expected_effective_batch=BATCH,
        expected_training_revision=REVISION,
        expected_training_branch=BRANCH,
        expected_builder_source_sha256="f" * 64,
        expected_builder_revision="1" * 40,
        expected_builder_branch="analysis/builder",
        cofitok_run_dir=tmp_path / "cofitok",
        dense_run_dir=tmp_path / "dense",
    )


def _trajectory_report(tmp_path: Path, args: SimpleNamespace) -> tuple[Path, Path, Path]:
    args.cofitok_run_dir.mkdir()
    args.dense_run_dir.mkdir()
    cofitok_snapshot = tmp_path / "cofitok_snapshot.jsonl"
    dense_snapshot = tmp_path / "dense_snapshot.jsonl"
    cofitok_snapshot.write_text('{"step":40000}\n', encoding="utf-8")
    dense_snapshot.write_text('{"step":40000}\n', encoding="utf-8")
    report = {
        "schema_version": 3,
        "role": waiter.TRAJECTORY_ROLE,
        "status": "pass",
        "cutoff_step": CUTOFF,
        "images_seen_per_method": CUTOFF * BATCH,
        "contract": {
            "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
            "effective_batch_size": BATCH,
            "generation_pair_contract": {"valid": True},
        },
        "trajectories": {
            "cofitok": {
                "last_step": CUTOFF,
                "samples_seen": CUTOFF * BATCH,
                "all_numeric_metrics_finite": True,
                "validation_provenance_complete": True,
            },
            "dense_identity": {
                "last_step": CUTOFF,
                "samples_seen": CUTOFF * BATCH,
                "all_numeric_metrics_finite": True,
                "validation_provenance_complete": True,
            },
        },
        "paired_fixed_validation": {
            "events": [
                {
                    "step": CUTOFF,
                    "observed_schedule_scales": {
                        "ema_teacher_consistency": 1.0
                    },
                }
            ]
        },
        "claim_boundary": {
            "schedule_regime_quality_claim_allowed": False,
            "quality_claim_allowed": False,
            "formal_50k_gate_substitute": False,
            "promotion_authorization_allowed": False,
            "full_training_launch_allowed": False,
            "sample_quality_metrics_present": False,
        },
        "builder": {
            "sha256": args.expected_builder_source_sha256,
            "git": {
                "revision": args.expected_builder_revision,
                "branch": args.expected_builder_branch,
                "tracked_dirty": False,
            },
        },
        "sources": {
            "cofitok_metrics": {
                "origin": (args.cofitok_run_dir / "train_metrics.jsonl").as_posix(),
                "observed_file": _identity(cofitok_snapshot),
            },
            "dense_metrics": {
                "origin": (args.dense_run_dir / "train_metrics.jsonl").as_posix(),
                "observed_file": _identity(dense_snapshot),
            },
        },
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, cofitok_snapshot, dense_snapshot


def test_trajectory_validation_preserves_non_authorizing_boundary(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    path, cofitok_snapshot, dense_snapshot = _trajectory_report(tmp_path, args)

    report = waiter.validate_trajectory_report(
        path,
        args=args,
        cofitok_snapshot=cofitok_snapshot,
        dense_snapshot=dense_snapshot,
    )

    assert report["claim_boundary"]["quality_claim_allowed"] is False
    assert report["claim_boundary"]["full_training_launch_allowed"] is False


def test_trajectory_validation_rejects_quality_claim(tmp_path: Path) -> None:
    args = _args(tmp_path)
    path, cofitok_snapshot, dense_snapshot = _trajectory_report(tmp_path, args)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["claim_boundary"]["quality_claim_allowed"] = True
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="claim boundary"):
        waiter.validate_trajectory_report(
            path,
            args=args,
            cofitok_snapshot=cofitok_snapshot,
            dense_snapshot=dense_snapshot,
        )


def test_once_waiter_writes_cpu_only_non_authorizing_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = SimpleNamespace(
        cutoff_step=CUTOFF,
        poll_seconds=1,
        timeout_seconds=10,
        builder_timeout_seconds=10,
        expected_effective_batch=BATCH,
        dense_schedule_report=tmp_path / "missing_schedule.json",
        output_dir=tmp_path / "output",
        status_output=tmp_path / "status.json",
    )
    control = {"checkout": {"revision": "a" * 40}, "source": {"sha256": "b" * 64}}
    builder = {"checkout": {"revision": "c" * 40}, "source": {"sha256": "d" * 64}}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(waiter, "verify_control_identity", lambda _: control)
    monkeypatch.setattr(waiter, "verify_builder_identity", lambda _: builder)
    args.once = True

    assert waiter.run_waiter(args) == 0
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "waiting"
    assert status["detail"] == "dense_schedule_report_missing"
    assert status["scope"]["gpu_required"] is False
    assert status["scope"]["training_process_signals_allowed"] is False
    assert status["scope"]["full_training_launch_allowed"] is False


def test_waiter_requires_cuda_hidden(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = SimpleNamespace(
        cutoff_step=CUTOFF,
        poll_seconds=1,
        timeout_seconds=10,
        builder_timeout_seconds=10,
        expected_effective_batch=BATCH,
    )

    with pytest.raises(ValueError, match="must be exactly -1"):
        waiter.run_waiter(args)
