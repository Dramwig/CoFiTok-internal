from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from cofitok.training.metrics import (
    append_metrics_resume_event,
    empty_metrics_resume_history,
    persist_metrics_resume_history,
    reconcile_metrics_for_resume,
)
from scripts import build_generation_matched_training_trajectory as trajectory


REVISION = "a" * 40
BRANCH = "scale/generation-stability-50k-preflight"
IDENTITY = "b" * 64
RUNTIME = "c" * 64


def _config(*, dense: bool) -> dict:
    return {
        "data": {"batch_size": 4, "dataset": "imagenet_256_10pct"},
        "diffusion": {"prediction_target": "epsilon"},
        "optimization": {
            "gradient_accumulation_steps": 2,
            "log_interval": 50,
        },
        "runtime": {
            "evaluation_interval": 100,
            "steps": 500,
            "protected_checkpoint_steps": [],
        },
        "model": {
            "base_channels": 32,
            "token_count": 1 if dense else 8,
            "token_channels": 3 if dense else 8,
            "predictor_use_feedback": not dense,
            "synthesis_mode": "dense_identity" if dense else "fixed_basis",
        },
        "loss": {
            "epsilon_weight": 1.0,
            "rollout_consistency_weight": 0.1,
            "rollout_consistency_start_step": 0,
            "rollout_consistency_warmup_steps": 200,
            "rollout_consistency_timestep_delta": 10,
            "rollout_consistency_unroll_steps": 2,
            "rollout_consistency_batch_fraction": 0.125,
            "rollout_consistency_clip_x0": True,
            "rollout_consistency_mode": "clipped_x0",
            "ema_teacher_consistency_weight": 0.25,
            "ema_teacher_consistency_start_step": 100,
            "ema_teacher_consistency_warmup_steps": 100,
            "ema_teacher_consistency_batch_fraction": 0.0625,
            "prefix_weight": 0.0,
        },
    }


def _manifest(*, dense: bool) -> dict:
    return {
        "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
        "parameter_count": 999 if dense else 1_000,
        "runtime_environment_sha256": RUNTIME,
        "dataset_provenance": {
            "status": "pass",
            "formal": True,
            "dataset": "imagenet_256_10pct",
            "identity_sha256": IDENTITY,
        },
        "config": _config(dense=dense),
    }


def _rows(*, dense: bool) -> list[dict]:
    result = []
    for step in (1, 50, 100, 150, 200):
        row = {
            "step": step,
            "samples_seen": step * 8,
            "epsilon": 0.5 / step + (0.001 if dense else 0.0),
            "rollout_consistency": 0.1,
            "rollout_consistency_scale": min(step / 200, 1.0),
            "ema_teacher_consistency": 0.05,
            "ema_teacher_consistency_scale": (
                0.0 if step < 100 else min((step - 100) / 100, 1.0)
            ),
        }
        if step in (100, 200):
            event = step // 100 - 1
            row.update(
                {
                    "validation_epsilon_mse": (
                        0.03 + event * 0.001 + (0.0002 if dense else 0.0)
                    ),
                    "validation_event_index": event,
                    "validation_batch_index": event,
                    "validation_num_images": 64,
                    "validation_noise_seed": 102030,
                }
            )
        result.append(row)
    return result


def _metrics(*, dense: bool) -> dict:
    return {"rows": _rows(dense=dense)}


def _resume_boundary_row(step: int) -> dict:
    return {
        "step": step,
        "samples_seen": step * 8,
        "epsilon": 0.5 / step,
        "rollout_consistency": 0.1,
        "rollout_consistency_scale": min(step / 200, 1.0),
        "ema_teacher_consistency": 0.05,
        "ema_teacher_consistency_scale": (
            0.0 if step < 100 else min((step - 100) / 100, 1.0)
        ),
    }


def _write_checkpoint(root: Path, step: int) -> tuple[Path, dict]:
    checkpoint = root / f"checkpoint_step_{step:08d}.pt"
    payload = f"checkpoint:{step}".encode("utf-8")
    checkpoint.write_bytes(payload)
    integrity = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_format_version": 1,
        "step": step,
    }
    checkpoint.with_name(f"{checkpoint.name}.integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint, integrity


def _rehash_history(history: dict) -> None:
    history["history_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "legacy_history_complete": history["legacy_history_complete"],
                "legacy_reconciliations": history["legacy_reconciliations"],
                "events": history["events"],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _history_backed_run(
    tmp_path: Path,
    *,
    event_steps: tuple[int, ...] = (125, 175),
) -> dict:
    root = tmp_path / "cofitok_run"
    root.mkdir()
    metrics = root / "train_metrics.jsonl"
    final_rows = _rows(dense=False) + [
        _resume_boundary_row(step + 1) for step in event_steps
    ]
    final_rows.sort(key=lambda row: row["step"])

    def write_rows(rows: list[dict]) -> None:
        metrics.write_text(
            "".join(
                json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    write_rows(final_rows)
    history = empty_metrics_resume_history()
    reconciliations = []
    for step in event_steps:
        checkpoint, integrity = _write_checkpoint(root, step)
        reconciliation = reconcile_metrics_for_resume(metrics, resume_step=step)
        history = append_metrics_resume_event(
            history,
            output_dir=root,
            resume_checkpoint=checkpoint,
            checkpoint_integrity=integrity,
            reconciliation=reconciliation,
        )
        reconciliations.append(reconciliation)
        retained = [row for row in final_rows if row["step"] <= step]
        future = [row for row in final_rows if row["step"] > step]
        with metrics.open("a", encoding="utf-8") as handle:
            for row in future:
                handle.write(
                    json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                )
        assert json.loads(metrics.read_text(encoding="utf-8").splitlines()[0]) == retained[0]
    persist_metrics_resume_history(root, history)
    manifest = _manifest(dense=False)
    manifest.update(
        {
            "resume": (root / f"checkpoint_step_{event_steps[-1]:08d}.pt")
            .resolve()
            .as_posix(),
            "metrics_resume_reconciliation": reconciliations[-1],
            "metrics_resume_history": history,
        }
    )
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "root": root,
        "metrics": metrics,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "history": history,
        "reconciliations": reconciliations,
        "rows": final_rows,
    }


def test_report_binds_matched_validation_without_claiming_quality() -> None:
    report = trajectory.build_report(
        cofitok_metrics=_metrics(dense=False),
        dense_metrics=_metrics(dense=True),
        cofitok_manifest=_manifest(dense=False),
        dense_manifest=_manifest(dense=True),
        cutoff_step=200,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )

    assert report["status"] == "pass"
    assert report["schema_version"] == 5
    assert report["images_seen_per_method"] == 1_600
    assert report["contract"]["generation_pair_contract"]["valid"] is True
    paired = report["paired_fixed_validation"]
    assert paired["summary"]["event_count"] == 2
    assert paired["summary"]["cofitok_lower_event_count"] == 2
    assert paired["events"][0]["validation_noise_seed"] == 102030
    regimes = paired["schedule_regimes"]
    assert regimes["contracts"]["rollout_consistency"]["full_scale_step"] == 200
    rollout = regimes["by_schedule"]["rollout_consistency"]
    assert [entry["phase"] for entry in rollout] == ["warmup", "full_scale"]
    assert rollout[0]["event_steps"] == [100]
    assert rollout[1]["event_steps"] == [200]
    ema_teacher = regimes["by_schedule"]["ema_teacher_consistency"]
    assert [entry["phase"] for entry in ema_teacher] == ["warmup", "full_scale"]
    assert regimes["individual_regime_significance_claim_allowed"] is False
    assert report["comparison_policy"]["total_loss_comparison_allowed"] is False
    assert report["comparison_policy"]["training_wall_clock_comparison_allowed"] is False
    assert report["claim_boundary"]["quality_claim_allowed"] is False
    assert report["claim_boundary"]["formal_50k_gate_substitute"] is False
    assert report["claim_boundary"]["full_training_launch_allowed"] is False
    assert report["claim_boundary"]["schedule_regime_quality_claim_allowed"] is False


def test_report_accepts_only_manifest_bound_first_post_resume_step() -> None:
    cofitok_manifest = _manifest(dense=False)
    cofitok_manifest["resume"] = "/runs/checkpoint_step_00000125.pt"
    cofitok_manifest["metrics_resume_reconciliation"] = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 125,
        "retained_rows": 3,
        "orphaned_rows": 0,
        "orphan_archive": None,
        "orphan_sha256": None,
    }
    cofitok = _metrics(dense=False)
    cofitok["rows"].insert(
        3,
        {
            "step": 126,
            "samples_seen": 1_008,
            "epsilon": 0.01,
            "rollout_consistency": 0.1,
            "rollout_consistency_scale": 0.63,
            "ema_teacher_consistency": 0.05,
            "ema_teacher_consistency_scale": 0.26,
        },
    )

    report = trajectory.build_report(
        cofitok_metrics=cofitok,
        dense_metrics=_metrics(dense=True),
        cofitok_manifest=cofitok_manifest,
        dense_manifest=_manifest(dense=True),
        cutoff_step=200,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )

    cofitok_result = report["trajectories"]["cofitok"]
    assert cofitok_result["row_count"] == 6
    assert cofitok_result["resume_boundary_steps"] == [126]
    assert cofitok_result["metrics_resume"]["resume_step"] == 125
    assert cofitok_result["metrics_resume_retained_rows_verified"] is True
    assert report["trajectories"]["dense_identity"]["resume_boundary_steps"] == []


def test_report_accepts_reconciled_resume_with_bound_orphan_evidence() -> None:
    cofitok_manifest = _manifest(dense=False)
    cofitok_manifest["resume"] = "/runs/checkpoint_step_00000125.pt"
    cofitok_manifest["metrics_resume_reconciliation"] = {
        "schema_version": 1,
        "status": "reconciled",
        "resume_step": 125,
        "retained_rows": 3,
        "orphaned_rows": 2,
        "orphan_archive": "/runs/train_metrics_orphaned_at_resume_00000125.jsonl",
        "orphan_sha256": "a" * 64,
    }
    cofitok = _metrics(dense=False)
    cofitok["rows"].insert(
        3,
        {
            "step": 126,
            "samples_seen": 1_008,
            "epsilon": 0.01,
            "rollout_consistency": 0.1,
            "rollout_consistency_scale": 0.63,
            "ema_teacher_consistency": 0.05,
            "ema_teacher_consistency_scale": 0.26,
        },
    )

    report = trajectory.build_report(
        cofitok_metrics=cofitok,
        dense_metrics=_metrics(dense=True),
        cofitok_manifest=cofitok_manifest,
        dense_manifest=_manifest(dense=True),
        cutoff_step=200,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )

    resume = report["trajectories"]["cofitok"]["metrics_resume"]
    assert resume["reconciliation_status"] == "reconciled"
    assert resume["orphaned_rows"] == 2
    assert resume["orphan_sha256"] == "a" * 64


def test_report_accepts_multiple_manifest_bound_resume_events() -> None:
    first = {
        "schema_version": 1,
        "status": "reconciled",
        "resume_step": 125,
        "retained_rows": 3,
        "orphaned_rows": 2,
        "orphan_archive": "/runs/orphan-125.jsonl",
        "orphan_sha256": "a" * 64,
    }
    latest = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 175,
        "retained_rows": 5,
        "orphaned_rows": 0,
        "orphan_archive": None,
        "orphan_sha256": None,
    }
    cofitok_manifest = _manifest(dense=False)
    cofitok_manifest["resume"] = "/runs/checkpoint_step_00000175.pt"
    cofitok_manifest["metrics_resume_reconciliation"] = latest
    cofitok = _metrics(dense=False)
    for step in (126, 176):
        cofitok["rows"].append(
            {
                "step": step,
                "samples_seen": step * 8,
                "epsilon": 0.01,
                "rollout_consistency": 0.1,
                "rollout_consistency_scale": step / 200,
                "ema_teacher_consistency": 0.05,
                "ema_teacher_consistency_scale": (step - 100) / 100,
            }
        )
    cofitok["rows"].sort(key=lambda row: row["step"])

    report = trajectory.build_report(
        cofitok_metrics=cofitok,
        dense_metrics=_metrics(dense=True),
        cofitok_manifest=cofitok_manifest,
        dense_manifest=_manifest(dense=True),
        cutoff_step=200,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        cofitok_reconciliation_history=[first, latest],
    )

    result = report["trajectories"]["cofitok"]
    assert result["resume_boundary_steps"] == [126, 176]
    assert result["metrics_resume"]["event_count"] == 2
    assert [event["resume_step"] for event in result["metrics_resume"]["events"]] == [
        125,
        175,
    ]


def test_report_accepts_protected_milestone_segment_boundary() -> None:
    cofitok_manifest = _manifest(dense=False)
    dense_manifest = _manifest(dense=True)
    for manifest in (cofitok_manifest, dense_manifest):
        manifest["config"]["runtime"]["protected_checkpoint_steps"] = [125]
    cofitok = _metrics(dense=False)
    cofitok["rows"].append(
        {
            "step": 126,
            "samples_seen": 1_008,
            "epsilon": 0.01,
            "rollout_consistency": 0.1,
            "rollout_consistency_scale": 0.63,
            "ema_teacher_consistency": 0.05,
            "ema_teacher_consistency_scale": 0.26,
        }
    )
    cofitok["rows"].sort(key=lambda row: row["step"])

    report = trajectory.build_report(
        cofitok_metrics=cofitok,
        dense_metrics=_metrics(dense=True),
        cofitok_manifest=cofitok_manifest,
        dense_manifest=dense_manifest,
        cutoff_step=200,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )

    boundary = report["trajectories"]["cofitok"]["resume_boundary_evidence"]
    assert boundary == [
        {
            "resume_step": 125,
            "first_post_resume_step": 126,
            "metrics_reconciliation": None,
            "protected_checkpoint_schedule": True,
        }
    ]


def test_report_rejects_reconciliation_history_not_ending_at_manifest() -> None:
    cofitok_manifest = _manifest(dense=False)
    cofitok_manifest["resume"] = "/runs/checkpoint_step_00000175.pt"
    cofitok_manifest["metrics_resume_reconciliation"] = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 175,
        "retained_rows": 4,
        "orphaned_rows": 0,
        "orphan_archive": None,
        "orphan_sha256": None,
    }

    with pytest.raises(ValueError, match="does not end at the manifest resume"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=cofitok_manifest,
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            cofitok_reconciliation_history=[
                {
                    "schema_version": 1,
                    "status": "unchanged",
                    "resume_step": 125,
                    "retained_rows": 3,
                    "orphaned_rows": 0,
                    "orphan_archive": None,
                    "orphan_sha256": None,
                }
            ],
        )


def test_reconciliation_loader_physically_binds_full_history(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text("{}\n", encoding="utf-8")
    reports = []
    events = []
    for step, content in ((125, b"first orphan\n"), (175, b"second orphan\n")):
        orphan_sha = hashlib.sha256(content).hexdigest()
        orphan = tmp_path / (
            f"train_metrics_orphaned_at_resume_{step:08d}_{orphan_sha[:12]}.jsonl"
        )
        orphan.write_bytes(content)
        event = {
            "schema_version": 1,
            "status": "reconciled",
            "resume_step": step,
            "metrics": metrics.resolve().as_posix(),
            "retained_rows": 3 if step == 125 else 5,
            "orphaned_rows": 1,
            "orphan_archive": orphan.resolve().as_posix(),
            "orphan_sha256": orphan_sha,
        }
        report = tmp_path / (
            f"metrics_resume_reconciliation_{step:08d}_{orphan_sha[:12]}.json"
        )
        report.write_text(json.dumps(event), encoding="utf-8")
        reports.append(report)
        events.append(event)
    manifest = {
        "metrics_resume_reconciliation": {
            **events[-1],
            "report": reports[-1].resolve().as_posix(),
        }
    }

    loaded, sources = trajectory._load_reconciliation_history(
        reports,
        metrics_path=metrics,
        manifest=manifest,
        label="cofitok",
    )

    assert loaded == events
    assert [source["resume_step"] for source in sources] == [125, 175]
    assert all(source["orphan_archive"]["sha256"] for source in sources)


def test_reconciliation_loader_rejects_orphan_payload_drift(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text("{}\n", encoding="utf-8")
    orphan = tmp_path / "orphan.jsonl"
    orphan.write_text("tampered\n", encoding="utf-8")
    event = {
        "schema_version": 1,
        "status": "reconciled",
        "resume_step": 125,
        "metrics": metrics.resolve().as_posix(),
        "retained_rows": 3,
        "orphaned_rows": 1,
        "orphan_archive": orphan.resolve().as_posix(),
        "orphan_sha256": "a" * 64,
    }
    report = tmp_path / "reconciliation.json"
    report.write_text(json.dumps(event), encoding="utf-8")
    manifest = {
        "metrics_resume_reconciliation": {
            **event,
            "report": report.resolve().as_posix(),
        }
    }

    with pytest.raises(ValueError, match="orphan SHA256 differs"):
        trajectory._load_reconciliation_history(
            [report],
            metrics_path=metrics,
            manifest=manifest,
            label="cofitok",
        )


@pytest.mark.parametrize("event_steps", [(125, 175), (75, 125, 175)])
def test_manifest_history_is_automatically_loaded_and_physically_verified(
    tmp_path,
    event_steps: tuple[int, ...],
) -> None:
    run = _history_backed_run(tmp_path, event_steps=event_steps)

    events, sources, evidence = trajectory._select_reconciliation_history(
        manifest_path=run["manifest_path"],
        manifest=run["manifest"],
        metrics_path=run["metrics"],
        manual_paths=None,
        label="cofitok",
    )

    assert [event["resume_step"] for event in events] == list(event_steps)
    assert len(sources) == len(event_steps)
    assert all(source["report"]["sha256"] for source in sources)
    assert all(source["checkpoint_binding_available"] is True for source in sources)
    assert evidence["history_sha256_verified"] is True
    assert evidence["manifest_journal_equality_verified"] is True
    assert evidence["metrics_prefix_bindings_verified"] is True
    assert evidence["current_manifest_binding_verified"] is True
    assert evidence["complete_recovery_chain_verified"] is True


@pytest.mark.parametrize("drift", ["digest", "report", "orphan", "metrics_prefix"])
def test_manifest_history_rejects_physical_or_digest_drift(tmp_path, drift: str) -> None:
    run = _history_backed_run(tmp_path)
    if drift == "digest":
        run["manifest"]["metrics_resume_history"]["history_sha256"] = "0" * 64
    elif drift == "report":
        with Path(run["reconciliations"][0]["report"]).open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(" ")
    elif drift == "orphan":
        with Path(run["reconciliations"][0]["orphan_archive"]).open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("{}\n")
    else:
        payload = bytearray(run["metrics"].read_bytes())
        payload[0] = ord("[")
        run["metrics"].write_bytes(bytes(payload))

    with pytest.raises((ValueError, FileNotFoundError)):
        trajectory._select_reconciliation_history(
            manifest_path=run["manifest_path"],
            manifest=run["manifest"],
            metrics_path=run["metrics"],
            manual_paths=None,
            label="cofitok",
        )


def test_manifest_history_rejects_bound_report_path_drift(tmp_path) -> None:
    run = _history_backed_run(tmp_path)
    history = run["manifest"]["metrics_resume_history"]
    history["events"][0]["reconciliation"]["report"]["path"] = "missing.json"
    _rehash_history(history)
    (run["root"] / "metrics_resume_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="reconciliation report"):
        trajectory._select_reconciliation_history(
            manifest_path=run["manifest_path"],
            manifest=run["manifest"],
            metrics_path=run["metrics"],
            manual_paths=None,
            label="cofitok",
        )


def test_manifest_history_rejects_current_reconciliation_divergence(tmp_path) -> None:
    run = _history_backed_run(tmp_path)
    run["manifest"]["metrics_resume_reconciliation"]["retained_rows"] += 1

    with pytest.raises(ValueError, match="latest history event"):
        trajectory._select_reconciliation_history(
            manifest_path=run["manifest_path"],
            manifest=run["manifest"],
            metrics_path=run["metrics"],
            manual_paths=None,
            label="cofitok",
        )


def test_cli_history_must_exactly_match_manifest_history(tmp_path) -> None:
    run = _history_backed_run(tmp_path, event_steps=(75, 125, 175))

    with pytest.raises(ValueError, match="diverges from manifest metrics resume history"):
        trajectory._select_reconciliation_history(
            manifest_path=run["manifest_path"],
            manifest=run["manifest"],
            metrics_path=run["metrics"],
            manual_paths=[Path(run["reconciliations"][-1]["report"])],
            label="cofitok",
        )


def test_matching_cli_history_corroborates_manifest_history(tmp_path) -> None:
    run = _history_backed_run(tmp_path, event_steps=(75, 125, 175))

    events, _, evidence = trajectory._select_reconciliation_history(
        manifest_path=run["manifest_path"],
        manifest=run["manifest"],
        metrics_path=run["metrics"],
        manual_paths=[
            Path(reconciliation["report"])
            for reconciliation in run["reconciliations"]
        ],
        label="cofitok",
    )

    assert len(events) == 3
    assert evidence["manual_cli_corroborated"] is True
    assert evidence["complete_recovery_chain_verified"] is True


def test_manifest_history_accepts_checkpoint_bound_absent_reconciliation(
    tmp_path,
) -> None:
    root = tmp_path / "absent_run"
    root.mkdir()
    metrics = root / "train_metrics.jsonl"
    checkpoint, integrity = _write_checkpoint(root, 25)
    reconciliation = reconcile_metrics_for_resume(metrics, resume_step=25)
    history = append_metrics_resume_event(
        empty_metrics_resume_history(),
        output_dir=root,
        resume_checkpoint=checkpoint,
        checkpoint_integrity=integrity,
        reconciliation=reconciliation,
    )
    persist_metrics_resume_history(root, history)
    metrics.write_text(
        json.dumps(_resume_boundary_row(26), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = _manifest(dense=False)
    manifest.update(
        {
            "resume": checkpoint.resolve().as_posix(),
            "metrics_resume_reconciliation": reconciliation,
            "metrics_resume_history": history,
        }
    )
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    events, _, evidence = trajectory._select_reconciliation_history(
        manifest_path=manifest_path,
        manifest=manifest,
        metrics_path=metrics,
        manual_paths=None,
        label="cofitok",
    )

    assert events[0]["status"] == "absent"
    assert events[0]["retained_rows"] == 0
    assert evidence["complete_recovery_chain_verified"] is True


def test_main_automatically_consumes_manifest_resume_history(
    tmp_path,
    monkeypatch,
) -> None:
    cofitok = _history_backed_run(tmp_path, event_steps=(125, 175))
    dense_root = tmp_path / "dense_run"
    dense_root.mkdir()
    dense_metrics = dense_root / "train_metrics.jsonl"
    dense_metrics.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in _rows(dense=True)
        ),
        encoding="utf-8",
    )
    dense_manifest = dense_root / "run_manifest.json"
    dense_manifest.write_text(
        json.dumps(_manifest(dense=True), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "trajectory.json"
    monkeypatch.setattr(
        trajectory,
        "_builder_identity",
        lambda _: {
            "path": "scripts/build_generation_matched_training_trajectory.py",
            "bytes": 1,
            "sha256": "d" * 64,
            "git": {
                "revision": "e" * 40,
                "branch": "analysis/test",
                "tracked_dirty": False,
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_generation_matched_training_trajectory.py",
            "--cofitok-metrics",
            str(cofitok["metrics"]),
            "--dense-metrics",
            str(dense_metrics),
            "--cofitok-manifest",
            str(cofitok["manifest_path"]),
            "--dense-manifest",
            str(dense_manifest),
            "--cutoff-step",
            "200",
            "--expected-revision",
            REVISION,
            "--expected-branch",
            BRANCH,
            "--output",
            str(output),
        ],
    )

    trajectory.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    evidence = report["sources"]["cofitok_metrics_resume_history"]
    assert report["schema_version"] == 5
    assert evidence["mode"] == "manifest_metrics_resume_history_v1"
    assert evidence["checkpoint_bound_event_count"] == 2
    assert evidence["complete_recovery_chain_verified"] is True


def test_incomplete_legacy_history_downgrades_complete_recovery_claim(tmp_path) -> None:
    run = _history_backed_run(tmp_path)
    history = run["manifest"]["metrics_resume_history"]
    history["legacy_history_complete"] = False
    _rehash_history(history)
    (run["root"] / "metrics_resume_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    events, _, evidence = trajectory._select_reconciliation_history(
        manifest_path=run["manifest_path"],
        manifest=run["manifest"],
        metrics_path=run["metrics"],
        manual_paths=None,
        label="cofitok",
    )

    report = trajectory.build_report(
        cofitok_metrics={"rows": run["rows"]},
        dense_metrics=_metrics(dense=True),
        cofitok_manifest=run["manifest"],
        dense_manifest=_manifest(dense=True),
        cutoff_step=200,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        cofitok_reconciliation_history=events,
        cofitok_resume_history_evidence=evidence,
    )

    assert evidence["complete_recovery_chain_verified"] is False
    assert report["contract"]["complete_recovery_chain_verified"] is False
    assert report["claim_boundary"]["complete_recovery_chain_claim_allowed"] is False
    assert report["claim_boundary"]["resume_history_limitation"]


def test_report_rejects_reconciled_resume_without_bound_orphan_hash() -> None:
    cofitok_manifest = _manifest(dense=False)
    cofitok_manifest["resume"] = "/runs/checkpoint_step_00000125.pt"
    cofitok_manifest["metrics_resume_reconciliation"] = {
        "schema_version": 1,
        "status": "reconciled",
        "resume_step": 125,
        "retained_rows": 3,
        "orphaned_rows": 2,
        "orphan_archive": "/runs/train_metrics_orphaned_at_resume_00000125.jsonl",
        "orphan_sha256": "not-a-sha256",
    }

    with pytest.raises(ValueError, match="lacks bound orphan evidence"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=cofitok_manifest,
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_unbound_irregular_logging_step() -> None:
    cofitok = _metrics(dense=False)
    cofitok["rows"].insert(
        3,
        {
            "step": 126,
            "samples_seen": 1_008,
            "epsilon": 0.01,
            "rollout_consistency": 0.1,
            "rollout_consistency_scale": 0.63,
            "ema_teacher_consistency": 0.05,
            "ema_teacher_consistency_scale": 0.26,
        },
    )

    with pytest.raises(ValueError, match="irregular logging step|exact logging schedule"):
        trajectory.build_report(
            cofitok_metrics=cofitok,
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_resume_checkpoint_reconciliation_step_drift() -> None:
    cofitok_manifest = _manifest(dense=False)
    cofitok_manifest["resume"] = "/runs/checkpoint_step_00000125.pt"
    cofitok_manifest["metrics_resume_reconciliation"] = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 124,
        "retained_rows": 3,
        "orphaned_rows": 0,
        "orphan_archive": None,
        "orphan_sha256": None,
    }

    with pytest.raises(ValueError, match="reconciliation steps differ"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=cofitok_manifest,
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_resume_retained_row_count_drift() -> None:
    cofitok_manifest = _manifest(dense=False)
    cofitok_manifest["resume"] = "/runs/checkpoint_step_00000125.pt"
    cofitok_manifest["metrics_resume_reconciliation"] = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 125,
        "retained_rows": 2,
        "orphaned_rows": 0,
        "orphan_archive": None,
        "orphan_sha256": None,
    }
    cofitok = _metrics(dense=False)
    cofitok["rows"].insert(
        3,
        {
            "step": 126,
            "samples_seen": 1_008,
            "epsilon": 0.01,
            "rollout_consistency": 0.1,
            "rollout_consistency_scale": 0.63,
            "ema_teacher_consistency": 0.05,
            "ema_teacher_consistency_scale": 0.26,
        },
    )

    with pytest.raises(ValueError, match="retained-row count"):
        trajectory.build_report(
            cofitok_metrics=cofitok,
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=cofitok_manifest,
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_sample_accounting_drift() -> None:
    dense = _metrics(dense=True)
    dense["rows"][-1]["samples_seen"] -= 1

    with pytest.raises(ValueError, match="samples_seen is invalid"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=dense,
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_validation_provenance_drift() -> None:
    dense = _metrics(dense=True)
    dense["rows"][-1]["validation_noise_seed"] += 1

    with pytest.raises(ValueError, match="validation provenance differs"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=dense,
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_observed_schedule_scale_drift() -> None:
    dense = _metrics(dense=True)
    dense["rows"][-1]["rollout_consistency_scale"] = 0.9

    with pytest.raises(ValueError, match="full-scale phase"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=dense,
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


@pytest.mark.parametrize(
    ("step", "contract", "expected"),
    [
        (50, {"enabled": False, "start_step": 0, "warmup_steps": 100}, "disabled"),
        (50, {"enabled": True, "start_step": 100, "warmup_steps": 100}, "inactive"),
        (100, {"enabled": True, "start_step": 100, "warmup_steps": 100}, "warmup"),
        (200, {"enabled": True, "start_step": 100, "warmup_steps": 100}, "full_scale"),
    ],
)
def test_schedule_phase_boundaries(
    step: int,
    contract: dict,
    expected: str,
) -> None:
    assert trajectory._schedule_phase(step, contract) == expected


def test_report_rejects_runtime_or_pair_contract_drift() -> None:
    dense_manifest = _manifest(dense=True)
    dense_manifest["runtime_environment_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="runtime_environment_sha256"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=dense_manifest,
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )

    dense_manifest = _manifest(dense=True)
    dense_manifest["config"]["data"]["batch_size"] = 8
    with pytest.raises(ValueError, match="training pair contract failed"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=dense_manifest,
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_metrics_prefix_identity_is_stable_after_later_rows(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = _rows(dense=False)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    before = trajectory._read_metrics_prefix(path, cutoff_step=200)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step": 250,
                    "samples_seen": 2_000,
                    "epsilon": 0.01,
                    "rollout_consistency": 0.1,
                    "rollout_consistency_scale": 1.0,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    after = trajectory._read_metrics_prefix(path, cutoff_step=200)

    assert before["identity"] == after["identity"]
    assert before["observed_file"]["sha256"] != after["observed_file"]["sha256"]


def test_builder_identity_binds_clean_git_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    script = tmp_path / "scripts" / "builder.py"
    script.parent.mkdir()
    script.write_text("print('builder')\n", encoding="utf-8")
    expected_git = {
        "revision": "a" * 40,
        "branch": "analysis/test-builder",
        "tracked_dirty": False,
    }
    monkeypatch.setattr(trajectory, "git_provenance", lambda _: expected_git)

    identity = trajectory._builder_identity(script)

    assert identity["path"] == "scripts/builder.py"
    assert identity["bytes"] == script.stat().st_size
    assert identity["git"] == expected_git


def test_builder_identity_rejects_dirty_checkout(tmp_path, monkeypatch) -> None:
    script = tmp_path / "scripts" / "builder.py"
    script.parent.mkdir()
    script.write_text("print('builder')\n", encoding="utf-8")
    monkeypatch.setattr(
        trajectory,
        "git_provenance",
        lambda _: {
            "revision": "a" * 40,
            "branch": "analysis/test-builder",
            "tracked_dirty": True,
        },
    )

    with pytest.raises(ValueError, match="Git provenance is invalid or dirty"):
        trajectory._builder_identity(script)


def test_report_rejects_nonfinite_shared_metrics() -> None:
    dense = _metrics(dense=True)
    dense["rows"][-1]["epsilon"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=dense,
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=_manifest(dense=True),
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_dirty_or_wrong_git_identity() -> None:
    dense_manifest = deepcopy(_manifest(dense=True))
    dense_manifest["git"]["dirty"] = True

    with pytest.raises(ValueError, match="dirty tracked worktree"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=dense_manifest,
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_report_rejects_nonformal_dataset_provenance() -> None:
    dense_manifest = deepcopy(_manifest(dense=True))
    dense_manifest["dataset_provenance"]["formal"] = False

    with pytest.raises(ValueError, match="lacks passing dataset provenance"):
        trajectory.build_report(
            cofitok_metrics=_metrics(dense=False),
            dense_metrics=_metrics(dense=True),
            cofitok_manifest=_manifest(dense=False),
            dense_manifest=dense_manifest,
            cutoff_step=200,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )
