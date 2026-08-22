from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cofitok.training.metrics import (
    append_metrics_resume_event,
    empty_metrics_resume_history,
    ensure_fresh_training_output,
    persist_metrics_resume_history,
    reconcile_metrics_for_resume,
    restore_metrics_resume_history,
    validate_metrics_resume_history,
    verify_metrics_resume_history_evidence,
)


def _write_rows(path: Path, steps: list[int]) -> list[str]:
    lines = [json.dumps({"step": step, "loss": step / 100.0}) + "\n" for step in steps]
    path.write_text("".join(lines), encoding="utf-8")
    return lines


def _steps(path: Path) -> list[int]:
    return [
        int(json.loads(line)["step"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_checkpoint(
    root: Path,
    step: int,
    *,
    marker: str = "",
) -> tuple[Path, dict[str, object]]:
    infix = f"_{marker}" if marker else ""
    checkpoint = root / f"checkpoint{infix}_step_{step:08d}.pt"
    payload = f"checkpoint:{step}:{marker}".encode("utf-8")
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


def _rehash_history(history: dict[str, object]) -> None:
    payload = {
        "legacy_history_complete": history["legacy_history_complete"],
        "legacy_reconciliations": history["legacy_reconciliations"],
        "events": history["events"],
    }
    history["history_sha256"] = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def test_resume_archives_rows_after_checkpoint_and_rewrites_atomically(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    lines = _write_rows(metrics, [1, 50, 100, 150])

    report = reconcile_metrics_for_resume(metrics, resume_step=100)

    assert report["status"] == "reconciled"
    assert report["retained_rows"] == 3
    assert report["orphaned_rows"] == 1
    assert _steps(metrics) == [1, 50, 100]
    archive = Path(report["orphan_archive"])
    assert archive.read_text(encoding="utf-8") == lines[-1]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == report["orphan_sha256"]
    assert Path(report["report"]).is_file()


def test_resume_keeps_latest_duplicate_trajectory_before_checkpoint(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    lines = _write_rows(metrics, [1, 50, 100, 75, 100, 125])

    report = reconcile_metrics_for_resume(metrics, resume_step=100)

    assert _steps(metrics) == [1, 50, 75, 100]
    archive = Path(report["orphan_archive"])
    assert archive.read_text(encoding="utf-8") == lines[2] + lines[5]


def test_resume_without_orphans_leaves_metrics_unchanged(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    original = "".join(_write_rows(metrics, [1, 50, 100]))

    report = reconcile_metrics_for_resume(metrics, resume_step=100)

    assert report["status"] == "unchanged"
    assert metrics.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*orphaned*"))
    assert Path(report["report"]).is_file()
    assert report["report_sha256"] == hashlib.sha256(
        Path(report["report"]).read_bytes()
    ).hexdigest()
    assert report["metrics_sha256"] == hashlib.sha256(metrics.read_bytes()).hexdigest()


def test_resume_rejects_invalid_metrics_without_modifying_them(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    original = '{"step": 1}\nnot-json\n'
    metrics.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid metrics JSON"):
        reconcile_metrics_for_resume(metrics, resume_step=1)

    assert metrics.read_text(encoding="utf-8") == original


def test_resume_with_no_metrics_reports_absent(tmp_path) -> None:
    report = reconcile_metrics_for_resume(
        tmp_path / "train_metrics.jsonl",
        resume_step=500,
    )

    assert report["status"] == "absent"
    assert report["orphaned_rows"] == 0
    assert Path(report["report"]).is_file()
    assert report["metrics_bytes"] == 0


def test_resume_history_preserves_three_strict_events_and_longest_prefix(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    history = empty_metrics_resume_history()
    reconciliations: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []

    for step in (1, 2, 3):
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "loss": step / 10.0}) + "\n")
        checkpoint, integrity = _write_checkpoint(tmp_path, step)
        reconciliation = reconcile_metrics_for_resume(metrics, resume_step=step)
        history = append_metrics_resume_event(
            history,
            output_dir=tmp_path,
            resume_checkpoint=checkpoint,
            checkpoint_integrity=integrity,
            reconciliation=reconciliation,
        )
        reconciliations.append(reconciliation)
        snapshots.append(json.loads(json.dumps(history)))

    persist_metrics_resume_history(tmp_path, history)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics_resume_history": snapshots[-1],
                "metrics_resume_reconciliation": reconciliations[-1],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "training_report.json").write_text(
        json.dumps(
            {
                "metrics_resume_history": snapshots[1],
                "metrics_resume_reconciliation": reconciliations[1],
            }
        ),
        encoding="utf-8",
    )

    restored = restore_metrics_resume_history(
        tmp_path,
        checkpoint_extra_state={
            "metrics_resume_history": snapshots[0],
            "metrics_resume_reconciliation": reconciliations[0],
        },
    )

    assert restored == history
    assert [event["resume_step"] for event in restored["events"]] == [1, 2, 3]
    assert [event["event_index"] for event in restored["events"]] == [1, 2, 3]
    assert restored["events"][-1]["resume_step"] == 3


@pytest.mark.parametrize("drift", ["report", "orphan", "metrics"])
def test_resume_history_rejects_physical_artifact_drift(tmp_path, drift: str) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    _write_rows(metrics, [1, 2])
    checkpoint, integrity = _write_checkpoint(tmp_path, 1)
    reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)
    history = append_metrics_resume_event(
        empty_metrics_resume_history(),
        output_dir=tmp_path,
        resume_checkpoint=checkpoint,
        checkpoint_integrity=integrity,
        reconciliation=reconciliation,
    )

    if drift == "report":
        with Path(reconciliation["report"]).open("a", encoding="utf-8") as handle:
            handle.write(" ")
    elif drift == "orphan":
        with Path(reconciliation["orphan_archive"]).open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
    else:
        content = metrics.read_text(encoding="utf-8")
        metrics.write_text(content.replace('"step": 1', '"step": 9', 1), encoding="utf-8")

    with pytest.raises((ValueError, FileNotFoundError)):
        validate_metrics_resume_history(history, output_dir=tmp_path)


def test_resume_history_rejects_checkpoint_and_manifest_divergence(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    _write_rows(metrics, [1])
    reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)
    checkpoint, integrity = _write_checkpoint(tmp_path, 1)
    checkpoint_history = append_metrics_resume_event(
        empty_metrics_resume_history(),
        output_dir=tmp_path,
        resume_checkpoint=checkpoint,
        checkpoint_integrity=integrity,
        reconciliation=reconciliation,
    )
    alternative, alternative_integrity = _write_checkpoint(tmp_path, 1, marker="other")
    manifest_history = append_metrics_resume_event(
        empty_metrics_resume_history(),
        output_dir=tmp_path,
        resume_checkpoint=alternative,
        checkpoint_integrity=alternative_integrity,
        reconciliation=reconciliation,
    )
    persist_metrics_resume_history(tmp_path, checkpoint_history)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics_resume_history": manifest_history,
                "metrics_resume_reconciliation": reconciliation,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sources diverge"):
        restore_metrics_resume_history(
            tmp_path,
            checkpoint_extra_state={
                "metrics_resume_history": checkpoint_history,
                "metrics_resume_reconciliation": reconciliation,
            },
        )


@pytest.mark.parametrize("source", ["checkpoint", "run_manifest.json", "training_report.json"])
def test_resume_history_migrates_legacy_reconciliation_without_fabricated_checkpoint_binding(
    tmp_path, source: str
) -> None:
    legacy = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 10,
        "metrics": (tmp_path / "train_metrics.jsonl").resolve().as_posix(),
        "retained_rows": 1,
        "orphaned_rows": 0,
        "orphan_archive": None,
        "orphan_sha256": None,
    }
    checkpoint_state: dict[str, object] = {}
    if source == "checkpoint":
        checkpoint_state["metrics_resume_reconciliation"] = legacy
    else:
        (tmp_path / source).write_text(
            json.dumps({"metrics_resume_reconciliation": legacy}),
            encoding="utf-8",
        )

    history = restore_metrics_resume_history(
        tmp_path,
        checkpoint_extra_state=checkpoint_state,
    )

    assert history["events"] == []
    assert history["legacy_history_complete"] is False
    assert len(history["legacy_reconciliations"]) == 1
    assert history["legacy_history_complete"] is False
    migrated = history["legacy_reconciliations"][0]
    assert migrated["resume_step"] == 10
    assert migrated["checkpoint_binding_available"] is False
    assert migrated["report"] is None


def test_resume_history_discovers_and_physically_binds_legacy_report(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    _write_rows(metrics, [1, 2])
    reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)
    report_path = Path(reconciliation["report"])

    history = restore_metrics_resume_history(tmp_path, checkpoint_extra_state={})

    assert len(history["legacy_reconciliations"]) == 1
    migrated = history["legacy_reconciliations"][0]
    assert migrated["report"]["sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert migrated["orphan_archive"]["sha256"] == reconciliation["orphan_sha256"]


def test_resume_history_can_upgrade_at_the_latest_legacy_resume_step(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    _write_rows(metrics, [1])
    legacy_reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)
    history = restore_metrics_resume_history(tmp_path, checkpoint_extra_state={})
    checkpoint, integrity = _write_checkpoint(tmp_path, 1)
    current_reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)

    upgraded = append_metrics_resume_event(
        history,
        output_dir=tmp_path,
        resume_checkpoint=checkpoint,
        checkpoint_integrity=integrity,
        reconciliation=current_reconciliation,
    )

    assert legacy_reconciliation["resume_step"] == 1
    assert upgraded["legacy_reconciliations"][0]["resume_step"] == 1
    assert upgraded["events"][0]["resume_step"] == 1


def test_resume_history_repeated_step_is_idempotent_but_cannot_drift(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    _write_rows(metrics, [1])
    checkpoint, integrity = _write_checkpoint(tmp_path, 1)
    reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)
    history = append_metrics_resume_event(
        empty_metrics_resume_history(),
        output_dir=tmp_path,
        resume_checkpoint=checkpoint,
        checkpoint_integrity=integrity,
        reconciliation=reconciliation,
    )

    assert append_metrics_resume_event(
        history,
        output_dir=tmp_path,
        resume_checkpoint=checkpoint,
        checkpoint_integrity=integrity,
        reconciliation=reconciliation,
    ) == history

    with metrics.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 2, "loss": 0.2}) + "\n")
    drifted_reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)
    with pytest.raises(ValueError, match="repeated resume step differs"):
        append_metrics_resume_event(
            history,
            output_dir=tmp_path,
            resume_checkpoint=checkpoint,
            checkpoint_integrity=integrity,
            reconciliation=drifted_reconciliation,
        )


def test_resume_history_rejects_bound_path_drift_even_with_recomputed_digest(tmp_path) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    _write_rows(metrics, [1])
    checkpoint, integrity = _write_checkpoint(tmp_path, 1)
    reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)
    history = append_metrics_resume_event(
        empty_metrics_resume_history(),
        output_dir=tmp_path,
        resume_checkpoint=checkpoint,
        checkpoint_integrity=integrity,
        reconciliation=reconciliation,
    )
    drifted = json.loads(json.dumps(history))
    drifted["events"][0]["reconciliation"]["report"]["path"] = "other-report.json"
    _rehash_history(drifted)

    with pytest.raises(FileNotFoundError, match="metrics reconciliation report"):
        validate_metrics_resume_history(drifted, output_dir=tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "run_manifest.json",
        "training_report.json",
        "latest.json",
        "train_metrics.jsonl",
        "checkpoint_step_00000010.pt.tmp-7",
        "train_metrics_orphaned_at_resume_00000010_deadbeef.jsonl",
        "metrics_resume_reconciliation_00000010_deadbeef.json",
        "metrics_resume_history.json",
    ],
)
def test_fresh_run_rejects_any_existing_training_state(tmp_path, name: str) -> None:
    (tmp_path / name).write_text("state", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --resume auto"):
        ensure_fresh_training_output(tmp_path)


def test_fresh_run_accepts_empty_output_directory(tmp_path) -> None:
    ensure_fresh_training_output(tmp_path)
    assert empty_metrics_resume_history()["legacy_history_complete"] is True


def _write_completed_resume_documents(
    root: Path,
    *,
    legacy_history_complete: bool = True,
) -> tuple[Path, Path, Path, dict[str, object]]:
    metrics = root / "train_metrics.jsonl"
    _write_rows(metrics, [1, 2])
    checkpoint, integrity = _write_checkpoint(root, 1)
    reconciliation = reconcile_metrics_for_resume(metrics, resume_step=1)
    history = append_metrics_resume_event(
        empty_metrics_resume_history(),
        output_dir=root,
        resume_checkpoint=checkpoint,
        checkpoint_integrity=integrity,
        reconciliation=reconciliation,
    )
    history["legacy_history_complete"] = legacy_history_complete
    _rehash_history(history)
    persist_metrics_resume_history(root, history)
    document = {
        "output_dir": root.resolve().as_posix(),
        "resume": checkpoint.resolve().as_posix(),
        "metrics_resume_reconciliation": reconciliation,
        "metrics_resume_history": history,
    }
    manifest = root / "run_manifest.json"
    training_report = root / "training_report.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    training_report.write_text(json.dumps(document), encoding="utf-8")
    return manifest, training_report, metrics, document


def test_resume_history_evidence_binds_manifest_journal_and_training_report(
    tmp_path,
) -> None:
    manifest, training_report, metrics, document = _write_completed_resume_documents(
        tmp_path
    )

    verified = verify_metrics_resume_history_evidence(
        manifest_path=manifest,
        metrics_path=metrics,
        manifest=document,
        training_report_path=training_report,
        training_report=document,
        require_complete=True,
        label="cofitok",
    )

    evidence = verified["evidence"]
    assert evidence["status"] == "verified"
    assert evidence["checkpoint_bound_event_count"] == 1
    assert evidence["manifest_journal_equality_verified"] is True
    assert evidence["training_report_history_equality_verified"] is True
    assert evidence["current_training_report_binding_verified"] is True
    assert evidence["complete_recovery_chain_verified"] is True
    assert evidence["checkpoint_payload_sha256_recomputed"] is False


def test_resume_history_evidence_rejects_training_report_current_binding_drift(
    tmp_path,
) -> None:
    manifest, training_report, metrics, document = _write_completed_resume_documents(
        tmp_path
    )
    drifted = json.loads(json.dumps(document))
    drifted["metrics_resume_reconciliation"]["retained_rows"] += 1
    training_report.write_text(json.dumps(drifted), encoding="utf-8")

    with pytest.raises(ValueError, match="latest history event"):
        verify_metrics_resume_history_evidence(
            manifest_path=manifest,
            metrics_path=metrics,
            manifest=document,
            training_report_path=training_report,
            training_report=drifted,
            require_complete=True,
            label="cofitok",
        )


def test_resume_history_evidence_explicitly_downgrades_legacy_history(
    tmp_path,
) -> None:
    manifest, training_report, metrics, document = _write_completed_resume_documents(
        tmp_path,
        legacy_history_complete=False,
    )

    degraded = verify_metrics_resume_history_evidence(
        manifest_path=manifest,
        metrics_path=metrics,
        manifest=document,
        training_report_path=training_report,
        training_report=document,
        require_complete=False,
        label="legacy",
    )
    assert degraded["evidence"]["complete_recovery_chain_verified"] is False
    assert degraded["evidence"]["limitation"]

    with pytest.raises(ValueError, match="complete checkpoint-bound"):
        verify_metrics_resume_history_evidence(
            manifest_path=manifest,
            metrics_path=metrics,
            manifest=document,
            training_report_path=training_report,
            training_report=document,
            require_complete=True,
            label="legacy",
        )
