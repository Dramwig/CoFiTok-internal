from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cofitok.training.metrics import (
    ensure_fresh_training_output,
    reconcile_metrics_for_resume,
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
    ],
)
def test_fresh_run_rejects_any_existing_training_state(tmp_path, name: str) -> None:
    (tmp_path / name).write_text("state", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --resume auto"):
        ensure_fresh_training_output(tmp_path)


def test_fresh_run_accepts_empty_output_directory(tmp_path) -> None:
    ensure_fresh_training_output(tmp_path)
