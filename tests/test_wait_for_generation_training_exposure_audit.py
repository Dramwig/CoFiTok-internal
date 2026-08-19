from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from cofitok.reporting import file_sha256
from scripts import wait_for_generation_training_exposure_audit as waiter


RUNBOOK = Path(
    "artifacts/runbooks/generation_quality_bridge_training_exposure_50k_waiter.sh"
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _training(step: int) -> dict:
    return {
        "completed_steps": step,
        "latest_checkpoint": {
            "step": step,
            "checkpoint_sha256": "e" * 64,
        },
    }


def _args(tmp_path: Path) -> argparse.Namespace:
    preparation = tmp_path / "preparation.json"
    _write(preparation, {"status": "prepared"})
    return argparse.Namespace(
        project=tmp_path,
        cofitok_training=tmp_path / "cofitok_training.json",
        dense_training=tmp_path / "dense_training.json",
        quality_bridge_preparation=preparation,
        expected_preparation_sha256=file_sha256(preparation),
        milestone_report=tmp_path / "milestone.json",
        expected_milestone_step=50_000,
        milestone_source_profile="quality_bridge",
        output_root=tmp_path / "snapshot",
        status=tmp_path / "waiter_status.json",
        expected_self_revision="a" * 40,
        expected_self_tree="b" * 40,
        expected_self_branch="scale/test",
        poll_seconds=1,
    )


def _fake_build_report(
    training_reports,
    *,
    quality_bridge_preparation,
    milestone_report,
    expected_milestone_step,
    milestone_source_profile,
):
    return {
        "status": "pass",
        "training_sources": {
            name: identity for name, (_, identity) in training_reports.items()
        },
        "preparation_source": quality_bridge_preparation[1],
        "milestone_source": milestone_report[1],
        "expected_milestone_step": expected_milestone_step,
        "milestone_source_profile": milestone_source_profile,
    }


def test_waiter_waits_for_dense_training_report(tmp_path: Path) -> None:
    args = _args(tmp_path)
    _write(args.cofitok_training, _training(50_000))

    status, detail, audit, manifest = waiter.poll_once(
        args,
        self_git={"revision": "a" * 40},
    )

    assert status == "waiting"
    assert detail == "waiting_for_dense_training_milestone"
    assert audit is None
    assert manifest is None


def test_waiter_snapshots_and_replays_exact_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    _write(args.cofitok_training, _training(50_000))
    _write(args.dense_training, _training(50_000))
    _write(args.milestone_report, {"status": "completed"})
    monkeypatch.setattr(waiter, "build_report", _fake_build_report)
    self_git = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "scale/test",
        "tracked_dirty": False,
    }

    first = waiter.poll_once(args, self_git=self_git)
    second = waiter.poll_once(args, self_git=self_git)

    assert first[0:2] == (
        "completed",
        "matched_milestone_training_exposure_frozen",
    )
    assert second[0:2] == first[0:2]
    assert first[2] == second[2]
    assert (args.output_root / waiter.AUDIT_FILENAME).is_file()
    assert (args.output_root / waiter.MANIFEST_FILENAME).is_file()
    snapshotted = json.loads(
        (args.output_root / "source" / "cofitok_training_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshotted == _training(50_000)


def test_waiter_rejects_report_that_advanced_past_milestone(tmp_path: Path) -> None:
    args = _args(tmp_path)
    _write(args.cofitok_training, _training(100_000))

    with pytest.raises(ValueError, match="advanced past"):
        waiter.poll_once(args, self_git={"revision": "a" * 40})


def test_waiter_rejects_preparation_identity_change(tmp_path: Path) -> None:
    args = _args(tmp_path)
    changed = copy.deepcopy({"status": "prepared"})
    changed["changed"] = True
    _write(args.quality_bridge_preparation, changed)

    with pytest.raises(ValueError, match="preparation SHA256 differs"):
        waiter.poll_once(args, self_git={"revision": "a" * 40})


def test_waiter_rejects_existing_snapshot_from_another_code_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    _write(args.cofitok_training, _training(50_000))
    _write(args.dense_training, _training(50_000))
    _write(args.milestone_report, {"status": "completed"})
    monkeypatch.setattr(waiter, "build_report", _fake_build_report)
    self_git = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "scale/test",
        "tracked_dirty": False,
    }
    waiter.poll_once(args, self_git=self_git)
    args.expected_self_revision = "f" * 40

    with pytest.raises(ValueError, match="code identity differs"):
        waiter.poll_once(args, self_git=self_git)


def test_waiter_rejects_tampered_existing_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    _write(args.cofitok_training, _training(50_000))
    _write(args.dense_training, _training(50_000))
    _write(args.milestone_report, {"status": "completed"})
    monkeypatch.setattr(waiter, "build_report", _fake_build_report)
    self_git = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "scale/test",
        "tracked_dirty": False,
    }
    waiter.poll_once(args, self_git=self_git)
    audit_path = args.output_root / waiter.AUDIT_FILENAME
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["status"] = "tampered"
    _write(audit_path, audit)

    with pytest.raises(ValueError, match="not reproducible"):
        waiter.poll_once(args, self_git=self_git)


def test_quality_bridge_waiter_runbook_is_cpu_only_and_exact_stage() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=-1" in source
    assert "--expected-milestone-step 50000" in source
    assert "--milestone-source-profile quality_bridge" in source
    assert "--poll-seconds 10" in source
    assert "EXPECTED_SELF_REVISION=${EXPECTED_SELF_REVISION:?" in source
    assert "EXPECTED_SELF_TREE=${EXPECTED_SELF_TREE:?" in source
    assert "EXPECTED_SELF_BRANCH=${EXPECTED_SELF_BRANCH:?" in source
    assert "training_launch_allowed" not in source
