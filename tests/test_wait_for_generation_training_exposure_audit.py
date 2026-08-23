from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cofitok.reporting import file_sha256
from scripts import wait_for_generation_training_exposure_audit as waiter


RUNBOOK = Path(
    "artifacts/runbooks/generation_quality_bridge_training_exposure_50k_waiter.sh"
)
TERMINAL_RUNBOOK = Path(
    "artifacts/runbooks/generation_quality_bridge_terminal_training_exposure_waiter.sh"
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
        quality_bridge_terminal_result=None,
        quality_bridge_execution_status=None,
        quality_bridge_verifier_project=None,
        quality_bridge_verifier_python=None,
        expected_quality_bridge_revision=None,
        expected_quality_bridge_tree=None,
        expected_quality_bridge_branch=None,
        expected_quality_bridge_verifier_sha256=None,
        expected_quality_bridge_builder_sha256=None,
        expected_quality_bridge_result_sha256=None,
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
    quality_bridge_terminal_result=None,
    quality_bridge_execution_status=None,
    quality_bridge_terminal_verification=None,
):
    report = {
        "status": "pass",
        "training_sources": {
            name: identity for name, (_, identity) in training_reports.items()
        },
        "preparation_source": quality_bridge_preparation[1],
        "milestone_source": milestone_report[1],
        "expected_milestone_step": expected_milestone_step,
        "milestone_source_profile": milestone_source_profile,
    }
    if quality_bridge_terminal_result is not None:
        report["terminal_source"] = quality_bridge_terminal_result[1]
        report["execution_source"] = quality_bridge_execution_status[1]
        report["terminal_verification"] = quality_bridge_terminal_verification
    return report


def _verification(terminal_result: Path, *, reported_path: Path) -> dict:
    identity = waiter._identity(terminal_result, reported_path=reported_path)
    terminal = json.loads(terminal_result.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "role": "generation_quality_bridge_authoritative_terminal_verification",
        "status": "verified",
        "quality_project": {
            "revision": "q" * 40,
            "tree": "t" * 40,
            "branch": "scale/quality",
            "tracked_dirty": False,
            "path": "/quality/project",
        },
        "verifier_source": {
            "path": "/quality/project/scripts/verify.py",
            "bytes": 1,
            "sha256": "a" * 64,
        },
        "builder_source": {
            "path": "/quality/project/scripts/build.py",
            "bytes": 1,
            "sha256": "b" * 64,
        },
        "python": {
            "path": "/python",
            "bytes": 1,
            "sha256": "c" * 64,
        },
        "terminal_result": identity,
        "verifier_output": {
            "status": "verified",
            "result": identity,
            "quality_screen": terminal.get("quality_screen"),
            "authorization_boundary": terminal.get("authorization_boundary"),
        },
        "execution_policy": {
            "cuda_visible_devices": "-1",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
        },
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


def test_waiter_freezes_only_after_verified_terminal_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.expected_milestone_step = 100_000
    args.quality_bridge_terminal_result = tmp_path / "quality_bridge_result.json"
    args.quality_bridge_execution_status = tmp_path / "execution_status.json"
    args.quality_bridge_verifier_project = tmp_path / "quality_project"
    args.quality_bridge_verifier_python = Path(sys.executable)
    args.expected_quality_bridge_revision = "q" * 40
    args.expected_quality_bridge_tree = "t" * 40
    args.expected_quality_bridge_branch = "scale/quality"
    args.expected_quality_bridge_verifier_sha256 = "a" * 64
    args.expected_quality_bridge_builder_sha256 = "b" * 64
    _write(args.cofitok_training, _training(100_000))
    _write(args.dense_training, _training(100_000))
    _write(args.milestone_report, {"status": "completed"})
    monkeypatch.setattr(waiter, "build_report", _fake_build_report)
    self_git = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "scale/test",
        "tracked_dirty": False,
    }

    assert waiter.poll_once(args, self_git=self_git)[1] == (
        "waiting_for_quality_bridge_terminal_result"
    )
    _write(args.quality_bridge_terminal_result, {"status": "completed"})
    _write(args.quality_bridge_execution_status, {"status": "running"})
    assert waiter.poll_once(args, self_git=self_git)[1] == (
        "waiting_for_quality_bridge_verified_completion"
    )
    _write(args.quality_bridge_execution_status, {"status": "completed"})
    args.expected_quality_bridge_result_sha256 = file_sha256(
        args.quality_bridge_terminal_result
    )

    def fake_verification(_args, *, reported_terminal_result):
        return _verification(
            args.quality_bridge_terminal_result,
            reported_path=reported_terminal_result,
        )

    monkeypatch.setattr(
        waiter,
        "_run_authoritative_quality_bridge_verifier",
        fake_verification,
    )
    monkeypatch.setattr(
        waiter,
        "_authoritative_quality_sources",
        lambda _args: {
            name: value
            for name, value in fake_verification(
                _args,
                reported_terminal_result=(
                    args.output_root
                    / "source"
                    / "quality_bridge_terminal_result.json"
                ),
            ).items()
            if name
            in {
                "quality_project",
                "verifier_source",
                "builder_source",
                "python",
            }
        },
    )

    first = waiter.poll_once(args, self_git=self_git)
    second = waiter.poll_once(args, self_git=self_git)

    assert first[0:2] == (
        "completed",
        "matched_terminal_training_exposure_frozen",
    )
    assert second[0:2] == first[0:2]
    manifest = json.loads(
        (args.output_root / waiter.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["binding_mode"] == "quality_bridge_terminal"
    assert (
        args.output_root / "source" / "quality_bridge_terminal_result.json"
    ).is_file()
    assert (
        args.output_root / "source" / "quality_bridge_execution_status.json"
    ).is_file()
    assert (
        args.output_root
        / "source"
        / waiter.AUTHORITATIVE_TERMINAL_VERIFICATION_FILENAME
    ).is_file()


def test_authoritative_verifier_receipt_normalizes_snapshot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.quality_bridge_verifier_project = tmp_path / "quality_project"
    args.quality_bridge_verifier_python = Path(sys.executable)
    args.expected_quality_bridge_revision = "q" * 40
    args.expected_quality_bridge_tree = "t" * 40
    args.expected_quality_bridge_branch = "scale/quality"
    args.expected_quality_bridge_verifier_sha256 = "a" * 64
    args.expected_quality_bridge_builder_sha256 = "b" * 64
    args.quality_bridge_terminal_result = tmp_path / "quality_bridge_result.json"
    sources = {
        name: {
            "path": (tmp_path / f"{name}.json").as_posix(),
            "bytes": 1,
            "sha256": "d" * 64,
        }
        for name in waiter.QUALITY_BRIDGE_SOURCE_ARGUMENTS
    }
    terminal = {
        "git": {
            "revision": args.expected_quality_bridge_revision,
            "branch": args.expected_quality_bridge_branch,
            "tracked_dirty": False,
        },
        "source_reports": sources,
        "quality_screen": {"status": "hold"},
        "authorization_boundary": {"full_300k_launch_allowed": False},
    }
    _write(args.quality_bridge_terminal_result, terminal)
    args.expected_quality_bridge_result_sha256 = file_sha256(
        args.quality_bridge_terminal_result
    )
    binding = {
        "quality_project": {
            "revision": args.expected_quality_bridge_revision,
            "tree": args.expected_quality_bridge_tree,
            "branch": args.expected_quality_bridge_branch,
            "tracked_dirty": False,
            "path": args.quality_bridge_verifier_project.resolve().as_posix(),
        },
        "verifier_source": {
            "path": "/quality/verifier.py",
            "bytes": 1,
            "sha256": "a" * 64,
        },
        "builder_source": {
            "path": "/quality/builder.py",
            "bytes": 1,
            "sha256": "b" * 64,
        },
        "python": {
            "path": Path(sys.executable).resolve().as_posix(),
            "bytes": 1,
            "sha256": "c" * 64,
        },
    }
    monkeypatch.setattr(
        waiter,
        "_authoritative_quality_sources",
        lambda _args: copy.deepcopy(binding),
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        actual_identity = waiter._identity(args.quality_bridge_terminal_result)
        output = {
            "status": "verified",
            "result": actual_identity,
            "quality_screen": terminal["quality_screen"],
            "authorization_boundary": terminal["authorization_boundary"],
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(output),
            stderr="",
        )

    monkeypatch.setattr(waiter.subprocess, "run", fake_run)
    reported = tmp_path / "snapshot" / "source" / "quality_bridge_result.json"

    receipt = waiter._run_authoritative_quality_bridge_verifier(
        args,
        reported_terminal_result=reported,
    )

    assert receipt["terminal_result"]["path"] == reported.resolve().as_posix()
    assert receipt["verifier_output"]["result"] == receipt["terminal_result"]
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert "--expected-result-sha256" in captured["command"]


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


def test_quality_bridge_terminal_waiter_is_cpu_only_and_exact_stage() -> None:
    source = TERMINAL_RUNBOOK.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=-1" in source
    assert "--expected-milestone-step 100000" in source
    assert "--milestone-source-profile quality_bridge" in source
    assert "--quality-bridge-terminal-result" in source
    assert "--quality-bridge-execution-status" in source
    assert "--quality-bridge-verifier-project" in source
    assert "--quality-bridge-verifier-python" in source
    assert "--expected-quality-bridge-revision" in source
    assert "--expected-quality-bridge-tree" in source
    assert "--expected-quality-bridge-verifier-sha256" in source
    assert "--expected-quality-bridge-builder-sha256" in source
    assert "--expected-quality-bridge-result-sha256" in source
    assert "authoritative_verifier_v2" in source
    assert "step_00100000.json" in source
    assert "--poll-seconds 10" in source
    assert "EXPECTED_SELF_REVISION=${EXPECTED_SELF_REVISION:?" in source
    assert "EXPECTED_SELF_TREE=${EXPECTED_SELF_TREE:?" in source
    assert "EXPECTED_SELF_BRANCH=${EXPECTED_SELF_BRANCH:?" in source
    assert "training_launch_allowed" not in source
