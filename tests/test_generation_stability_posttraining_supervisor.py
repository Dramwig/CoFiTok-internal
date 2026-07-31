from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cofitok.reporting import file_sha256
from scripts import run_generation_stability_posttraining_supervisor as supervisor


TRAINING_REVISION = "a" * 40
TRAINING_BRANCH = "scale/generation-large-capacity"
EVALUATION_REVISION = "b" * 40
EVALUATION_BRANCH = "scale/generation-stability-full-eval"


def _monitor(*, status: str = "pass") -> dict:
    complete = status == "pass"
    runs = {}
    for method in ("cofitok", "dense_identity"):
        runs[method] = {
            "complete": complete,
            "expected_steps": 300_000,
            "last_step": 300_000 if complete else 200_000,
            "health_issues": [],
        }
    return {
        "schema_version": 2,
        "monitor": supervisor.FULL_MONITOR_NAME,
        "status": status,
        "stage": "complete" if complete else "cofitok_training",
        "issues": [],
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        },
        "runs": runs,
        "updated_at": "2026-07-31T00:00:00+00:00",
    }


def test_full_monitor_requires_both_exact_300k_runs() -> None:
    observation = supervisor.validate_full_monitor(
        _monitor(),
        expected_revision=TRAINING_REVISION,
        expected_branch=TRAINING_BRANCH,
    )
    assert observation["complete"] is True

    report = _monitor()
    report["runs"]["dense_identity"]["last_step"] = 299_999
    with pytest.raises(ValueError, match="dense_identity"):
        supervisor.validate_full_monitor(
            report,
            expected_revision=TRAINING_REVISION,
            expected_branch=TRAINING_BRANCH,
        )


def test_full_launch_receipt_rehashes_every_bound_source(tmp_path: Path) -> None:
    full_root = tmp_path / "stability_full_300k_ema_teacher"
    sources = {}
    for name in (
        "deployment_receipt",
        "promotion_gate",
        "full_readiness",
        "cofitok_config",
        "dense_config",
        "config_validation",
        "storage_capacity",
        "runtime_selection",
        "launch_storage_capacity",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        sources[name] = supervisor._source_identity(path)
    sources["promotion_gate"]["sha256"] = "c" * 64
    sources["full_readiness"]["sha256"] = "d" * 64
    report = {
        "schema_version": 1,
        "status": "pass",
        "role": supervisor.FULL_LAUNCH_ROLE,
        "stage": "stability_full",
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        },
        "readiness_sha256": "d" * 64,
        "full_training_launch_authorized": True,
        "formal_generation_completion_claimed": False,
        "source_reports": sources,
        "training_run_dirs": [
            (full_root / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher").as_posix(),
            (full_root / "dense_rollout_x0_u2_ema_teacher").as_posix(),
        ],
    }
    receipt = tmp_path / "full_training_launch_receipt.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")

    # Restore the two intentionally synthetic hashes through matching file identities.
    for name, digest in (("promotion_gate", "c" * 64), ("full_readiness", "d" * 64)):
        sources[name]["sha256"] = supervisor.file_sha256(Path(sources[name]["path"]))
    report["readiness_sha256"] = sources["full_readiness"]["sha256"]
    receipt.write_text(json.dumps(report), encoding="utf-8")
    evidence = supervisor.validate_full_launch_receipt(
        report,
        receipt_path=receipt,
        expected_receipt_sha256=file_sha256(receipt),
        expected_readiness_sha256=sources["full_readiness"]["sha256"],
        expected_scaling_gate_sha256=sources["promotion_gate"]["sha256"],
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        checkpoint_root=tmp_path,
    )
    assert evidence["source_count"] == 9

    Path(sources["dense_config"]["path"]).write_text("replaced", encoding="ascii")
    with pytest.raises(ValueError, match="dense_config"):
        supervisor.validate_full_launch_receipt(
            report,
            receipt_path=receipt,
            expected_receipt_sha256=file_sha256(receipt),
            expected_readiness_sha256=sources["full_readiness"]["sha256"],
            expected_scaling_gate_sha256=sources["promotion_gate"]["sha256"],
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            checkpoint_root=tmp_path,
        )


def test_completion_reuse_requires_exact_expectations() -> None:
    expected = {
        "decision_sha256": "a" * 64,
        "full_training_revision": TRAINING_REVISION,
    }
    report = {
        "profile": supervisor.COMPLETION_PROFILE,
        "status": "pass",
        "complete": True,
        "failed_checks": [],
        "missing_checks": [],
        "checks": [{"name": "all", "status": "pass"}],
        "expectations": dict(expected),
    }
    assert supervisor.validate_completion(report, expected=expected)["check_count"] == 1
    report["expectations"]["full_training_revision"] = "f" * 40
    with pytest.raises(ValueError, match="expectations differ"):
        supervisor.validate_completion(report, expected=expected)


def test_stage_runner_stops_on_nonretryable_audit_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "completion.json"
    report.write_text(
        json.dumps({"status": "failed", "complete": False}),
        encoding="utf-8",
    )
    calls: list[dict] = []

    class Child:
        pid = 19

        @staticmethod
        def wait() -> int:
            return 1

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: Child())
    with pytest.raises(RuntimeError, match="nonretryable report"):
        supervisor._run_stage(
            name="completion_audit",
            runbook=tmp_path / "audit.sh",
            project=tmp_path,
            environment={},
            max_attempts=3,
            retry_seconds=1.0,
            write_status=lambda **kwargs: calls.append(kwargs),
            nonretryable_report=report,
        )
    assert len(calls) == 2
    assert calls[-1]["detail"] == "completion_audit_nonretryable_report"


def test_posttraining_supervisor_runbook_cannot_launch_training() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/"
        "generation_stability_ema_teacher_posttraining_supervisor.sh"
    ).read_text(encoding="utf-8")

    assert "EXPECTED_FULL_LAUNCH_RECEIPT_SHA256=${" in source
    assert "run_generation_stability_posttraining_supervisor.py" in source
    assert "generation_stability_ema_teacher_full_posteval_50k.sh" in source
    assert "generation_stability_ema_teacher_export_inference_artifacts.sh" in source
    assert "generation_stability_ema_teacher_completion_audit.sh" in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source
    assert "train_generation.py" not in source
