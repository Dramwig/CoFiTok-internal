from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from cofitok.reporting import file_sha256
from scripts.audit_generation_stability_completion import (
    aggregate_checks,
    gate_evidence,
    monitor_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/"
    "generation_stability_ema_teacher_completion_audit.sh"
)


def _completed_monitor() -> dict:
    revision = "a" * 40
    branch = "scale/generation-stability-full"
    runs = {}
    for method in ("cofitok", "dense_identity"):
        runs[method] = {
            "complete": True,
            "expected_steps": 300_000,
            "last_step": 300_000,
            "metric_rows": 6_000,
            "health_issues": [],
            "run_manifest": {
                "status": "verified",
                "runtime_environment_sha256": "b" * 64,
                "dataset_identity_sha256": "c" * 64,
            },
            "checkpoint_integrity": {
                "policy": "required",
                "expected_checkpoint_revision": revision,
                "manifests": [
                    {"step": 300_000, "status": "metadata_verified"}
                ],
                "latest_binding": {"status": "metadata_verified"},
            },
        }
    return {
        "schema_version": 2,
        "monitor": "generation_stability_ema_teacher_full_matched_300k",
        "status": "pass",
        "stage": "complete",
        "issues": [],
        "git": {
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
        "runs": runs,
    }


def test_stability_completion_aggregate_is_fail_closed() -> None:
    passed = {"name": "a", "status": "pass", "evidence": {}}
    missing = {"name": "b", "status": "missing", "evidence": None}
    failed = {"name": "c", "status": "fail", "evidence": None}

    assert aggregate_checks([passed])["status"] == "pass"
    assert aggregate_checks([passed, missing])["status"] == "incomplete"
    report = aggregate_checks([passed, missing, failed])
    assert report["status"] == "failed"
    assert report["complete"] is False
    assert report["failed_checks"] == ["c"]
    assert report["missing_checks"] == ["b"]


def test_stability_completion_monitor_requires_exact_clean_identity() -> None:
    report = _completed_monitor()
    evidence = monitor_evidence(
        report,
        expected_name="generation_stability_ema_teacher_full_matched_300k",
        expected_steps=300_000,
        expected_revision="a" * 40,
        expected_branch="scale/generation-stability-full",
    )
    assert evidence["runs"]["cofitok"]["last_step"] == 300_000

    report["git"]["tracked_dirty"] = True
    try:
        monitor_evidence(
            report,
            expected_name=(
                "generation_stability_ema_teacher_full_matched_300k"
            ),
            expected_steps=300_000,
            expected_revision="a" * 40,
            expected_branch="scale/generation-stability-full",
        )
    except ValueError as error:
        assert "Git identity" in str(error)
    else:
        raise AssertionError("dirty stability completion monitor was accepted")


def test_stability_gate_rejects_wrong_profile_before_authorization(
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    gate = {
        "source_profile": "full",
        "provenance_contract": {
            "training_revision": "a" * 40,
            "training_branch": "train",
            "evaluation_revision": "b" * 40,
            "evaluation_branch": "eval",
        },
    }
    try:
        gate_evidence(
            gate,
            gate_path=gate_path,
            expected_sha256=file_sha256(gate_path),
            stage="full",
            source_profile="stability_full",
            training_revision="a" * 40,
            training_branch="train",
            evaluation_revision="b" * 40,
            evaluation_branch="eval",
        )
    except ValueError as error:
        assert "source profile" in str(error)
    else:
        raise AssertionError("wrong stability gate profile was accepted")


def test_stability_completion_runbook_is_a_read_only_exact_identity_gate() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "scripts/audit_generation_stability_completion.py" in source
    assert "EXPECTED_DECISION_SHA256=${" in source
    assert "EXPECTED_SCALING_GATE_SHA256=${" in source
    assert "EXPECTED_FINAL_GATE_SHA256=${" in source
    assert "EXPECTED_FULL_TRAINING_REVISION=${" in source
    assert "EXPECTED_FULL_EVALUATION_REVISION=${" in source
    assert "EXPECTED_AUDIT_REVISION=${" in source
    assert "scripts/train_generation.py" not in source
    assert "scripts/generate_samples.py" not in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source


def test_stability_completion_cli_reports_empty_workspace_as_incomplete(
    tmp_path: Path,
) -> None:
    output = tmp_path / "completion.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "scripts/audit_generation_stability_completion.py"),
        "--project-root",
        str(ROOT),
        "--output-root",
        str(tmp_path / "empty"),
        "--expected-decision-sha256",
        "a" * 64,
        "--expected-scaling-training-revision",
        "b" * 40,
        "--expected-scaling-training-branch",
        "scaling-train",
        "--expected-scaling-evaluation-revision",
        "c" * 40,
        "--expected-scaling-evaluation-branch",
        "scaling-eval",
        "--expected-scaling-gate-sha256",
        "d" * 64,
        "--expected-full-training-revision",
        "e" * 40,
        "--expected-full-training-branch",
        "full-train",
        "--expected-full-evaluation-revision",
        "f" * 40,
        "--expected-full-evaluation-branch",
        "full-eval",
        "--expected-final-gate-sha256",
        "1" * 64,
        "--expected-export-revision",
        "2" * 40,
        "--expected-export-branch",
        "export",
        "--output",
        str(output),
        "--allow-incomplete",
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert report["complete"] is False
    assert report["failed_checks"] == []
    assert len(report["missing_checks"]) == 13
