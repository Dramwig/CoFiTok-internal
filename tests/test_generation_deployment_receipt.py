from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.write_generation_deployment_receipt import build_deployment_receipt


TEN_REVISION = "a" * 40
TARGET_REVISION = "b" * 40
ROOT = Path(__file__).resolve().parents[1]


def _kwargs() -> dict:
    return {
        "expected_training_revision": TEN_REVISION,
        "target_revision": TARGET_REVISION,
        "bundle_path": "/tmp/upgrade.bundle",
        "bundle_bytes": 1234,
        "bundle_sha256": "c" * 64,
        "bundle_heads": [TARGET_REVISION],
        "pair_validation_path": "/outputs/pair.json",
        "pair_validation_sha256": "d" * 64,
        "pair_validation": {
            "status": "pass",
            "expected_revision": TEN_REVISION,
        },
        "conflict_scan_path": "/outputs/conflict.json",
        "conflict_scan_bytes": 234,
        "conflict_scan_sha256": "1" * 64,
        "conflict_scan": {
            "schema_version": 1,
            "status": "pass",
            "current_commit": TEN_REVISION,
            "target_commit": TARGET_REVISION,
            "target_added_path_count": 12,
            "conflict_count": 0,
            "conflicts": [],
        },
        "runbook_syntax_path": "/outputs/runbooks.json",
        "runbook_syntax_bytes": 345,
        "runbook_syntax_sha256": "2" * 64,
        "runbook_syntax": {
            "schema_version": 1,
            "role": "generation_runbook_syntax_check",
            "status": "pass",
            "enumeration": "git_ls_files",
            "discovered_count": 2,
            "checked_count": 2,
            "failed_count": 0,
            "runbooks": [
                "artifacts/runbooks/a.sh",
                "artifacts/runbooks/b.sh",
            ],
            "failures": [],
            "git": {
                "revision": TARGET_REVISION,
                "branch": "scale/generative-system",
                "tracked_dirty": False,
            },
        },
        "pytest_report_path": "/outputs/pytest.xml",
        "pytest_report_bytes": 456,
        "pytest_report_sha256": "3" * 64,
        "pytest_summary": {
            "status": "pass",
            "tests": 530,
            "failures": 0,
            "errors": 0,
            "skipped": 1,
        },
        "git_revision": TARGET_REVISION,
        "git_branch": "scale/generative-system",
        "tracked_dirty": False,
        "hostname": "pro6000",
        "deployed_at": "2026-07-12T00:00:00+00:00",
    }


def test_deployment_receipt_binds_bundle_pair_and_revisions() -> None:
    receipt = build_deployment_receipt(**_kwargs())

    assert receipt["status"] == "pass"
    assert receipt["bundle"]["sha256"] == "c" * 64
    assert receipt["training_pair_validation"]["sha256"] == "d" * 64
    assert receipt["schema_version"] == 2
    assert receipt["verification"]["untracked_target_conflicts"]["conflict_count"] == 0
    assert receipt["verification"]["runbook_syntax"]["checked_count"] == 2
    assert receipt["verification"]["pytest"]["tests"] == 530
    assert receipt["git"]["revision"] == TARGET_REVISION


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.update(git_revision="e" * 40), "target revision"),
        (lambda values: values.update(tracked_dirty=True), "dirty tracked worktree"),
        (
            lambda values: values["pair_validation"].update(status="failed"),
            "did not pass",
        ),
        (
            lambda values: values["pair_validation"].update(
                expected_revision="f" * 40
            ),
            "differs from the pinned queue",
        ),
        (
            lambda values: values.update(bundle_heads=["f" * 40]),
            "does not contain the target revision",
        ),
        (
            lambda values: values["conflict_scan"].update(conflict_count=1),
            "conflict scan",
        ),
        (
            lambda values: values["runbook_syntax"].update(checked_count=1),
            "runbook syntax",
        ),
        (
            lambda values: values["pytest_summary"].update(failures=1),
            "pytest report",
        ),
    ],
)
def test_deployment_receipt_rejects_untrusted_transition(mutation, message) -> None:
    values = _kwargs()
    mutation(values)

    with pytest.raises(ValueError, match=message):
        build_deployment_receipt(**values)


def test_deployment_receipt_cli_verifies_real_bundle_head(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "-b", "scale/generative-system"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "cofitok@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CoFiTok Test"],
        cwd=repository,
        check=True,
    )
    (repository / "tracked.txt").write_text("target\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "target"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bundle = tmp_path / "upgrade.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    pair_validation = tmp_path / "pair.json"
    pair_validation.write_text(
        json.dumps({"status": "pass", "expected_revision": TEN_REVISION}),
        encoding="utf-8",
    )
    conflict_scan = tmp_path / "conflict.json"
    conflict_scan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "current_commit": TEN_REVISION,
                "target_commit": revision,
                "target_added_path_count": 1,
                "conflict_count": 0,
                "conflicts": [],
            }
        ),
        encoding="utf-8",
    )
    runbook_syntax = tmp_path / "runbooks.json"
    runbook_syntax.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "generation_runbook_syntax_check",
                "status": "pass",
                "enumeration": "git_ls_files",
                "discovered_count": 1,
                "checked_count": 1,
                "failed_count": 0,
                "runbooks": ["artifacts/runbooks/test.sh"],
                "failures": [],
                "git": {
                    "revision": revision,
                    "branch": "scale/generative-system",
                    "tracked_dirty": False,
                },
            }
        ),
        encoding="utf-8",
    )
    pytest_report = tmp_path / "pytest.xml"
    pytest_report.write_text(
        '<testsuites><testsuite tests="3" failures="0" errors="0" '
        'skipped="1" /></testsuites>',
        encoding="utf-8",
    )
    output = tmp_path / "receipt.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/write_generation_deployment_receipt.py"),
            "--output",
            str(output),
            "--bundle",
            str(bundle),
            "--pair-validation",
            str(pair_validation),
            "--conflict-scan",
            str(conflict_scan),
            "--runbook-syntax",
            str(runbook_syntax),
            "--pytest-report",
            str(pytest_report),
            "--expected-training-revision",
            TEN_REVISION,
            "--target-revision",
            revision,
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["bundle"]["heads"] == [revision]
    assert receipt["target_revision"] == revision
    assert receipt["verification"]["pytest"]["tests"] == 3
    assert receipt["verification"]["runbook_syntax"]["checked_count"] == 1
